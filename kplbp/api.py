from __future__ import annotations

import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import request
from urllib.parse import parse_qs, urlparse

from .analysis import analyze_lineup, answer_question, compare_lineups, split_hero_names
from .agent import BPAgent
from .bp_rules import (
    legal_heroes_for_state,
    modes_payload,
    rerank_for_global_bp,
    schedule_for_mode,
)
from .data import load_hero_meta, load_json
from .schema import BPState


ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / "web"


class KPLBPHandler(SimpleHTTPRequestHandler):
    hero_meta_path = ROOT / "hero_meta.csv"
    model_path = ROOT / "models" / "kpl_bp_agent.json"
    bp_knowledge_path = ROOT / "docs" / "BP_EXPERIENCE.md"
    deepseek_model = "deepseek-chat"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"ok": True})
            return
        if parsed.path == "/api/heroes":
            self._send_json(self._heroes_payload())
            return
        if parsed.path == "/api/analyze":
            query = parse_qs(parsed.query)
            heroes = split_hero_names(query.get("heroes", [""])[0])
            enemy_heroes = split_hero_names(query.get("enemy_heroes", [""])[0])
            self._send_json(self._analyze_payload(heroes, enemy_heroes))
            return
        if parsed.path == "/api/bp/config":
            self._send_json(self._bp_config_payload())
            return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {
            "/api/analyze",
            "/api/ask",
            "/api/bp/recommend",
            "/api/deepseek",
        }:
            self._send_json({"error": "unknown endpoint"}, status=404)
            return
        payload = self._read_json()
        own_heroes = split_hero_names(payload.get("heroes") or payload.get("own_heroes"))
        enemy_heroes = split_hero_names(payload.get("enemy_heroes"))
        if parsed.path == "/api/analyze":
            self._send_json(self._analyze_payload(own_heroes, enemy_heroes))
            return
        question = str(payload.get("question", "")).strip()
        if parsed.path == "/api/ask":
            hero_meta = load_hero_meta(self.hero_meta_path)
            self._send_json(answer_question(question, own_heroes, enemy_heroes, hero_meta))
            return
        if parsed.path == "/api/bp/recommend":
            self._send_json(self._bp_recommend_payload(payload))
            return
        self._send_json(self._deepseek_payload(payload))

    def _heroes_payload(self) -> dict[str, object]:
        hero_meta = load_hero_meta(self.hero_meta_path)
        heroes = [meta.to_dict() for meta in sorted(hero_meta.values(), key=lambda item: item.hero)]
        lanes = sorted({lane for meta in hero_meta.values() for lane in meta.lanes})
        roles = sorted({meta.role for meta in hero_meta.values()})
        damage_types = sorted({meta.damage_type for meta in hero_meta.values()})
        tags = sorted({tag for meta in hero_meta.values() for tag in meta.tags})
        return {
            "heroes": heroes,
            "lanes": lanes,
            "roles": roles,
            "damage_types": damage_types,
            "tags": tags,
        }

    def _analyze_payload(self, heroes: list[str], enemy_heroes: list[str]) -> dict[str, object]:
        hero_meta = load_hero_meta(self.hero_meta_path)
        if enemy_heroes:
            return compare_lineups(heroes, enemy_heroes, hero_meta)
        return analyze_lineup(heroes, hero_meta).to_dict()

    def _bp_config_payload(self) -> dict[str, object]:
        agent = self._load_agent()
        hero_meta = load_hero_meta(self.hero_meta_path)
        return {
            "heroes": self._heroes_payload()["heroes"],
            "hero_names": agent.heroes,
            "schedule": agent.schedule,
            "modes": modes_payload(),
            "lanes": sorted({lane for meta in hero_meta.values() for lane in meta.lanes}),
            "tags": sorted({tag for meta in hero_meta.values() for tag in meta.tags}),
        }

    def _bp_recommend_payload(self, payload: dict[str, object]) -> dict[str, object]:
        agent = self._load_agent()
        state = self._state_from_payload(payload, agent)
        schedule = self._schedule_from_payload(payload, agent)
        agent_for_mode = BPAgent(agent.policy_model, agent.value_model, schedule, agent.heroes)
        top_k = int(payload.get("top_k", 6) or 6)
        search_depth = int(payload.get("search_depth", 2) or 2)
        legal_heroes = self._legal_heroes(payload, state, agent)
        recommendations = agent_for_mode.recommend(
            state,
            top_k=top_k,
            search_depth=search_depth,
            legal_heroes=legal_heroes,
        )
        recommendations = self._strategy_adjusted_recommendations(
            recommendations,
            payload,
            state,
            agent,
        )[:top_k]
        return {
            "state": state.to_dict(),
            "schedule": schedule,
            "legal_heroes": legal_heroes,
            "recommendations": recommendations,
            "done": state.order >= len(schedule),
        }

    def _state_from_payload(self, payload: dict[str, object], agent: BPAgent) -> BPState:
        if "state" in payload and isinstance(payload["state"], dict):
            raw = payload["state"]
            return BPState(
                match_id=str(raw.get("match_id", "manual")),
                battle_id=str(raw.get("battle_id", "manual")),
                order=int(raw.get("order", 0)),
                camp=int(raw.get("camp", 1)),
                action_type=str(raw.get("action_type", "ban")),
                banned=split_hero_names(raw.get("banned")),
                camp1_picks=split_hero_names(raw.get("camp1_picks")),
                camp2_picks=split_hero_names(raw.get("camp2_picks")),
                camp1_team=raw.get("camp1_team"),
                camp2_team=raw.get("camp2_team"),
            )
        schedule = self._schedule_from_payload(payload, agent)
        actions = payload.get("actions", [])
        if not isinstance(actions, list):
            actions = []
        camp1_team = payload.get("camp1_team")
        camp2_team = payload.get("camp2_team")
        banned: list[str] = []
        camp1_picks: list[str] = []
        camp2_picks: list[str] = []
        applied = 0
        for action in actions[: len(schedule)]:
            if not isinstance(action, dict):
                continue
            hero = str(action.get("hero", "")).strip()
            if not hero:
                continue
            slot = schedule[applied]
            if str(slot["action_type"]) == "ban":
                banned.append(hero)
            elif int(slot["camp"]) == 1:
                camp1_picks.append(hero)
            else:
                camp2_picks.append(hero)
            applied += 1
        if applied >= len(schedule):
            last_slot = schedule[-1]
            order = len(schedule)
            camp = int(last_slot["camp"])
            action_type = str(last_slot["action_type"])
        else:
            slot = schedule[applied]
            order = int(slot["order"])
            camp = int(slot["camp"])
            action_type = str(slot["action_type"])
        return BPState(
            match_id="manual",
            battle_id="manual",
            order=order,
            camp=camp,
            action_type=action_type,
            banned=banned,
            camp1_picks=camp1_picks,
            camp2_picks=camp2_picks,
            camp1_team=str(camp1_team) if camp1_team else None,
            camp2_team=str(camp2_team) if camp2_team else None,
        )

    def _schedule_from_payload(
        self,
        payload: dict[str, object],
        agent: BPAgent,
    ) -> list[dict[str, int | str]]:
        mode = str(payload.get("mode", "single") or "single")
        game_index = int(payload.get("game_index", 1) or 1)
        schedule = schedule_for_mode(agent.schedule, mode, game_index)
        return sorted(schedule, key=lambda item: int(item["order"]))

    def _legal_heroes(
        self,
        payload: dict[str, object],
        state: BPState,
        agent: BPAgent,
    ) -> list[str]:
        mode = str(payload.get("mode", "single") or "single")
        game_index = int(payload.get("game_index", 1) or 1)
        return legal_heroes_for_state(
            agent.heroes,
            state.order,
            state.camp,
            state.action_type,
            mode,
            game_index,
            state.banned,
            state.camp1_picks,
            state.camp2_picks,
            split_hero_names(payload.get("camp1_global_used")),
            split_hero_names(payload.get("camp2_global_used")),
        )

    def _deepseek_payload(self, payload: dict[str, object]) -> dict[str, object]:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        agent = self._load_agent()
        state = self._state_from_payload(payload, agent)
        schedule = self._schedule_from_payload(payload, agent)
        agent_for_mode = BPAgent(agent.policy_model, agent.value_model, schedule, agent.heroes)
        own_heroes = split_hero_names(payload.get("heroes") or state.own_picks())
        enemy_heroes = split_hero_names(payload.get("enemy_heroes") or state.enemy_picks())
        hero_meta = load_hero_meta(self.hero_meta_path)
        local_answer = answer_question(
            str(payload.get("question", "请分析当前 BP 和阵容思路")),
            own_heroes,
            enemy_heroes,
            hero_meta,
        )
        recommendations = agent_for_mode.recommend(
            state,
            top_k=6,
            search_depth=2,
            legal_heroes=self._legal_heroes(payload, state, agent),
        )
        recommendations = self._strategy_adjusted_recommendations(
            recommendations,
            payload,
            state,
            agent,
        )[:6]
        prompt = self._deepseek_prompt(payload, state, local_answer, recommendations)
        if not api_key:
            return {
                "error": "DEEPSEEK_API_KEY is not set",
                "prompt": prompt,
                "local_answer": local_answer,
                "recommendations": recommendations,
            }

        body = json.dumps(
            {
                "model": self.deepseek_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是 KPL 职业赛 BP 教练。只基于用户给出的 BP 状态、英雄标签、本地模型推荐、阵容分析和 BP 经验知识库回答。回答必须具体，必须分别讨论蓝方和红方，不要只给空泛建议，不要编造不存在的数据。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 1200,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = request.Request(
            "https://api.deepseek.com/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return {
                "error": str(exc),
                "prompt": prompt,
                "local_answer": local_answer,
                "recommendations": recommendations,
            }
        content = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return {
            "answer": content,
            "prompt": prompt,
            "local_answer": local_answer,
            "recommendations": recommendations,
            "raw": result,
        }

    def _deepseek_prompt(
        self,
        payload: dict[str, object],
        state: BPState,
        local_answer: dict[str, object],
        recommendations: list[dict[str, object]],
    ) -> str:
        question = str(payload.get("question", "请分析当前 BP 和阵容思路")).strip()
        knowledge = self._load_bp_knowledge()
        return "\n".join(
            [
                "请作为 KPL BP 教练分析下面局面。",
                "",
                f"用户问题：{question}",
                f"当前 BP 状态：{json.dumps(state.to_dict(), ensure_ascii=False)}",
                f"本地模型推荐：{json.dumps(recommendations, ensure_ascii=False)}",
                f"英雄标签阵容分析：{json.dumps(local_answer['analysis'], ensure_ascii=False)}",
                f"BP 经验知识库：\n{knowledge}",
                "",
                "请严格按以下结构回答，不能省略蓝方或红方：",
                "1. 当前 BP 关键矛盾：一句话说明现在争夺的是线权、开团、保护、后期输出、全局 BP 资源还是特定英雄克制。",
                "2. 蓝方阵容分析：列出蓝方强点、短板、当前最怕什么。",
                "3. 红方阵容分析：列出红方强点、短板、当前最怕什么。",
                "4. 下一手建议：结合本地模型推荐，给出 2-3 个候选，并说明为什么选、为什么不选其他高分英雄。",
                "5. 全局 BP 资源判断：如果是 BO3/BO5/BO7，说明这手是否值得消耗高优先级英雄，是否应该保留给后续局。",
                "6. 进入游戏后的思路：分别写蓝方前期/中期/后期，以及红方前期/中期/后期。",
                "7. 风险提醒：列出双方最容易输掉比赛的 2-3 个风险点。",
            ]
        )

    def _load_bp_knowledge(self) -> str:
        try:
            text = self.bp_knowledge_path.read_text(encoding="utf-8")
        except OSError:
            return "未找到 BP 经验知识库。"
        return text[:6000]

    def _strategy_adjusted_recommendations(
        self,
        recommendations: list[dict[str, object]],
        payload: dict[str, object],
        state: BPState,
        agent: BPAgent,
    ) -> list[dict[str, object]]:
        hero_priorities = self._hero_priorities(agent)
        return rerank_for_global_bp(
            recommendations,
            str(payload.get("mode", "single") or "single"),
            int(payload.get("game_index", 1) or 1),
            state.action_type,
            hero_priorities,
        )

    def _hero_priorities(self, agent: BPAgent) -> dict[str, float]:
        priorities: dict[str, float] = {}
        raw_values: dict[str, float] = {}
        for hero in agent.heroes:
            stat = agent.policy_model.hero_stats.get(hero)
            if stat is None:
                raw_values[hero] = 0.0
                continue
            raw_values[hero] = 0.45 * stat.pick_rate + 0.35 * stat.ban_rate + 0.20 * stat.win_rate
        if not raw_values:
            return priorities
        min_value = min(raw_values.values())
        max_value = max(raw_values.values())
        span = max(1e-9, max_value - min_value)
        for hero, value in raw_values.items():
            priorities[hero] = (value - min_value) / span
        return priorities

    def _load_agent(self) -> BPAgent:
        data = load_json(self.model_path)
        if not isinstance(data, dict):
            raise ValueError("Model file must contain a JSON object.")
        return BPAgent.from_dict(data)

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve KPL BP frontend and metadata API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hero-meta", default=str(ROOT / "hero_meta.csv"))
    parser.add_argument("--model", default=str(ROOT / "models" / "kpl_bp_agent.json"))
    parser.add_argument("--bp-knowledge", default=str(ROOT / "docs" / "BP_EXPERIENCE.md"))
    parser.add_argument("--deepseek-model", default="deepseek-chat")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    KPLBPHandler.hero_meta_path = Path(args.hero_meta)
    KPLBPHandler.model_path = Path(args.model)
    KPLBPHandler.bp_knowledge_path = Path(args.bp_knowledge)
    KPLBPHandler.deepseek_model = args.deepseek_model
    server = ThreadingHTTPServer((args.host, args.port), KPLBPHandler)
    print(f"KPL BP app running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
