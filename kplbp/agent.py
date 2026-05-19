from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .models import PolicyModel, ValueModel
from .schema import BPState


class BPAgent:
    def __init__(
        self,
        policy_model: PolicyModel,
        value_model: ValueModel,
        schedule: list[dict[str, int | str]],
        heroes: Iterable[str],
    ) -> None:
        self.policy_model = policy_model
        self.value_model = value_model
        self.schedule = sorted(schedule, key=lambda item: int(item["order"]))
        self.heroes = sorted(set(heroes))

    def recommend(
        self,
        state: BPState,
        top_k: int = 5,
        policy_width: int = 12,
        search_depth: int = 2,
        legal_heroes: Iterable[str] | None = None,
    ) -> list[dict[str, float | str]]:
        hero_pool = sorted(set(legal_heroes if legal_heroes is not None else self.heroes))
        candidates = self.policy_model.predict(state, top_k=policy_width, legal_heroes=hero_pool)
        scored = []
        for hero, policy_score in candidates:
            next_state = self.apply_action(state, hero)
            search_score = self._search(
                next_state,
                perspective_camp=state.camp,
                depth=max(0, search_depth - 1),
                width=min(6, policy_width),
                legal_heroes=hero_pool,
            )
            combined = 0.65 * policy_score + 0.35 * search_score
            scored.append(
                {
                    "hero": hero,
                    "policy_score": round(policy_score, 6),
                    "value_score": round(search_score, 6),
                    "score": round(combined, 6),
                }
            )
        return sorted(scored, key=lambda item: float(item["score"]), reverse=True)[:top_k]

    def apply_action(self, state: BPState, hero: str) -> BPState:
        banned = list(state.banned)
        camp1_picks = list(state.camp1_picks)
        camp2_picks = list(state.camp2_picks)
        if state.action_type == "ban":
            banned.append(hero)
        elif state.camp == 1:
            camp1_picks.append(hero)
        else:
            camp2_picks.append(hero)

        next_slot = self._next_slot(state.order)
        if next_slot is None:
            return replace(
                state,
                order=state.order + 1,
                banned=banned,
                camp1_picks=camp1_picks,
                camp2_picks=camp2_picks,
            )
        return BPState(
            match_id=state.match_id,
            battle_id=state.battle_id,
            order=int(next_slot["order"]),
            camp=int(next_slot["camp"]),
            action_type=str(next_slot["action_type"]),
            banned=banned,
            camp1_picks=camp1_picks,
            camp2_picks=camp2_picks,
            camp1_team=state.camp1_team,
            camp2_team=state.camp2_team,
        )

    def _next_slot(self, order: int) -> dict[str, int | str] | None:
        for slot in self.schedule:
            if int(slot["order"]) > order:
                return slot
        return None

    def _state_value(self, state: BPState, perspective_camp: int) -> float:
        if perspective_camp == 1:
            own_picks = state.camp1_picks
            enemy_picks = state.camp2_picks
        else:
            own_picks = state.camp2_picks
            enemy_picks = state.camp1_picks
        return self.value_model.predict(
            own_picks,
            enemy_picks,
            state.own_team(),
            state.enemy_team(),
            include_strength=False,
        )

    def _search(
        self,
        state: BPState,
        perspective_camp: int,
        depth: int,
        width: int,
        legal_heroes: Iterable[str] | None = None,
    ) -> float:
        if depth <= 0 or state.order >= len(self.schedule):
            return self._state_value(state, perspective_camp)

        candidates = self.policy_model.predict(
            state,
            top_k=width,
            legal_heroes=legal_heroes if legal_heroes is not None else self.heroes,
        )
        if not candidates:
            return self._state_value(state, perspective_camp)

        child_scores = [
            self._search(
                self.apply_action(state, hero),
                perspective_camp,
                depth - 1,
                width,
                legal_heroes,
            )
            for hero, _ in candidates
        ]
        if state.camp == perspective_camp:
            return max(child_scores)
        return min(child_scores)

    def to_dict(self) -> dict[str, object]:
        return {
            "heroes": self.heroes,
            "schedule": self.schedule,
            "policy_model": self.policy_model.to_dict(),
            "value_model": self.value_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "BPAgent":
        return cls(
            policy_model=PolicyModel.from_dict(data["policy_model"]),
            value_model=ValueModel.from_dict(data["value_model"]),
            schedule=data["schedule"],
            heroes=data["heroes"],
        )
