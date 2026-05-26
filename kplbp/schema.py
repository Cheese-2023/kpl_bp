from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BPStep:
    match_id: str
    battle_id: str
    order: int
    camp: int
    action_type: str
    hero: str


@dataclass(frozen=True)
class PlayerRow:
    match_id: str
    battle_id: str
    team: str
    player: str
    hero: str
    kill: int
    death: int
    assist: int
    kda: float
    gold: int
    win: int


@dataclass(frozen=True)
class HeroStats:
    league_id: str
    hero_name: str
    battle_count: int
    win_rate: float
    avg_kda: float
    ban_rate: float
    pick_rate: float


@dataclass(frozen=True)
class HeroMeta:
    hero: str
    lane: str
    lanes: tuple[str, ...]
    role: str
    damage_type: str
    tags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hero": self.hero,
            "lane": self.lane,
            "lanes": list(self.lanes),
            "role": self.role,
            "damage_type": self.damage_type,
            "tags": list(self.tags),
        }


@dataclass
class BPState:
    match_id: str
    battle_id: str
    order: int
    camp: int
    action_type: str
    banned: list[str] = field(default_factory=list)
    camp1_picks: list[str] = field(default_factory=list)
    camp2_picks: list[str] = field(default_factory=list)
    camp1_team: str | None = None
    camp2_team: str | None = None

    def used_heroes(self) -> set[str]:
        return set(self.banned) | set(self.camp1_picks) | set(self.camp2_picks)

    def own_picks(self) -> list[str]:
        return self.camp1_picks if self.camp == 1 else self.camp2_picks

    def enemy_picks(self) -> list[str]:
        return self.camp2_picks if self.camp == 1 else self.camp1_picks

    def own_team(self) -> str | None:
        return self.camp1_team if self.camp == 1 else self.camp2_team

    def enemy_team(self) -> str | None:
        return self.camp2_team if self.camp == 1 else self.camp1_team

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "battle_id": self.battle_id,
            "order": self.order,
            "camp": self.camp,
            "action_type": self.action_type,
            "banned": list(self.banned),
            "camp1_picks": list(self.camp1_picks),
            "camp2_picks": list(self.camp2_picks),
            "camp1_team": self.camp1_team,
            "camp2_team": self.camp2_team,
        }


@dataclass(frozen=True)
class BPSample:
    state: BPState
    label: str


@dataclass(frozen=True)
class LineupSample:
    battle_id: str
    camp: int
    own_picks: tuple[str, ...]
    enemy_picks: tuple[str, ...]
    win: int
    team: str | None = None
    enemy_team: str | None = None
    players: tuple[str, ...] = ()
    enemy_players: tuple[str, ...] = ()


@dataclass
class PlayerProfile:
    name: str = ""
    lane: str = ""
    style: str = ""
    tier: str = ""
    preferred_heroes: list[str] = field(default_factory=list)
    avoid_heroes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lane": self.lane,
            "style": self.style,
            "tier": self.tier,
            "preferred_heroes": list(self.preferred_heroes),
            "avoid_heroes": list(self.avoid_heroes),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> PlayerProfile:
        if not isinstance(raw, dict):
            return cls()
        preferred = raw.get("preferred_heroes") or []
        avoid = raw.get("avoid_heroes") or []
        return cls(
            name=str(raw.get("name", "") or "").strip(),
            lane=str(raw.get("lane", "") or "").strip(),
            style=str(raw.get("style", "") or "").strip(),
            tier=str(raw.get("tier", "") or "").strip(),
            preferred_heroes=[str(item).strip() for item in preferred if str(item).strip()],
            avoid_heroes=[str(item).strip() for item in avoid if str(item).strip()],
        )


@dataclass
class TeamProfile:
    camp: int = 1
    team_name: str = ""
    players: list[PlayerProfile] = field(default_factory=list)
    overall_style: str = ""
    win_condition: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "camp": self.camp,
            "team_name": self.team_name,
            "players": [player.to_dict() for player in self.players],
            "overall_style": self.overall_style,
            "win_condition": self.win_condition,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> TeamProfile:
        if not isinstance(raw, dict):
            return cls()
        players_raw = raw.get("players") or []
        players = [
            PlayerProfile.from_dict(item)
            for item in players_raw
            if isinstance(item, dict)
        ]
        return cls(
            camp=int(raw.get("camp", 1) or 1),
            team_name=str(raw.get("team_name", "") or "").strip(),
            players=players,
            overall_style=str(raw.get("overall_style", "") or "").strip(),
            win_condition=str(raw.get("win_condition", "") or "").strip(),
        )

    def is_empty(self) -> bool:
        if self.team_name or self.overall_style or self.win_condition:
            return False
        return not any(
            player.name
            or player.lane
            or player.style
            or player.tier
            or player.preferred_heroes
            or player.avoid_heroes
            for player in self.players
        )

    def to_prompt_text(self) -> str:
        if self.is_empty():
            return "未提供队伍画像。"
        side = "蓝方" if self.camp == 1 else "红方"
        lines = [f"队伍画像（{side} / Camp {self.camp}）："]
        if self.team_name:
            lines.append(f"- 队伍名称：{self.team_name}")
        for player in self.players:
            if not (
                player.name
                or player.lane
                or player.style
                or player.tier
                or player.preferred_heroes
                or player.avoid_heroes
            ):
                continue
            label = player.name or "未命名选手"
            lane = f"（{player.lane}）" if player.lane else ""
            style = f"，打法风格：{player.style}" if player.style else ""
            tier = f"，当前状态：{player.tier}" if player.tier else ""
            preferred = (
                f"，擅长英雄：{'/'.join(player.preferred_heroes)}"
                if player.preferred_heroes
                else ""
            )
            avoid = (
                f"，回避英雄：{'/'.join(player.avoid_heroes)}"
                if player.avoid_heroes
                else ""
            )
            lines.append(f"  - {label}{lane}{style}{tier}{preferred}{avoid}")
        if self.overall_style:
            lines.append(f"- 整体风格：{self.overall_style}")
        if self.win_condition:
            lines.append(f"- 胜利条件偏好：{self.win_condition}")
        return "\n".join(lines)


def team_profiles_from_payload(payload: dict[str, object]) -> tuple[TeamProfile, TeamProfile]:
    camp1 = TeamProfile.from_dict(payload.get("camp1_profile") if isinstance(payload.get("camp1_profile"), dict) else None)
    camp2 = TeamProfile.from_dict(payload.get("camp2_profile") if isinstance(payload.get("camp2_profile"), dict) else None)
    if camp1.is_empty() and isinstance(payload.get("team_profile"), dict):
        legacy = TeamProfile.from_dict(payload.get("team_profile"))  # type: ignore[arg-type]
        if legacy.camp == 1:
            camp1 = legacy
        elif legacy.camp == 2:
            camp2 = legacy
    camp1_name = str(payload.get("camp1_team", "") or "").strip()
    camp2_name = str(payload.get("camp2_team", "") or "").strip()
    if camp1_name and not camp1.team_name:
        camp1.team_name = camp1_name
    if camp2_name and not camp2.team_name:
        camp2.team_name = camp2_name
    camp1.camp = 1
    camp2.camp = 2
    return camp1, camp2


def team_profiles_prompt_text(camp1: TeamProfile, camp2: TeamProfile) -> str:
    sections = [camp1.to_prompt_text(), camp2.to_prompt_text()]
    if camp1.is_empty() and camp2.is_empty():
        return "未提供队伍画像。"
    return "\n\n".join(sections) + "\n\n请结合以上队伍画像，调整推荐优先级并解释选手风格/状态带来的取舍。"
