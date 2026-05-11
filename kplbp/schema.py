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
