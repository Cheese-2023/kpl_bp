from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .schema import BPSample, BPState, HeroStats, LineupSample


ContextKey = tuple[str, ...]


def _logit(value: float) -> float:
    clipped = max(0.01, min(0.99, value))
    return math.log(clipped / (1 - clipped))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def _context_to_json_key(key: ContextKey) -> str:
    return "\t".join(key)


def _json_key_to_context(key: str) -> ContextKey:
    return tuple(key.split("\t"))


@dataclass
class Evaluation:
    total: int
    top1: float
    top3: float
    top5: float
    legal_rate: float


class PolicyModel:
    """A supervised, context-aware frequency baseline for BP actions."""

    def __init__(
        self,
        heroes: Iterable[str],
        hero_stats: dict[str, HeroStats] | None = None,
    ) -> None:
        self.heroes = sorted(set(heroes))
        self.hero_stats = hero_stats or {}
        self.counts: dict[ContextKey, Counter[str]] = defaultdict(Counter)
        self.hero_priors: Counter[str] = Counter()

    def _contexts(self, state: BPState) -> list[tuple[float, ContextKey]]:
        own_picks = state.own_picks()
        enemy_picks = state.enemy_picks()
        contexts: list[tuple[float, ContextKey]] = [
            (4.0, ("exact", state.action_type, str(state.order), str(state.camp))),
            (2.0, ("order", state.action_type, str(state.order))),
            (1.0, ("action", state.action_type)),
        ]
        for hero in own_picks:
            contexts.append((0.65, ("own", state.action_type, hero)))
        for hero in enemy_picks:
            contexts.append((0.65, ("enemy", state.action_type, hero)))
        contexts.append(
            (
                0.8,
                (
                    "shape",
                    state.action_type,
                    str(len(own_picks)),
                    str(len(enemy_picks)),
                    str(len(state.banned)),
                ),
            )
        )
        return contexts

    def fit(self, samples: Iterable[BPSample]) -> "PolicyModel":
        for sample in samples:
            self.hero_priors[sample.label] += 1
            for _, context in self._contexts(sample.state):
                self.counts[context][sample.label] += 1
        return self

    def predict(
        self,
        state: BPState,
        top_k: int = 10,
        legal_heroes: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        legal = set(legal_heroes if legal_heroes is not None else self.heroes)
        legal -= state.used_heroes()
        if not legal:
            return []

        scores = {hero: 1e-6 for hero in legal}
        for hero in legal:
            prior_count = self.hero_priors[hero]
            stat = self.hero_stats.get(hero)
            if stat is not None:
                if state.action_type == "ban":
                    scores[hero] += 0.25 * stat.ban_rate + 0.05 * stat.pick_rate
                else:
                    scores[hero] += 0.25 * stat.pick_rate + 0.10 * stat.win_rate
            scores[hero] += math.log1p(prior_count) * 0.02

        for weight, context in self._contexts(state):
            counter = self.counts.get(context)
            if not counter:
                continue
            total = sum(counter.values())
            denominator = total + len(self.heroes)
            for hero in legal:
                scores[hero] += weight * ((counter[hero] + 1.0) / denominator)

        total_score = sum(scores.values())
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(hero, score / total_score) for hero, score in ranked[:top_k]]

    def evaluate(self, samples: Iterable[BPSample], top_k: int = 5) -> Evaluation:
        total = 0
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        legal_hits = 0
        for sample in samples:
            total += 1
            ranked = self.predict(sample.state, top_k=max(top_k, 5))
            heroes = [hero for hero, _ in ranked]
            if sample.label not in sample.state.used_heroes():
                legal_hits += 1
            if heroes[:1] and sample.label == heroes[0]:
                top1_hits += 1
            if sample.label in heroes[:3]:
                top3_hits += 1
            if sample.label in heroes[:5]:
                top5_hits += 1
        if total == 0:
            return Evaluation(total=0, top1=0.0, top3=0.0, top5=0.0, legal_rate=0.0)
        return Evaluation(
            total=total,
            top1=top1_hits / total,
            top3=top3_hits / total,
            top5=top5_hits / total,
            legal_rate=legal_hits / total,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "heroes": self.heroes,
            "counts": {
                _context_to_json_key(key): dict(counter)
                for key, counter in self.counts.items()
            },
            "hero_priors": dict(self.hero_priors),
            "hero_stats": {
                hero: {
                    "league_id": stat.league_id,
                    "hero_name": stat.hero_name,
                    "battle_count": stat.battle_count,
                    "win_rate": stat.win_rate,
                    "avg_kda": stat.avg_kda,
                    "ban_rate": stat.ban_rate,
                    "pick_rate": stat.pick_rate,
                }
                for hero, stat in self.hero_stats.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PolicyModel":
        raw_stats = data.get("hero_stats", {})
        hero_stats = {
            hero: HeroStats(
                league_id=str(value["league_id"]),
                hero_name=str(value["hero_name"]),
                battle_count=int(value["battle_count"]),
                win_rate=float(value["win_rate"]),
                avg_kda=float(value["avg_kda"]),
                ban_rate=float(value["ban_rate"]),
                pick_rate=float(value["pick_rate"]),
            )
            for hero, value in raw_stats.items()
        }
        model = cls(heroes=data["heroes"], hero_stats=hero_stats)
        for key, counter in data.get("counts", {}).items():
            model.counts[_json_key_to_context(key)] = Counter(counter)
        model.hero_priors = Counter(data.get("hero_priors", {}))
        return model


class ValueModel:
    """A small empirical win-rate model for evaluating partial or full lineups."""

    def __init__(self, heroes: Iterable[str], hero_stats: dict[str, HeroStats] | None = None) -> None:
        self.heroes = sorted(set(heroes))
        self.hero_stats = hero_stats or {}
        self.base_rate = 0.5
        self.own_rates: dict[str, float] = {}
        self.enemy_rates: dict[str, float] = {}
        self.hero_effects: dict[str, float] = {}

    def fit(self, samples: Iterable[LineupSample]) -> "ValueModel":
        sample_list = list(samples)
        if not sample_list:
            return self
        self.base_rate = sum(sample.win for sample in sample_list) / len(sample_list)
        own_counts: dict[str, list[int]] = defaultdict(list)
        enemy_counts: dict[str, list[int]] = defaultdict(list)
        for sample in sample_list:
            for hero in sample.own_picks:
                own_counts[hero].append(sample.win)
            for hero in sample.enemy_picks:
                enemy_counts[hero].append(sample.win)
        for hero in self.heroes:
            stat = self.hero_stats.get(hero)
            stat_prior = stat.win_rate if stat is not None else self.base_rate
            own = own_counts.get(hero, [])
            enemy = enemy_counts.get(hero, [])
            self.own_rates[hero] = (sum(own) + 4 * stat_prior) / (len(own) + 4)
            self.enemy_rates[hero] = (sum(enemy) + 4 * (1 - stat_prior)) / (len(enemy) + 4)
            self.hero_effects[hero] = _logit(self.own_rates[hero]) - _logit(self.base_rate)
        return self

    def predict(self, own_picks: Iterable[str], enemy_picks: Iterable[str]) -> float:
        own = list(own_picks)
        enemy = list(enemy_picks)
        scale = max(1, len(own) + len(enemy))
        score = _logit(self.base_rate)
        score += sum(self.hero_effects.get(hero, 0.0) for hero in own) / scale
        score -= sum(self.hero_effects.get(hero, 0.0) for hero in enemy) / scale
        return _sigmoid(score)

    def to_dict(self) -> dict[str, object]:
        return {
            "heroes": self.heroes,
            "base_rate": self.base_rate,
            "own_rates": self.own_rates,
            "enemy_rates": self.enemy_rates,
            "hero_effects": self.hero_effects,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ValueModel":
        model = cls(heroes=data["heroes"])
        model.base_rate = float(data.get("base_rate", 0.5))
        model.own_rates = {hero: float(value) for hero, value in data.get("own_rates", {}).items()}
        model.enemy_rates = {
            hero: float(value) for hero, value in data.get("enemy_rates", {}).items()
        }
        model.hero_effects = {
            hero: float(value) for hero, value in data.get("hero_effects", {}).items()
        }
        return model


def value_accuracy(model: ValueModel, samples: Iterable[LineupSample]) -> dict[str, float]:
    total = 0
    correct = 0
    brier = 0.0
    for sample in samples:
        total += 1
        probability = model.predict(sample.own_picks, sample.enemy_picks)
        correct += int((probability >= 0.5) == bool(sample.win))
        brier += (probability - sample.win) ** 2
    if total == 0:
        return {"total": 0, "accuracy": 0.0, "brier": 0.0}
    return {"total": total, "accuracy": correct / total, "brier": brier / total}
