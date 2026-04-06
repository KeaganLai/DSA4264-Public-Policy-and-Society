from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from .schemas import ResolvedLocation


ONEMAP_SEARCH_URL = "https://www.onemap.gov.sg/api/common/elastic/search"


def _normalize_text(value: str) -> str:
    value = value.upper().strip()
    value = re.sub(r"\s+", " ", value)
    return value


@dataclass
class _GeocodeResult:
    method: str
    search_query: str
    matched_address: str | None
    latitude: float
    longitude: float


class OneMapGeocoder:
    def __init__(self, reference_df: pd.DataFrame) -> None:
        self.df = reference_df.copy()
        self.df["town_norm"] = self.df["town"].astype(str).map(_normalize_text)
        self.df["street_norm"] = self.df["street_name"].astype(str).map(_normalize_text)
        self.df["block_norm"] = self.df["block"].astype(str).map(_normalize_text)
        self.df["full_address_norm"] = self.df["full_address"].astype(str).map(_normalize_text)

    def resolve(self, town: str, street_name: str, block: str) -> ResolvedLocation:
        town_norm = _normalize_text(town)
        street_norm = _normalize_text(street_name)
        block_norm = _normalize_text(block)

        street_candidates = self._expand_street_candidates(town_norm, street_norm)

        # 1) Preferred path: exact local match using the same keys as your policy users provide.
        # This avoids ambiguous OneMap hits when block+street tokens exist in multiple towns.
        for street_candidate in street_candidates:
            local_exact = self._local_strict_match(town_norm, street_candidate, block_norm)
            if local_exact is not None:
                return ResolvedLocation(
                    method="local_fallback",
                    normalized_town=town_norm,
                    normalized_street_name=str(local_exact["street_norm"]),
                    normalized_block=block_norm,
                    search_query=f"{block_norm} {street_candidate}, {town_norm}",
                    matched_address=str(local_exact["full_address"]),
                    latitude=float(local_exact["lat"]),
                    longitude=float(local_exact["lon"]),
                )

        # 1b) Strong local fallback on (town + block) when street fragment is inconsistent.
        # Example: Town=ANG MO KIO, Block=406, Street=AVE 8 should stay in ANG MO KIO instead
        # of drifting to another town that has a similarly numbered address.
        local_block = self._local_block_match(town_norm, block_norm)
        if local_block is not None:
            return ResolvedLocation(
                method="local_fallback",
                normalized_town=town_norm,
                normalized_street_name=str(local_block["street_norm"]),
                normalized_block=block_norm,
                search_query=f"{block_norm} {street_candidates[0]}, {town_norm}",
                matched_address=str(local_block["full_address"]),
                latitude=float(local_block["lat"]),
                longitude=float(local_block["lon"]),
            )

        # 2) OneMap lookup as secondary path.
        for street_candidate in street_candidates:
            search_candidates = self._build_queries(block_norm, street_candidate, town_norm)
            for query in search_candidates:
                result = self._query_onemap(query, block_norm, street_candidate, town_norm)
                if result is not None:
                    return ResolvedLocation(
                        method="onemap",
                        normalized_town=town_norm,
                        normalized_street_name=street_candidate,
                        normalized_block=block_norm,
                        search_query=result.search_query,
                        matched_address=result.matched_address,
                        latitude=result.latitude,
                        longitude=result.longitude,
                    )

        # 3) Relaxed local fallback for partial or noisy street input.
        for street_candidate in street_candidates:
            local = self._local_fallback(town_norm, street_candidate, block_norm)
            if local is not None:
                return ResolvedLocation(
                    method="local_fallback",
                    normalized_town=town_norm,
                    normalized_street_name=str(local["street_norm"]),
                    normalized_block=block_norm,
                    search_query=f"{block_norm} {street_candidate}, {town_norm}",
                    matched_address=str(local["full_address"]),
                    latitude=float(local["lat"]),
                    longitude=float(local["lon"]),
                )

        town_row = self._town_fallback(town_norm)
        return ResolvedLocation(
            method="town_fallback",
            normalized_town=town_norm,
            normalized_street_name=street_candidates[0],
            normalized_block=block_norm,
            search_query=f"{block_norm} {street_candidates[0]}, {town_norm}",
            matched_address=str(town_row["full_address"]),
            latitude=float(town_row["lat"]),
            longitude=float(town_row["lon"]),
        )

    def _expand_street_candidates(self, town_norm: str, street_norm: str) -> list[str]:
        """Allow shorthand street input (e.g. 'AVE 8') by matching town-specific full names."""
        candidates: list[str] = []

        def add(val: str) -> None:
            val = _normalize_text(val)
            if val and val not in candidates:
                candidates.append(val)

        add(street_norm)
        if town_norm and town_norm not in street_norm:
            add(f"{town_norm} {street_norm}")

        town_streets = (
            self.df.loc[self.df["town_norm"] == town_norm, "street_norm"].dropna().astype(str).unique().tolist()
        )
        suffix = f" {street_norm}"
        for s in town_streets:
            if s.endswith(suffix):
                add(s)
        for s in town_streets:
            if street_norm in s:
                add(s)

        return candidates[:8]

    @staticmethod
    def _build_queries(block: str, street: str, town: str) -> list[str]:
        return [
            f"{block} {street} {town}",
            f"{block} {street}",
            f"{block} {street} SINGAPORE",
            f"{street} {town}",
        ]

    def _query_onemap(
        self, search_value: str, block_norm: str, street_norm: str, town_norm: str
    ) -> _GeocodeResult | None:
        params = (
            f"searchVal={quote(search_value)}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
        )
        url = f"{ONEMAP_SEARCH_URL}?{params}"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

        try:
            with urlopen(request, timeout=4) as resp:  # nosec B310
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

        raw_results = payload.get("results", [])
        if not raw_results:
            return None

        best = self._choose_best_result(
            raw_results, block_norm=block_norm, street_norm=street_norm, town_norm=town_norm
        )
        if best is None:
            return None

        lat = float(best.get("LATITUDE"))
        lon = float(best.get("LONGITUDE"))
        matched = best.get("ADDRESS", "") or best.get("SEARCHVAL", "")

        return _GeocodeResult(
            method="onemap",
            search_query=search_value,
            matched_address=str(matched),
            latitude=lat,
            longitude=lon,
        )

    def _choose_best_result(
        self, results: list[dict[str, Any]], block_norm: str, street_norm: str, town_norm: str
    ) -> dict[str, Any] | None:
        scored: list[tuple[int, dict[str, Any]]] = []
        for r in results:
            address = _normalize_text(str(r.get("ADDRESS", "")))
            block = _normalize_text(str(r.get("BLK_NO", "")))
            road = _normalize_text(str(r.get("ROAD_NAME", "")))
            town_match = self._address_matches_town(address, town_norm)
            block_match = bool(block and block == block_norm)
            road_match = bool(street_norm and road == street_norm)
            street_in_address = bool(street_norm and street_norm in address)

            score = 0
            if town_match:
                score += 4
            if block_match:
                score += 3
            if road_match:
                score += 3
            if street_in_address:
                score += 1
            if block_norm and block_norm in address:
                score += 1

            # Reject weak hits that only match block number (common across different towns).
            # Require one of:
            # - town + block
            # - town + road
            # - block + road
            if not ((town_match and block_match) or (town_match and road_match) or (block_match and road_match)):
                continue

            scored.append((score, r))

        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_row = scored[0]
        # Require at least a minimally coherent hit to avoid cross-town false positives.
        if best_score < 4:
            return None
        return best_row

    @staticmethod
    def _address_matches_town(address: str, town_norm: str) -> bool:
        if not town_norm:
            return False
        if town_norm in address:
            return True

        tokens = [t for t in re.split(r"[^A-Z0-9]+", town_norm) if t]
        long_tokens = [t for t in tokens if len(t) >= 4]
        if long_tokens and any(t in address for t in long_tokens):
            return True

        short_tokens = [t for t in tokens if len(t) >= 3]
        short_hits = sum(1 for t in short_tokens if t in address)
        return short_hits >= 2

    def _local_strict_match(self, town_norm: str, street_norm: str, block_norm: str) -> pd.Series | None:
        subset = self.df.loc[
            (self.df["town_norm"] == town_norm)
            & (self.df["street_norm"] == street_norm)
            & (self.df["block_norm"] == block_norm)
        ]
        if subset.empty:
            return None
        return subset.sort_values("month").iloc[-1]

    def _local_block_match(self, town_norm: str, block_norm: str) -> pd.Series | None:
        subset = self.df.loc[
            (self.df["town_norm"] == town_norm)
            & (self.df["block_norm"] == block_norm)
        ]
        if subset.empty:
            return None
        return subset.sort_values("month").iloc[-1]

    def _local_fallback(self, town_norm: str, street_norm: str, block_norm: str) -> pd.Series | None:
        subset = self.df.loc[
            (self.df["town_norm"] == town_norm)
            & (self.df["street_norm"] == street_norm)
            & (self.df["block_norm"] == block_norm)
        ]
        if subset.empty:
            subset = self.df.loc[
                (self.df["town_norm"] == town_norm) & (self.df["street_norm"] == street_norm)
            ]
        if subset.empty:
            subset = self.df.loc[(self.df["town_norm"] == town_norm) & (self.df["block_norm"] == block_norm)]
        if subset.empty:
            return None
        # Use latest available row when there are duplicates over time.
        return subset.sort_values("month").iloc[-1]

    def _town_fallback(self, town_norm: str) -> pd.Series:
        subset = self.df.loc[self.df["town_norm"] == town_norm]
        if subset.empty:
            subset = self.df
        lat_mid = float(subset["lat"].median())
        lon_mid = float(subset["lon"].median())
        d = (subset["lat"] - lat_mid) ** 2 + (subset["lon"] - lon_mid) ** 2
        return subset.loc[d.idxmin()]
