"""
explainer.py
------------
Interpretability layer for the Syntact TalentEngine.

Generates human-readable, data-driven rationale strings explaining each
candidate's prioritised rank position.  All logic is programmatic and
template-driven — zero LLM calls, zero network latency, deterministic output.

Design principles
-----------------
* Score Band Classification  : thresholds map numeric signals to qualitative
  bands (ELITE / HIGH / MEDIUM / LOW) to drive template selection.
* Template Pools             : each signal dimension has multiple phrasing
  variants.  A stable hash of ``candidate_id`` deterministically picks one
  variant, ensuring reproducibility without randomness.
* Signal Decomposition       : rationale explicitly attributes rank to the
  proportional mix of semantic (capability) and behavioural (intent) signals.
* Enriched Fields (Optional) : when career trajectory features are present
  in the candidate row (from ``data_pipeline.engineer_features``), they
  contribute additional trajectory and velocity commentary.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Score band thresholds
# ---------------------------------------------------------------------------

# Cosine similarity bands  (semantic capability alignment)
_SIM_ELITE:  float = 0.75
_SIM_HIGH:   float = 0.50
_SIM_MEDIUM: float = 0.25
# < 0.25 -> LOW

# Intent score bands  (normalised behavioural activation, [0, 1])
_INT_ELITE:  float = 0.75
_INT_HIGH:   float = 0.50
_INT_MEDIUM: float = 0.30
# < 0.30 -> LOW

# Alignment score bands  (LGBMRanker output, unbounded)
_ALN_ELITE:  float = 2.5
_ALN_HIGH:   float = 0.5
_ALN_MEDIUM: float = -0.5
# < -0.5 -> LOW

# Career trajectory bands
_VEL_HIGH:   float = 0.40   # promotions per year
_TEN_LOW:    float = 1.5    # avg tenure < this -> fast mover
_DUR_LONG:   float = 8.0    # total career years

# ---------------------------------------------------------------------------
# Template pools  (indexed deterministically by hash)
# ---------------------------------------------------------------------------

_SIM_ELITE_POOL: List[str] = [
    "Demonstrates elite semantic alignment with the job criteria — deep technical vocabulary overlap confirmed.",
    "Capability vector registers exceptional resonance with the target role specification.",
    "Semantic profile matches the job description at an elite confidence level across core technical domains.",
]
_SIM_HIGH_POOL: List[str] = [
    "Strong semantic alignment detected with the primary technical requirements of the role.",
    "Capability embedding shows high-fidelity correspondence with the job's skill criteria.",
    "Technical experience text correlates strongly with the target job description.",
]
_SIM_MEDIUM_POOL: List[str] = [
    "Moderate semantic alignment with the role — candidate covers a meaningful subset of the required skill space.",
    "Partial capability match identified; candidate demonstrates relevant but not exhaustive technical coverage.",
    "Experience narrative overlaps with core elements of the job description at a moderate confidence level.",
]
_SIM_LOW_POOL: List[str] = [
    "Semantic alignment with the role specification is limited — candidate's technical profile diverges from primary criteria.",
    "Capability vector shows low correspondence with the job description; elevated by strong behavioural signals.",
    "Weak direct skill match detected; ranking is driven primarily by behavioural intent and model re-ranking.",
]

_INT_ELITE_POOL: List[str] = [
    "Intent signal is in the elite tier — confirms sustained, high-frequency platform engagement.",
    "Behavioural activations are maximally elevated, indicating proactive and consistent platform presence.",
    "Platform activity metrics register at elite levels, signalling exceptional engagement intensity.",
]
_INT_HIGH_POOL: List[str] = [
    "Intent score is high — platform behaviour indicates consistent, recent engagement.",
    "Behavioural metrics reflect strong active presence on the platform.",
    "High intent activation confirms regular submission activity and profile investment.",
]
_INT_MEDIUM_POOL: List[str] = [
    "Behavioural engagement is at a moderate level — candidate demonstrates intermittent platform activity.",
    "Intent metrics indicate periodic engagement; activity cadence is consistent but not intensive.",
    "Platform signals suggest a moderate engagement pattern across interaction and submission dimensions.",
]
_INT_LOW_POOL: List[str] = [
    "Intent signal is below threshold — limited recent platform activity detected.",
    "Behavioural activation is low; candidate may be passive on the platform relative to peers.",
    "Platform engagement metrics are minimal; ranking relies more heavily on the semantic capability signal.",
]

_VEL_HIGH_POOL: List[str] = [
    "Career velocity is elevated — role transitions indicate rapid upward progression.",
    "Promotion cadence exceeds the cohort average, signalling accelerated career momentum.",
    "High velocity metric confirms rapid role advancement despite a compact career timeline.",
]
_VEL_LOW_POOL: List[str] = [
    "Career trajectory shows deliberate, steady tenure — indicative of deep domain expertise accumulation.",
    "Extended role tenures signal depth over breadth — candidate likely possesses specialised institutional knowledge.",
    "Low velocity reflects stability-focused career development rather than frequent role-hopping.",
]
_TEN_SHORT_POOL: List[str] = [
    "Average role tenure is compact — candidate moves fast and adapts quickly across contexts.",
    "Short mean tenure indicates high adaptability and willingness to take on new challenges rapidly.",
]
_DUR_LONG_POOL: List[str] = [
    "Total career duration is substantial — candidate brings a significant depth of accumulated experience.",
    "Extended career timeline confirms multi-year professional exposure across the technical domain.",
]

_CONCLUSION_ELITE_POOL: List[str] = [
    "Overall composite score places this candidate in the elite tier — recommended for immediate shortlisting.",
    "Combined signal profile is exceptional — candidate is a high-confidence match for priority consideration.",
    "Unified ranking score is in the top percentile — strongly recommended for first-round evaluation.",
]
_CONCLUSION_HIGH_POOL: List[str] = [
    "Strong unified score indicates a high-confidence fit — candidate is well-positioned for evaluation.",
    "Composite alignment score is high — candidate merits prioritised review.",
    "Aggregate signal profile supports a strong candidacy recommendation.",
]
_CONCLUSION_MEDIUM_POOL: List[str] = [
    "Moderate composite score — candidate is a viable option; manual review of supporting profile is advised.",
    "Aggregate alignment score is mid-tier; candidate may be suitable with additional context review.",
    "Balanced but moderate overall signal — recommended for secondary evaluation.",
]
_CONCLUSION_LOW_POOL: List[str] = [
    "Composite score is below the primary threshold — included on the shortlist as a pool-floor reference candidate.",
    "Low overall alignment; candidate is ranked as a contingency option pending top-tier availability.",
    "Aggregate signal profile is below the high-confidence band — proceed with contextual caution.",
]

# Proportion commentary templates
_PROP_SIM_DOMINANT: str = (
    "Signal decomposition: semantic capability contributes {sim_pct:.0f}% of the attributable score weight, "
    "with behavioural intent accounting for the remaining {int_pct:.0f}%."
)
_PROP_INT_DOMINANT: str = (
    "Signal decomposition: platform behavioural intent drives {int_pct:.0f}% of the attributable score weight, "
    "with semantic capability contributing {sim_pct:.0f}%."
)
_PROP_BALANCED: str = (
    "Signal decomposition: semantic capability and behavioural intent contribute "
    "approximately equally ({sim_pct:.0f}% / {int_pct:.0f}%) to the attributable score weight."
)


class CandidateExplainer:
    """
    Generates structured, human-readable rationale strings for ranked candidates.

    Parameters
    ----------
    sim_weight : float
        Weight assigned to the cosine similarity dimension when computing
        the proportional signal attribution display.  Defaults to 0.6.
    int_weight : float
        Weight assigned to the intent dimension.  Defaults to 0.4.
        Note: these weights only govern the displayed proportion text;
        actual re-ranking is performed by the LGBMRanker.
    """

    def __init__(self, sim_weight: float = 0.6, int_weight: float = 0.4) -> None:
        if abs((sim_weight + int_weight) - 1.0) > 1e-6:
            raise ValueError("sim_weight and int_weight must sum to 1.0.")
        self.sim_weight = sim_weight
        self.int_weight = int_weight

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stable_pick(pool: List[str], candidate_id: Any) -> str:
        """
        Deterministically selects one template from ``pool`` using a stable
        hash of ``candidate_id``, ensuring reproducible output across runs.
        """
        h = int(hashlib.md5(str(candidate_id).encode()).hexdigest(), 16)
        return pool[h % len(pool)]

    @staticmethod
    def _classify_sim(score: float) -> str:
        if score >= _SIM_ELITE:  return "ELITE"
        if score >= _SIM_HIGH:   return "HIGH"
        if score >= _SIM_MEDIUM: return "MEDIUM"
        return "LOW"

    @staticmethod
    def _classify_int(score: float) -> str:
        if score >= _INT_ELITE:  return "ELITE"
        if score >= _INT_HIGH:   return "HIGH"
        if score >= _INT_MEDIUM: return "MEDIUM"
        return "LOW"

    @staticmethod
    def _classify_aln(score: float) -> str:
        if score >= _ALN_ELITE:  return "ELITE"
        if score >= _ALN_HIGH:   return "HIGH"
        if score >= _ALN_MEDIUM: return "MEDIUM"
        return "LOW"

    def _sim_phrase(self, band: str, cid: Any) -> str:
        pools = {
            "ELITE": _SIM_ELITE_POOL, "HIGH": _SIM_HIGH_POOL,
            "MEDIUM": _SIM_MEDIUM_POOL, "LOW": _SIM_LOW_POOL,
        }
        return self._stable_pick(pools[band], str(cid) + "sim")

    def _int_phrase(self, band: str, cid: Any) -> str:
        pools = {
            "ELITE": _INT_ELITE_POOL, "HIGH": _INT_HIGH_POOL,
            "MEDIUM": _INT_MEDIUM_POOL, "LOW": _INT_LOW_POOL,
        }
        return self._stable_pick(pools[band], str(cid) + "int")

    def _conclusion_phrase(self, band: str, cid: Any) -> str:
        pools = {
            "ELITE": _CONCLUSION_ELITE_POOL, "HIGH": _CONCLUSION_HIGH_POOL,
            "MEDIUM": _CONCLUSION_MEDIUM_POOL, "LOW": _CONCLUSION_LOW_POOL,
        }
        return self._stable_pick(pools[band], str(cid) + "aln")

    def _proportion_phrase(self, sim_score: float, int_score: float) -> str:
        """Generates the proportional signal attribution commentary."""
        sim_abs = sim_score * self.sim_weight
        int_abs = int_score * self.int_weight
        total = sim_abs + int_abs

        if total < 1e-9:
            sim_pct, int_pct = 50.0, 50.0
        else:
            sim_pct = (sim_abs / total) * 100.0
            int_pct = 100.0 - sim_pct

        balance_delta = abs(sim_pct - int_pct)
        if balance_delta <= 15.0:
            template = _PROP_BALANCED
        elif sim_pct > int_pct:
            template = _PROP_SIM_DOMINANT
        else:
            template = _PROP_INT_DOMINANT

        return template.format(sim_pct=sim_pct, int_pct=int_pct)

    def _trajectory_phrase(self, row: pd.Series) -> Optional[str]:
        """
        Generates optional career trajectory commentary when engineer_features
        output columns are present in the candidate row.
        """
        parts: List[str] = []

        if "promotion_velocity" in row.index and pd.notna(row["promotion_velocity"]):
            vel = float(row["promotion_velocity"])
            cid = row.get("candidate_id", "x")
            if vel >= _VEL_HIGH:
                parts.append(self._stable_pick(_VEL_HIGH_POOL, str(cid) + "vel"))
            elif vel > 0:
                parts.append(self._stable_pick(_VEL_LOW_POOL, str(cid) + "vel"))

        if "avg_tenure_per_role" in row.index and pd.notna(row["avg_tenure_per_role"]):
            avg_ten = float(row["avg_tenure_per_role"])
            cid = row.get("candidate_id", "x")
            if 0 < avg_ten < _TEN_LOW:
                parts.append(self._stable_pick(_TEN_SHORT_POOL, str(cid) + "ten"))

        if "total_career_duration" in row.index and pd.notna(row["total_career_duration"]):
            dur = float(row["total_career_duration"])
            cid = row.get("candidate_id", "x")
            if dur >= _DUR_LONG:
                parts.append(self._stable_pick(_DUR_LONG_POOL, str(cid) + "dur"))

        return " ".join(parts) if parts else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_rationale(
        self,
        candidate_row: pd.Series,
        similarity_score: float,
        intent_score: float,
        alignment_score: float = 0.0,
    ) -> str:
        """
        Generates a structured, human-readable rationale string for one candidate.

        The rationale is composed of up to four clauses:
          1. Semantic capability assessment (cosine similarity band)
          2. Proportional signal attribution (sim% vs intent%)
          3. Behavioural intent assessment (intent score band)
          4. Career trajectory commentary (if enriched columns present)
          5. Overall conclusion (alignment score band)

        Parameters
        ----------
        candidate_row : pd.Series
            One row from the ranked shortlist or enriched profiles DataFrame.
            Must contain ``candidate_id``; may optionally contain
            ``promotion_velocity``, ``avg_tenure_per_role``,
            ``total_career_duration`` from ``engineer_features``.
        similarity_score : float
            Cosine similarity between job embedding and candidate capability
            vector (from ``PredictiveRankerEngine``).
        intent_score : float
            Aggregated, normalised behavioural intent score (from ranker output).
        alignment_score : float
            Final LGBMRanker alignment score (used for overall conclusion tier).

        Returns
        -------
        str
            Multi-clause rationale string.
        """
        cid = candidate_row.get("candidate_id", "unknown")

        sim_band = self._classify_sim(float(similarity_score))
        int_band = self._classify_int(float(intent_score))
        aln_band = self._classify_aln(float(alignment_score))

        # Clause 1 — Semantic capability
        clause_sim = self._sim_phrase(sim_band, cid)

        # Clause 2 — Proportional attribution
        clause_prop = self._proportion_phrase(float(similarity_score), float(intent_score))

        # Clause 3 — Behavioural intent
        clause_int = self._int_phrase(int_band, cid)

        # Clause 4 — Career trajectory (optional, only if enriched fields present)
        clause_traj = self._trajectory_phrase(candidate_row)

        # Clause 5 — Conclusion
        clause_conclusion = self._conclusion_phrase(aln_band, cid)

        # Assemble
        clauses = [clause_sim, clause_prop, clause_int]
        if clause_traj:
            clauses.append(clause_traj)
        clauses.append(clause_conclusion)

        rationale = " ".join(clauses)
        return rationale

    def explain_shortlist(
        self,
        shortlist_df: pd.DataFrame,
        enriched_profiles: Optional[pd.DataFrame] = None,
        rationale_col: str = "rationale",
    ) -> pd.DataFrame:
        """
        Appends a ``rationale`` column to the full ranked shortlist DataFrame.

        When ``enriched_profiles`` is supplied (output of
        ``TalentDataPipeline.engineer_features``), career trajectory fields are
        joined by ``candidate_id`` before rationale generation, enabling
        trajectory commentary in the output strings.

        Parameters
        ----------
        shortlist_df : pd.DataFrame
            Ranked shortlist from ``PredictiveRankerEngine.generate_shortlist``.
            Must contain: ``candidate_id``, ``cosine_similarity``,
            ``intent_score``, ``alignment_score``.
        enriched_profiles : pd.DataFrame, optional
            Feature-engineered candidate profiles.  If provided, joined to
            shortlist on ``candidate_id`` to augment trajectory fields.
        rationale_col : str
            Name of the column to add.  Defaults to ``"rationale"``.

        Returns
        -------
        pd.DataFrame
            Copy of ``shortlist_df`` with the ``rationale_col`` appended.
        """
        # Optional join with enriched profiles for trajectory features
        if enriched_profiles is not None and "candidate_id" in enriched_profiles.columns:
            traj_cols = [
                c for c in ["candidate_id", "promotion_velocity",
                             "avg_tenure_per_role", "total_career_duration",
                             "years_of_experience"]
                if c in enriched_profiles.columns
            ]
            shortlist_df = shortlist_df.merge(
                enriched_profiles[traj_cols],
                on="candidate_id",
                how="left",
            )

        def _row_rationale(row: pd.Series) -> str:
            # Support both old 'alignment_score' and new competition 'score' column name
            alignment_score = row.get("alignment_score", row.get("score", 0.0))
            return self.generate_rationale(
                candidate_row=row,
                similarity_score=row.get("cosine_similarity", 0.0),
                intent_score=row.get("intent_score", 0.0),
                alignment_score=float(alignment_score),
            )

        explained = shortlist_df.copy()
        explained[rationale_col] = explained.apply(_row_rationale, axis=1)

        logger.info(
            "Rationale generated for %d candidates — column '%s' appended.",
            len(explained),
            rationale_col,
        )
        return explained

    def save_explained_shortlist(
        self,
        explained_df: pd.DataFrame,
        output_path: str = "outputs/ranked_shortlist_explained.csv",
    ) -> str:
        """
        Persists the explained shortlist (with rationale column) to CSV.

        Parameters
        ----------
        explained_df : pd.DataFrame
            Output of ``explain_shortlist``.
        output_path : str
            Destination file path.

        Returns
        -------
        str
            Absolute path of the saved CSV.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        explained_df.to_csv(output_path, index=False)
        abs_path = os.path.abspath(output_path)
        logger.info("Explained shortlist saved to: %s  (%d rows)", abs_path, len(explained_df))
        return abs_path


# ---------------------------------------------------------------------------
# Execution block — verification against the live ranker output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    SHORTLIST_PATH = "outputs/ranked_shortlist.csv"
    OUTPUT_PATH    = "outputs/ranked_shortlist_explained.csv"

    print("=" * 60)
    print("   Candidate Explainer -- End-to-End Verification")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load ranked shortlist produced by ranker.py
    # ------------------------------------------------------------------
    if not os.path.exists(SHORTLIST_PATH):
        print(f"ERROR: Shortlist not found at '{SHORTLIST_PATH}'. Run ranker.py first.")
        sys.exit(1)

    shortlist = pd.read_csv(SHORTLIST_PATH)
    print(f"\nLoaded shortlist: {shortlist.shape[0]} candidates from '{SHORTLIST_PATH}'")
    print(f"Columns: {list(shortlist.columns)}\n")

    # ------------------------------------------------------------------
    # Instantiate explainer and generate rationale for each candidate
    # ------------------------------------------------------------------
    explainer = CandidateExplainer(sim_weight=0.6, int_weight=0.4)
    explained = explainer.explain_shortlist(shortlist)

    # ------------------------------------------------------------------
    # Print individual rationales
    # ------------------------------------------------------------------
    print("-" * 60)
    for _, row in explained.iterrows():
        print(f"\nRank #{int(row['rank'])}  |  Candidate {row['candidate_id']}")
        print(f"  Cosine Sim   : {row['cosine_similarity']:.4f}")
        print(f"  Intent Score : {row['intent_score']:.4f}")
        print(f"  Alignment    : {row['alignment_score']:.4f}")
        print(f"  Rationale    : {row['rationale']}")
    print("-" * 60)

    # ------------------------------------------------------------------
    # Persist the explained shortlist
    # ------------------------------------------------------------------
    saved_path = explainer.save_explained_shortlist(explained, OUTPUT_PATH)

    # ------------------------------------------------------------------
    # Verify saved CSV schema
    # ------------------------------------------------------------------
    verified = pd.read_csv(saved_path)
    print(f"\nExplained CSV saved : {saved_path}")
    print(f"  Rows    : {len(verified)}")
    print(f"  Columns : {list(verified.columns)}")
    assert "rationale" in verified.columns, "rationale column missing from output!"
    assert len(verified) == len(shortlist),  "Row count mismatch!"

    print("\nExplainer verification completed successfully.")
    sys.exit(0)
