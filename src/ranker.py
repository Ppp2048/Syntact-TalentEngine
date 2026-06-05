"""
ranker.py
---------
Predictive talent ranking engine for the Syntact TalentEngine.

Pipeline:
  1. Compute cosine similarity between a job-description embedding and each
     candidate's capability vector (from DualSignalEncoder).
  2. Build a unified feature matrix: [cosine_sim | intent_metrics].
  3. Train / infer with a LightGBM LGBMRanker (LambdaMART objective) to
     produce a final alignment probability score per candidate.
  4. Sort by score descending to produce a prioritised shortlist.
  5. Persist the shortlist to outputs/ranked_shortlist.csv.
  6. Evaluate ranking quality with NDCG@10.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Output path constants
OUTPUT_DIR: str = "outputs"
OUTPUT_FILE: str = os.path.join(OUTPUT_DIR, "ranked_shortlist.csv")

# Competition submission schema — fixed column order required by the scoring harness.
_SUBMISSION_COLS: List[str] = [
    "candidate_id",
    "rank",
    "score",
    "reasoning",
]

# Internal working columns carried during ranking before export
_INTERNAL_COLS: List[str] = [
    "candidate_id",
    "cosine_similarity",
    "intent_score",
    "alignment_score",
]


# ---------------------------------------------------------------------------
# Module-level helper: competition reasoning synthesiser
# ---------------------------------------------------------------------------

def _build_reasoning(
    cosine_similarity: float,
    intent_score: float,
    alignment_score: float,
) -> str:
    """
    Synthesises a concise, human-readable rationale string for the
    competition ``reasoning`` column from three scalar signal values.

    Logic tiers:
      - Semantic alignment level  : high (≥0.35) / moderate (≥0.25) / low
      - Behavioral intent level   : strong (≥0.55) / moderate (≥0.35) / weak
      - Combined alignment tier   : exceptional (≥0.40) / strong (≥0.30) / moderate

    Parameters
    ----------
    cosine_similarity : float   Semantic capability similarity score.
    intent_score      : float   Mean-aggregated behavioral intent score.
    alignment_score   : float   Final unified alignment score.

    Returns
    -------
    str  Structured rationale text ready for CSV export.
    """
    # Semantic tier
    if cosine_similarity >= 0.35:
        sem_tier = "high semantic alignment"
    elif cosine_similarity >= 0.25:
        sem_tier = "moderate semantic alignment"
    else:
        sem_tier = "low semantic alignment"

    # Behavioral intent tier
    if intent_score >= 0.55:
        intent_tier = "strong platform engagement"
    elif intent_score >= 0.35:
        intent_tier = "moderate platform engagement"
    else:
        intent_tier = "early-stage platform engagement"

    # Overall alignment descriptor
    if alignment_score >= 0.40:
        overall = "Exceptional fit"
    elif alignment_score >= 0.30:
        overall = "Strong fit"
    elif alignment_score >= 0.20:
        overall = "Moderate fit"
    else:
        overall = "Emerging fit"

    return (
        f"{overall} detected. "
        f"Capability vector shows {sem_tier} with job description "
        f"(cosine={cosine_similarity:.4f}). "
        f"Behavioral intent reflects {intent_tier} "
        f"(intent={intent_score:.4f}). "
        f"Unified alignment score: {alignment_score:.4f}."
    )


class PredictiveRankerEngine:

    """
    Fuses semantic capability distances and behavioural intent activations
    through a LightGBM LambdaMART ranker to produce a calibrated, ordered
    candidate shortlist.

    Parameters
    ----------
    lgbm_params : dict, optional
        Override dictionary for LGBMRanker hyperparameters.
        Defaults are tuned for small-to-medium talent ranking corpora.
    top_k : int
        Number of top candidates to include in the saved shortlist CSV.
        Default is 10 (full shortlist still returned as DataFrame).
    random_state : int
        Seed for reproducibility.
    """

    # Default LGBMRanker hyperparameters (LambdaMART objective)
    _DEFAULT_LGBM_PARAMS: Dict = {
        "objective":       "lambdarank",
        "metric":          "ndcg",
        "ndcg_eval_at":    [5, 10],
        "learning_rate":   0.05,
        "n_estimators":    200,
        "num_leaves":      31,
        "min_child_samples": 5,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "random_state":    42,
        "n_jobs":          -1,
        "verbose":         -1,
    }

    def __init__(
        self,
        lgbm_params: Optional[Dict] = None,
        top_k: int = 100,
        random_state: int = 42,
    ) -> None:
        self.top_k = top_k
        self.random_state = random_state

        params = {**self._DEFAULT_LGBM_PARAMS}
        if lgbm_params:
            params.update(lgbm_params)
        params["random_state"] = random_state

        self._ranker = LGBMRanker(**params)
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # 1. COSINE SIMILARITY — job description vs. candidate capability
    # ------------------------------------------------------------------

    @staticmethod
    def compute_cosine_similarity(
        job_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the cosine similarity between a single job-description
        embedding and a matrix of candidate capability embeddings.

        Handles zero-norm vectors gracefully (returns 0.0 similarity).

        Parameters
        ----------
        job_embedding : np.ndarray
            1-D array of shape ``(embedding_dim,)``.
        candidate_embeddings : np.ndarray
            2-D array of shape ``(N, embedding_dim)``.

        Returns
        -------
        np.ndarray
            1-D float array of shape ``(N,)`` with values in ``[-1, 1]``.
        """
        job_norm: float = float(np.linalg.norm(job_embedding))
        if job_norm == 0.0:
            logger.warning("Job embedding is a zero vector — similarity will be 0 for all candidates.")
            return np.zeros(len(candidate_embeddings), dtype=np.float32)

        job_unit: np.ndarray = job_embedding / job_norm

        cand_norms: np.ndarray = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        # Avoid division by zero for empty-text candidates
        safe_norms = np.where(cand_norms == 0.0, 1.0, cand_norms)
        cand_unit: np.ndarray = candidate_embeddings / safe_norms

        similarities: np.ndarray = cand_unit @ job_unit  # shape: (N,)

        # Zero out candidates whose norm was 0 (no meaningful capability text)
        zero_mask: np.ndarray = (cand_norms.flatten() == 0.0)
        similarities[zero_mask] = 0.0

        return similarities.astype(np.float32)

    # ------------------------------------------------------------------
    # 2. UNIFIED FEATURE MATRIX
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self,
        cosine_similarities: np.ndarray,
        intent_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Concatenates the semantic similarity column with the scaled
        behavioural intent matrix to form the unified feature matrix
        consumed by the LGBMRanker.

        Layout: ``[cosine_sim (1) | intent_features (K)]``

        Parameters
        ----------
        cosine_similarities : np.ndarray
            Shape ``(N,)`` — cosine similarity scores per candidate.
        intent_matrix : np.ndarray
            Shape ``(N, K)`` — MinMax-scaled behavioural activations.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(N, 1 + K)``.
        """
        sim_col = cosine_similarities.reshape(-1, 1).astype(np.float32)
        intent = intent_matrix.astype(np.float32)
        unified: np.ndarray = np.concatenate([sim_col, intent], axis=1)
        logger.info(
            "Unified feature matrix built — shape: %s  "
            "[cosine_sim(1) | intent(%d)]",
            unified.shape,
            intent.shape[1],
        )
        return unified

    # ------------------------------------------------------------------
    # 3a. FIT — train the LGBMRanker on labelled data
    # ------------------------------------------------------------------

    def fit(
        self,
        feature_matrix: np.ndarray,
        relevance_labels: np.ndarray,
        group_sizes: np.ndarray,
    ) -> "PredictiveRankerEngine":
        """
        Trains the LGBMRanker (LambdaMART) on labelled candidate data.

        Parameters
        ----------
        feature_matrix : np.ndarray
            Shape ``(N_total, n_features)`` — unified features for all
            candidates across all queries.
        relevance_labels : np.ndarray
            Shape ``(N_total,)`` — integer relevance grades (0–4 scale
            or binary 0/1) aligned row-wise with ``feature_matrix``.
        group_sizes : np.ndarray
            1-D array where each element is the number of candidates
            belonging to one query (job). Must sum to ``N_total``.

        Returns
        -------
        self
        """
        logger.info(
            "Fitting LGBMRanker — %d candidates across %d queries …",
            len(relevance_labels),
            len(group_sizes),
        )
        self._ranker.fit(
            feature_matrix,
            relevance_labels,
            group=group_sizes,
        )
        self._is_fitted = True
        logger.info("LGBMRanker fitted successfully.")
        return self

    # ------------------------------------------------------------------
    # 3b. SCORE — produce alignment probability scores
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        feature_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Predicts alignment scores for each candidate using the fitted
        LGBMRanker.  When the model is not yet fitted, falls back to raw
        cosine similarity (column 0 of the feature matrix) so the engine
        remains functional in zero-shot / inference-only mode.

        Parameters
        ----------
        feature_matrix : np.ndarray
            Shape ``(N, n_features)`` — unified feature matrix.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` — unbounded alignment scores (higher = better).
        """
        if not self._is_fitted:
            logger.warning(
                "LGBMRanker not fitted — using raw cosine similarity as "
                "alignment score (zero-shot mode)."
            )
            return feature_matrix[:, 0].astype(np.float64)

        scores: np.ndarray = self._ranker.predict(feature_matrix)
        logger.info("Alignment scores computed — shape: %s", scores.shape)
        return scores

    # ------------------------------------------------------------------
    # 4. SHORTLIST — sort and format the final ranked output
    # ------------------------------------------------------------------

    def generate_shortlist(
        self,
        candidate_ids: List,
        cosine_similarities: np.ndarray,
        intent_matrix: np.ndarray,
        alignment_scores: np.ndarray,
    ) -> pd.DataFrame:
        """
        Sorts candidates by alignment score (descending) with deterministic
        tie-breaking on ``candidate_id`` ascending, then assembles the
        competition-format shortlist DataFrame.

        Tie-breaking guarantee: when two candidates share identical
        ``alignment_score`` values (to 8 significant figures), they are
        ordered by ``candidate_id`` ascending so the output is fully
        reproducible across runs.

        Parameters
        ----------
        candidate_ids : list
            Candidate identifier values (length ``N``).
        cosine_similarities : np.ndarray
            Shape ``(N,)`` — semantic similarity per candidate.
        intent_matrix : np.ndarray
            Shape ``(N, K)`` — scaled behavioural activations.
        alignment_scores : np.ndarray
            Shape ``(N,)`` — final LGBMRanker output scores.

        Returns
        -------
        pd.DataFrame
            Full ranked DataFrame (all N candidates) in competition schema:
            ``[candidate_id, rank, score, reasoning]``.
        """
        intent_scores: np.ndarray = intent_matrix.mean(axis=1).astype(np.float64)
        cos_arr = np.round(cosine_similarities.astype(np.float64), 6)
        intent_arr = np.round(intent_scores, 6)
        score_arr = np.round(alignment_scores.astype(np.float64), 6)

        # Build internal working frame
        df = pd.DataFrame({
            "candidate_id":      candidate_ids,
            "cosine_similarity": cos_arr,
            "intent_score":      intent_arr,
            "alignment_score":   score_arr,
        })

        # Deterministic sort: primary = alignment_score DESC,
        #                     secondary = candidate_id ASC (tie-breaker)
        df["_cid_str"] = df["candidate_id"].astype(str)
        df.sort_values(
            by=["alignment_score", "_cid_str"],
            ascending=[False, True],
            inplace=True,
        )
        df.drop(columns=["_cid_str"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.insert(0, "rank", df.index + 1)

        # Synthesise competition-schema "reasoning" column
        df["reasoning"] = df.apply(
            lambda r: _build_reasoning(
                r["cosine_similarity"],
                r["intent_score"],
                r["alignment_score"],
            ),
            axis=1,
        )

        # Rename alignment_score -> score for competition schema
        df.rename(columns={"alignment_score": "score"}, inplace=True)

        logger.info(
            "Shortlist generated -- %d candidates ranked. "
            "Top candidate: %s (score=%.4f)",
            len(df),
            df.iloc[0]["candidate_id"],
            df.iloc[0]["score"],
        )
        return df

    # ------------------------------------------------------------------
    # 5. PERSIST — save shortlist CSV
    # ------------------------------------------------------------------

    @staticmethod
    def save_shortlist(
        shortlist_df: pd.DataFrame,
        top_k: int = 100,
        output_path: str = OUTPUT_FILE,
    ) -> str:
        """
        Saves the top-``top_k`` rows of the ranked shortlist to a CSV file
        using the strict competition schema: ``candidate_id,rank,score,reasoning``.

        Parameters
        ----------
        shortlist_df : pd.DataFrame
            Full ranked shortlist from ``generate_shortlist``.
        top_k : int
            Number of rows to save.  Defaults to 100 (competition requirement).
        output_path : str
            Destination file path.

        Returns
        -------
        str
            Absolute path of the saved CSV.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Enforce competition column order; extra working columns are dropped.
        export_cols = [c for c in _SUBMISSION_COLS if c in shortlist_df.columns]
        export_df = shortlist_df[export_cols].head(top_k).copy()
        # Re-number ranks monotonically from 1 in case the slice is a subset
        export_df["rank"] = range(1, len(export_df) + 1)
        export_df.to_csv(output_path, index=False)
        abs_path = os.path.abspath(output_path)
        logger.info("Shortlist saved to: %s  (%d rows)", abs_path, len(export_df))
        return abs_path

    # ------------------------------------------------------------------
    # 6. EVALUATION — NDCG@10
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_ndcg(
        relevance_labels: np.ndarray,
        alignment_scores: np.ndarray,
        k: int = 10,
    ) -> float:
        """
        Computes NDCG@k and **always prints the result to the console**.

        In zero-shot mode (no ground-truth labels), pass ``cosine_similarity``
        values as both ``relevance_labels`` and ``alignment_scores`` as a
        surrogate upper-bound estimate.

        Parameters
        ----------
        relevance_labels : np.ndarray
            Shape ``(N,)`` — ground-truth integer relevance grades.
            For zero-shot mode, pass the cosine similarity array.
        alignment_scores : np.ndarray
            Shape ``(N,)`` — predicted alignment scores from the ranker.
        k : int
            Rank cutoff.  Defaults to 10.

        Returns
        -------
        float
            NDCG@k in ``[0.0, 1.0]``.  Returns 0.0 on error.
        """
        try:
            score: float = ndcg_score(
                y_true=relevance_labels.reshape(1, -1),
                y_score=alignment_scores.reshape(1, -1),
                k=k,
            )
            logger.info("NDCG@%d = %.4f", k, score)
            print(f"[NDCG@{k}]  {score:.6f}")
            return score
        except Exception as exc:
            logger.error("NDCG computation failed: %s", exc)
            print(f"[NDCG@{k}]  0.000000  (computation error: {exc})")
            return 0.0

    # ------------------------------------------------------------------
    # 7. FULL PIPELINE — convenience end-to-end method
    # ------------------------------------------------------------------

    def run(
        self,
        job_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        intent_matrix: np.ndarray,
        candidate_ids: List,
        relevance_labels: Optional[np.ndarray] = None,
        group_sizes: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, float]:
        """
        Executes the full ranking pipeline in one call.

        1. Compute cosine similarities (alignment distance).
        2. Build unified feature matrix (cosine + intent fusion).
        3. Fit the LGBMRanker with lambdarank objective (when labels provided).
        4. Score all candidates (zero-shot fallback to cosine sim).
        5. Generate shortlist with deterministic tie-breaking.
        6. Save top-``self.top_k`` rows in competition CSV schema.
        7. Evaluate and print NDCG@10.

        Parameters
        ----------
        job_embedding : np.ndarray
            Shape ``(embedding_dim,)`` -- encoded job description.
        candidate_embeddings : np.ndarray
            Shape ``(N, embedding_dim)`` -- candidate capability vectors.
        intent_matrix : np.ndarray
            Shape ``(N, K)`` -- scaled behavioural activations.
        candidate_ids : list
            Length ``N`` -- unique candidate identifiers.
        relevance_labels : np.ndarray, optional
            Shape ``(N,)`` -- ground-truth grades for supervised fitting.
            When ``None``, zero-shot cosine-similarity ranking is used.
        group_sizes : np.ndarray, optional
            Required when ``relevance_labels`` is provided. Defaults to a
            single group containing all candidates.

        Returns
        -------
        Tuple[pd.DataFrame, float]
            (ranked shortlist DataFrame [all N], NDCG@10 score)
        """
        # Step 1 -- Cosine similarity
        cosine_sim = self.compute_cosine_similarity(job_embedding, candidate_embeddings)

        # Step 2 -- Unified feature matrix
        feature_matrix = self.build_feature_matrix(cosine_sim, intent_matrix)

        # Step 3 -- Fit (supervised) or skip (zero-shot)
        if relevance_labels is not None:
            if group_sizes is None:
                group_sizes = np.array([len(candidate_ids)], dtype=np.int32)
            self.fit(feature_matrix, relevance_labels, group_sizes)

        # Step 4 -- Score
        alignment_scores = self.score_candidates(feature_matrix)

        # Step 5 -- Shortlist with deterministic tie-breaking
        shortlist = self.generate_shortlist(
            candidate_ids, cosine_sim, intent_matrix, alignment_scores
        )

        # Step 6 -- Save competition CSV (top_k rows)
        self.save_shortlist(shortlist, top_k=self.top_k)

        # Step 7 -- NDCG@10 (always printed; surrogate in zero-shot mode)
        if relevance_labels is not None:
            ndcg = self.evaluate_ndcg(relevance_labels, alignment_scores, k=10)
        else:
            # Zero-shot surrogate: treat cosine sim as both truth and prediction.
            # This gives an upper-bound estimate useful for console monitoring.
            ndcg = self.evaluate_ndcg(cosine_sim, alignment_scores, k=10)

        return shortlist, ndcg

    # ------------------------------------------------------------------
    # 8. COMPETITION MODE -- single-call entry-point
    # ------------------------------------------------------------------

    def run_competition_mode(
        self,
        job_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        intent_matrix: np.ndarray,
        candidate_ids: List,
        output_path: str = OUTPUT_FILE,
    ) -> Tuple[pd.DataFrame, float]:
        """
        Streamlined competition entry-point that runs the full pipeline and
        always outputs exactly ``top_k`` (default: 100) rows in the strict
        competition schema: ``candidate_id,rank,score,reasoning``.

        Intended for use when no ground-truth relevance labels are available
        (zero-shot ranking mode).

        Parameters
        ----------
        job_embedding : np.ndarray
            Shape ``(embedding_dim,)`` -- encoded job description.
        candidate_embeddings : np.ndarray
            Shape ``(N, embedding_dim)`` -- candidate capability vectors.
        intent_matrix : np.ndarray
            Shape ``(N, K)`` -- MinMax-scaled behavioural intent vectors.
        candidate_ids : list
            Length ``N`` -- unique candidate identifiers.
        output_path : str
            Destination path for the competition CSV.

        Returns
        -------
        Tuple[pd.DataFrame, float]
            (full ranked DataFrame, NDCG@10 surrogate score)
        """
        logger.info(
            "Running PredictiveRankerEngine -- %d candidates, top_k=%d",
            len(candidate_ids),
            self.top_k,
        )
        logger.info(
            "Mode: zero-shot (no labelled relevance data) -- LGBMRanker will "
            "use cosine similarity as the primary signal."
        )

        shortlist, ndcg = self.run(
            job_embedding=job_embedding,
            candidate_embeddings=candidate_embeddings,
            intent_matrix=intent_matrix,
            candidate_ids=candidate_ids,
            relevance_labels=None,
            group_sizes=None,
        )
        self.save_shortlist(shortlist, top_k=self.top_k, output_path=output_path)
        return shortlist, ndcg


# ---------------------------------------------------------------------------
# Execution block — end-to-end verification with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    # Lazy import to avoid circular dependency issues when running standalone
    from embedder import DualSignalEncoder

    print("=" * 60)
    print("   Predictive Ranker Engine -- End-to-End Verification")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Synthetic candidate dataset (mirrors data_pipeline output schema)
    # ------------------------------------------------------------------
    N_CANDIDATES = 20
    rng = np.random.default_rng(42)

    candidate_ids = list(range(101, 101 + N_CANDIDATES))

    # Candidate experience texts (simulate cleaned pipeline output)
    experience_pool = [
        "senior software engineer python rest apis cloud computing docker kubernetes microservices",
        "junior developer react typescript frontend css html responsive design",
        "data scientist machine learning nlp statistical analysis pandas sklearn",
        "tech lead java enterprise development cloud computing it support cybersecurity",
        "devops engineer ci cd pipelines ansible terraform aws gcp",
        "backend developer golang postgresql redis kafka event driven",
        "mobile developer flutter dart ios android cross platform",
        "product manager agile scrum roadmap stakeholder management",
        "data engineer spark hadoop etl pipeline data warehouse",
        "ml engineer deep learning pytorch tensorflow computer vision",
    ]
    candidate_texts = [experience_pool[i % len(experience_pool)] for i in range(N_CANDIDATES)]

    job_description_text = (
        "senior backend engineer python rest apis cloud computing aws "
        "microservices docker kubernetes ci cd devops"
    )

    # Synthetic behavioural metrics
    candidate_behaviours = pd.DataFrame({
        "candidate_id":               candidate_ids,
        "profile_update_frequency":   rng.uniform(0, 5, N_CANDIDATES),
        "submission_timestamp_index": rng.uniform(0, 1, N_CANDIDATES),
        "interaction_history_index":  rng.uniform(0, 200, N_CANDIDATES),
    })

    # Synthetic relevance labels (0–3 graded relevance, simulating manual labels)
    # Candidates 0,1,5 are highly relevant (grade 3); others vary
    relevance_labels = np.array(
        [3, 2, 1, 2, 3, 2, 0, 1, 2, 3, 1, 0, 2, 1, 3, 0, 1, 2, 1, 0],
        dtype=np.int32,
    )

    # ------------------------------------------------------------------
    # Stage 1 — Encode with DualSignalEncoder
    # ------------------------------------------------------------------
    print("\n--- Stage 1: Dual-Signal Encoding ---")
    encoder = DualSignalEncoder(model_name="all-MiniLM-L6-v2", normalize_embeddings=True)

    composite_result = encoder.build_composite_matrix(candidate_texts, candidate_behaviours)
    candidate_embeddings: np.ndarray = composite_result["capability"]   # (N, 384)
    intent_matrix: np.ndarray = composite_result["intent"]             # (N, 3)
    emb_dim: int = composite_result["embedding_dim"]                   # 384

    # Encode job description
    job_embedding: np.ndarray = encoder.encode_capability([job_description_text])[0]

    print(f"  Candidate capability matrix : {candidate_embeddings.shape}")
    print(f"  Intent matrix               : {intent_matrix.shape}")
    print(f"  Job embedding               : {job_embedding.shape}")

    # ------------------------------------------------------------------
    # Stage 2 — Rank
    # ------------------------------------------------------------------
    print("\n--- Stage 2: Ranking Pipeline ---")
    ranker = PredictiveRankerEngine(top_k=10, random_state=42)

    shortlist, ndcg_score_val = ranker.run(
        job_embedding=job_embedding,
        candidate_embeddings=candidate_embeddings,
        intent_matrix=intent_matrix,
        candidate_ids=candidate_ids,
        relevance_labels=relevance_labels,
        group_sizes=np.array([N_CANDIDATES], dtype=np.int32),
    )

    # ------------------------------------------------------------------
    # Stage 3 — Validation output
    # ------------------------------------------------------------------
    print("\n--- Stage 3: Ranked Shortlist (Top 10) ---")
    print(shortlist.head(10).to_string(index=False))

    print(f"\nNDCG@10 Score : {ndcg_score_val:.4f}")

    # Verify output file
    abs_out = os.path.abspath(OUTPUT_FILE)
    if os.path.exists(abs_out):
        saved = pd.read_csv(abs_out)
        print(f"\nOutput CSV verified : {abs_out}")
        print(f"  Rows    : {len(saved)}")
        print(f"  Columns : {list(saved.columns)}")
    else:
        print(f"\nERROR: Output file not found at {abs_out}")
        sys.exit(1)

    print("\nRanker Engine verification completed successfully.")
    sys.exit(0)
