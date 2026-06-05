"""
embedder.py
-----------
Dual-signal tensor representation layer for the Syntact TalentEngine.

Two independent signal channels:
  1. CAPABILITY  – dense semantic embeddings of experience/project text,
                   produced by a lightweight sentence-transformer model.
  2. INTENT      – normalised numerical matrix of platform behavioural
                   metrics (profile update frequency, submission cadence,
                   interaction history indices).

Both channels are concatenated into a composite feature matrix that
preserves independent distance semantics for downstream ranking models.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from typing import List, Optional

from tqdm import tqdm

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_MODEL: str = "all-MiniLM-L6-v2"

# Behavioural metric columns sourced from the engineered DataFrame
# (output of TalentDataPipeline.engineer_features).
# Absent columns are zero-filled before scaling.
_BEHAVIOURAL_COLS: List[str] = [
    "profile_update_frequency",   # Proxied from profile_completeness_score
    "submission_timestamp_index", # Normalised recency score
    "interaction_history_index",  # Proxied from connection_count
]

# Target columns extracted directly from the raw `redrob_signals` dictionary.
# These are the canonical behavioral intent signal keys for encode_intent_from_signals().
# Missing keys are zero-filled; boolean flags are cast to 0.0 / 1.0.
_REDROB_SIGNAL_COLS: List[str] = [
    "profile_completeness_score",   # float  [0, 1]   — profile fill rate
    "open_to_work_flag",            # bool   → float  — active job-seeking signal
    "profile_views_received_30d",   # int             — inbound recruiter interest
    "applications_submitted_30d",   # int             — outbound activity cadence
    "recruiter_response_rate",      # float  [0, 1]   — quality signal
    "avg_response_time_hours",      # float           — responsiveness (lower = better)
    "connection_count",             # int             — network density
    "notice_period_days",           # int             — availability latency
    "github_activity_score",        # float           — code portfolio engagement
    "interview_completion_rate",    # float  [0, 1]   — pipeline completion rate
    "offer_acceptance_rate",        # float  [0, 1]   — commitment signal
]


class DualSignalEncoder:
    """
    Encodes candidate profiles into two independent signal matrices:

    - **Capability matrix** (`np.ndarray`, shape ``[N, embedding_dim]``):
      Semantic dense vectors derived from experience and project text via a
      pre-trained sentence-transformer.

    - **Intent matrix** (`np.ndarray`, shape ``[N, n_behavioural_features]``):
      Min-max scaled platform behavioural activations.

    Both matrices are concatenated along axis-1 to produce a composite
    feature matrix that preserves the independent distance geometry of
    each signal channel.

    Parameters
    ----------
    model_name : str
        HuggingFace / sentence-transformers model identifier.
        Defaults to ``all-MiniLM-L6-v2`` (22 M params, 384-dim embeddings).
    device : str, optional
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        Auto-detected when ``None`` (default).
    batch_size : int
        Batch size used during sentence-transformer inference.
    normalize_embeddings : bool
        Whether to L2-normalise the capability embeddings so that cosine
        similarity equals dot-product similarity. Defaults to ``True``.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        cache_filename: str = "capability_embeddings.npy",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        # Build instance-level cache path from cache_filename.
        # Resolved relative to project root (parent of src/) so it always
        # points to data/ regardless of cwd.
        #   Production : cache_filename = "capability_embeddings.npy"
        #   Trial run  : cache_filename = "trial_embeddings.npy"
        _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._cache_path: str = os.path.join(_proj_root, "data", cache_filename)

        logger.info("Loading sentence-transformer model: '%s' ...", model_name)
        self._encoder = SentenceTransformer(model_name, device=device)
        self.embedding_dim: int = self._encoder.get_embedding_dimension()
        logger.info("Model loaded -- embedding dimension: %d", self.embedding_dim)

        # Pin PyTorch intra-op thread pool to _ENCODE_NUM_WORKERS cores.
        # This is the correct CPU parallelism lever for sentence-transformers
        # v3+ on Windows (num_workers kwarg was removed from encode()).
        try:
            import torch
            n_threads = min(self._ENCODE_NUM_WORKERS, os.cpu_count() or 1)
            torch.set_num_threads(n_threads)
            logger.info("PyTorch intra-op threads set to %d.", n_threads)
        except Exception:
            pass  # torch not available — no-op; CPU count still used

        # MinMaxScaler for the DataFrame-based encode_intent() path.
        # Fitted on first call; reusable for identical-schema batches.
        self._scaler: MinMaxScaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self._scaler_fitted: bool = False

        # Separate scaler for the raw-dict encode_intent_from_signals() path
        # so the two code paths never share scaler state.
        self._signal_scaler: MinMaxScaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self._signal_scaler_fitted: bool = False

        # Names of the behavioural features used in the last encode_intent call.
        # Populated automatically; use this for column labelling downstream.
        self._intent_feature_names: List[str] = []
        # Names from the last encode_intent_from_signals call.
        self._signal_feature_names: List[str] = list(_REDROB_SIGNAL_COLS)

    # ------------------------------------------------------------------
    # 1. CAPABILITY SIGNAL — semantic dense embeddings
    # ------------------------------------------------------------------

    # Chunk size for the streaming encode loop. 1000 texts per chunk keeps
    # peak RAM flat regardless of corpus size (100k+).
    _CHUNK_SIZE: int = 1_000

    # Number of CPU cores to hand to PyTorch's intra-op thread pool.
    # Set via torch.set_num_threads() in __init__; sentence-transformers v3+
    # removed num_workers from encode() so this is the correct CPU parallelism
    # lever on Windows.
    _ENCODE_NUM_WORKERS: int = 4

    # _CACHE_PATH is intentionally NOT a class attribute here.
    # It is constructed per-instance in __init__ from the cache_filename
    # argument so that trial runs and production runs maintain separate
    # cache files without any class-level mutation.

    def encode_capability(
        self,
        text_list: List[str],
        is_job_desc: bool = False,
    ) -> np.ndarray:
        """
        Encodes a list of free-text strings into dense semantic embeddings
        using a chunked, tqdm-monitored streaming strategy.

        Processing strategy
        -------------------
        1. Separate empty strings (→ zero vectors) from valid texts so the
           model never sees meaningless padding.
        2. Slice valid texts into chunks of ``_CHUNK_SIZE`` (default 1 000).
        3. Encode each chunk with ``batch_size=64``; PyTorch intra-op threads
           are set to ``_ENCODE_NUM_WORKERS`` (4) in ``__init__`` via
           ``torch.set_num_threads()`` to utilise all available CPU cores.
        4. Vertically stack all chunk arrays with ``np.vstack()`` to assemble
           the final ``(N, embedding_dim)`` matrix.
        5. Scatter embeddings back into the pre-allocated output buffer using
           the original index mapping so output order always matches input order.

        A tqdm progress bar is shown in the terminal for every call where
        ``len(valid_texts) > _CHUNK_SIZE`` so large 100k runs remain
        observable.

        Parameters
        ----------
        text_list : List[str]
            Iterable of ``N`` text strings, one per candidate.
        is_job_desc : bool, optional
            When ``True``, the call is treated as a single job-description
            embedding request.  The disk cache is bypassed entirely — no
            read, no write — so ``capability_embeddings.npy`` (or
            ``trial_embeddings.npy``) is never corrupted.  Default ``False``
            (candidate corpus path with full cache read/write).

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(N, embedding_dim)``.
            L2-normalised when ``normalize_embeddings=True``.

        Raises
        ------
        ValueError
            If ``text_list`` is empty.
        """
        if not text_list:
            raise ValueError("text_list must contain at least one string.")

        # ------------------------------------------------------------------
        # 0. Disk-cache check — skip encoding entirely on repeated runs.
        #    BYPASSED when is_job_desc=True: the single job-description text
        #    is encoded purely in RAM and never touches the candidate cache
        #    file, preventing any shape-mismatch overwrite.
        # ------------------------------------------------------------------
        cache_path: str = self._cache_path
        if not is_job_desc and os.path.exists(cache_path):
            cached: np.ndarray = np.load(cache_path, allow_pickle=False)
            if cached.shape == (len(text_list), self.embedding_dim):
                logger.info(
                    "Cache hit — loading capability embeddings from disk: %s  shape=%s",
                    cache_path,
                    cached.shape,
                )
                return cached.astype(np.float32)
            else:
                logger.warning(
                    "Cache shape %s does not match expected (%d, %d) — "
                    "re-encoding and overwriting cache.",
                    cached.shape,
                    len(text_list),
                    self.embedding_dim,
                )

        indices_empty: List[int] = [i for i, t in enumerate(text_list) if not t.strip()]
        indices_valid: List[int] = [i for i, t in enumerate(text_list) if t.strip()]
        valid_texts: List[str]   = [text_list[i] for i in indices_valid]

        # Pre-allocate output buffer (zero rows will stay zero for empty strings)
        result = np.zeros((len(text_list), self.embedding_dim), dtype=np.float32)

        if valid_texts:
            n_valid  = len(valid_texts)
            n_chunks = math.ceil(n_valid / self._CHUNK_SIZE)

            logger.info(
                "Encoding %d capability texts — chunk_size=%d, n_chunks=%d, "
                "batch_size=64, torch_threads=%d, normalize=%s",
                n_valid,
                self._CHUNK_SIZE,
                n_chunks,
                self._ENCODE_NUM_WORKERS,
                self.normalize_embeddings,
            )

            # ------------------------------------------------------------------
            # 2-3. Chunked encode loop with tqdm progress bar
            # ------------------------------------------------------------------
            chunk_arrays: List[np.ndarray] = []

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                for chunk_start in tqdm(
                    range(0, n_valid, self._CHUNK_SIZE),
                    desc="Encoding capability",
                    unit="chunk",
                    total=n_chunks,
                    dynamic_ncols=True,
                ):
                    chunk = valid_texts[chunk_start : chunk_start + self._CHUNK_SIZE]
                    chunk_emb: np.ndarray = self._encoder.encode(
                        chunk,
                        batch_size=64,
                        normalize_embeddings=self.normalize_embeddings,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
                    chunk_arrays.append(chunk_emb.astype(np.float32))

            # ------------------------------------------------------------------
            # 4. Stack all chunk arrays into one contiguous matrix
            # ------------------------------------------------------------------
            all_embeddings: np.ndarray = np.vstack(chunk_arrays)  # (n_valid, embedding_dim)

            # ------------------------------------------------------------------
            # 5. Scatter back into output buffer preserving original index order
            # ------------------------------------------------------------------
            for out_idx, emb_row in zip(indices_valid, all_embeddings):
                result[out_idx] = emb_row

            logger.info("Capability encoding complete — output shape: %s", result.shape)

            # ------------------------------------------------------------------
            # 6. Persist to disk cache — BYPASSED when is_job_desc=True.
            #    Only candidate corpus embeddings are written to disk.
            # ------------------------------------------------------------------
            if not is_job_desc:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    np.save(cache_path, result)
                    logger.info(
                        "Capability embeddings cached to disk: %s  shape=%s",
                        cache_path,
                        result.shape,
                    )
                except Exception as exc:
                    logger.warning("Cache write failed (non-fatal): %s", exc)

        if indices_empty:
            logger.warning(
                "%d empty text(s) — zero vectors assigned (indices not shown for brevity).",
                len(indices_empty),
            )

        return result  # shape: (N, embedding_dim)

    # ------------------------------------------------------------------
    # 2. INTENT SIGNAL — behavioural metric normalisation
    # ------------------------------------------------------------------

    def encode_intent(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Extracts and scales platform behavioural metrics into a normalised
        numerical activation matrix.

        Expected behavioural columns (zero-filled if absent):
          - ``profile_update_frequency``   : updates per calendar month
          - ``submission_timestamp_index`` : recency score (raw or derived)
          - ``interaction_history_index``  : aggregated engagement index

        Additional numeric columns beyond the three defaults are
        automatically included if they are present in the DataFrame,
        giving callers flexibility to extend the behavioural feature set.

        The ``MinMaxScaler`` is fitted on the first call and reused on
        subsequent calls to support streaming / incremental inference.
        Pass ``refit=True`` to force a refit on new data distributions.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Candidate profile frame containing at least the behavioural
            metric columns. Non-numeric columns are silently ignored.

        Returns
        -------
        np.ndarray
            Float64 array of shape ``(N, n_behavioural_features)``
            with all values in ``[0.0, 1.0]``.

        Raises
        ------
        ValueError
            If the resulting behavioural feature matrix is empty.
        """
        if dataframe is None or dataframe.empty:
            raise ValueError("dataframe must be a non-empty pd.DataFrame.")

        df = dataframe.copy()

        # Ensure all expected behavioural columns exist; zero-fill if missing
        for col in _BEHAVIOURAL_COLS:
            if col not in df.columns:
                logger.warning(
                    "Behavioural column '%s' not found — zero-filling.", col
                )
                df[col] = 0.0

        # Collect all numeric columns that carry behavioural signal.
        # Priority: declared defaults first, then any additional numeric cols.
        # Columns whose names end with '_id' or equal 'id' are excluded as
        # they are row identifiers, not behavioural signals.
        extra_numeric: List[str] = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in _BEHAVIOURAL_COLS
            and not (c.lower() == 'id' or c.lower().endswith('_id'))
        ]
        behavioural_cols_present: List[str] = _BEHAVIOURAL_COLS + extra_numeric

        # Persist for downstream introspection (e.g. column labelling)
        self._intent_feature_names = behavioural_cols_present

        feature_matrix: np.ndarray = df[behavioural_cols_present].fillna(0.0).values.astype(np.float64)

        if feature_matrix.size == 0:
            raise ValueError("Behavioural feature matrix is empty after column selection.")

        # Fit-or-transform depending on scaler state
        if not self._scaler_fitted:
            logger.info(
                "Fitting MinMaxScaler on %d behavioural features across %d candidates …",
                feature_matrix.shape[1],
                feature_matrix.shape[0],
            )
            scaled: np.ndarray = self._scaler.fit_transform(feature_matrix)
            self._scaler_fitted = True
        else:
            logger.info("Reusing fitted MinMaxScaler for intent encoding.")
            scaled = self._scaler.transform(feature_matrix)

        logger.info(
            "Intent matrix encoded — shape: %s, features: %s",
            scaled.shape,
            behavioural_cols_present,
        )
        return scaled  # shape: (N, n_behavioural_features)

    # ------------------------------------------------------------------
    # 2b. INTENT SIGNAL (raw dict path) — redrob_signals ingestion
    # ------------------------------------------------------------------

    def encode_intent_from_signals(
        self,
        list_of_candidate_signals: List[dict],
    ) -> np.ndarray:
        """
        Extracts and scales behavioral intent from a list of raw
        ``redrob_signals`` dictionaries (one per candidate).

        Processes exactly the 11 target columns defined in
        ``_REDROB_SIGNAL_COLS``:

        .. code-block:: text

            profile_completeness_score   open_to_work_flag
            profile_views_received_30d   applications_submitted_30d
            recruiter_response_rate      avg_response_time_hours
            connection_count             notice_period_days
            github_activity_score        interview_completion_rate
            offer_acceptance_rate

        Missing / null entries are zero-filled.  Boolean flags are cast to
        ``1.0`` (True) or ``0.0`` (False).  All columns are scaled to
        ``[0.0, 1.0]`` via a ``MinMaxScaler`` (fitted on the first call,
        reused thereafter).

        Parameters
        ----------
        list_of_candidate_signals : List[dict]
            Length ``N`` — each element is the ``redrob_signals`` sub-dict
            from a raw candidate JSON record.  Non-dict elements are treated
            as empty dicts (all columns zero-filled).

        Returns
        -------
        np.ndarray
            Float64 array of shape ``(N, 11)`` with all values in
            ``[0.0, 1.0]`` representing the Behavioral Intent Vector.

        Raises
        ------
        ValueError
            If ``list_of_candidate_signals`` is empty.
        """
        if not list_of_candidate_signals:
            raise ValueError("list_of_candidate_signals must contain at least one element.")

        rows: List[List[float]] = []
        for signals in list_of_candidate_signals:
            if not isinstance(signals, dict):
                signals = {}
            row: List[float] = []
            for col in _REDROB_SIGNAL_COLS:
                raw = signals.get(col)
                if raw is None:
                    row.append(0.0)
                elif isinstance(raw, bool):
                    row.append(1.0 if raw else 0.0)
                else:
                    try:
                        row.append(float(raw))
                    except (ValueError, TypeError):
                        row.append(0.0)
            rows.append(row)

        feature_matrix: np.ndarray = np.array(rows, dtype=np.float64)

        # Replace any residual NaN/Inf with 0
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if not self._signal_scaler_fitted:
            logger.info(
                "Fitting signal MinMaxScaler on %d redrob features across %d candidates ...",
                feature_matrix.shape[1],
                feature_matrix.shape[0],
            )
            scaled: np.ndarray = self._signal_scaler.fit_transform(feature_matrix)
            self._signal_scaler_fitted = True
        else:
            logger.info("Reusing fitted signal MinMaxScaler for intent encoding.")
            scaled = self._signal_scaler.transform(feature_matrix)

        self._signal_feature_names = list(_REDROB_SIGNAL_COLS)
        logger.info(
            "Behavioral Intent Vector encoded -- shape: %s, features: %s",
            scaled.shape,
            self._signal_feature_names,
        )
        return scaled  # shape: (N, 11)

    # ------------------------------------------------------------------
    # 3. COMPOSITE MATRIX — concatenated dual signal
    # ------------------------------------------------------------------

    def build_composite_matrix(
        self,
        text_list: List[str],
        dataframe: pd.DataFrame,
    ) -> dict:
        """
        Generates all three matrices from a single call:

        - ``capability``  : ``(N, embedding_dim)`` semantic embeddings
        - ``intent``      : ``(N, n_behavioural)`` scaled behavioural activations
        - ``composite``   : ``(N, embedding_dim + n_behavioural)`` concatenation

        The composite matrix concatenates the two independent signal channels
        along axis-1. Downstream models can slice by ``embedding_dim`` to
        recover each channel for independent distance computations.

        Parameters
        ----------
        text_list : List[str]
            Experience / project text per candidate (length ``N``).
        dataframe : pd.DataFrame
            Candidate profile frame with behavioural metric columns (``N`` rows).

        Returns
        -------
        dict with keys:
            ``"capability"``  -> ``np.ndarray``  shape ``(N, embedding_dim)``
            ``"intent"``      -> ``np.ndarray``  shape ``(N, n_behavioural)``
            ``"composite"``   -> ``np.ndarray``  shape ``(N, embedding_dim + n_behavioural)``
            ``"embedding_dim"`` -> ``int``  boundary index for channel slicing

        Raises
        ------
        ValueError
            If the number of texts does not match the number of DataFrame rows.
        """
        if len(text_list) != len(dataframe):
            raise ValueError(
                f"Mismatch: text_list has {len(text_list)} entries but "
                f"dataframe has {len(dataframe)} rows. They must be equal."
            )

        capability_matrix: np.ndarray = self.encode_capability(text_list)
        intent_matrix: np.ndarray = self.encode_intent(dataframe)

        composite_matrix: np.ndarray = np.concatenate(
            [capability_matrix, intent_matrix.astype(np.float32)],
            axis=1,
        )

        logger.info(
            "Composite matrix built -- capability: %s | intent: %s | composite: %s",
            capability_matrix.shape,
            intent_matrix.shape,
            composite_matrix.shape,
        )

        return {
            "capability": capability_matrix,
            "intent": intent_matrix,
            "composite": composite_matrix,
            "embedding_dim": self.embedding_dim,
        }

    def build_composite_from_signals(
        self,
        text_list: List[str],
        list_of_candidate_signals: List[dict],
    ) -> dict:
        """
        End-to-end composite builder using raw ``redrob_signals`` dicts
        instead of an engineered DataFrame.  Suitable for direct JSON
        pipeline integration without a prior TalentDataPipeline pass.

        Parameters
        ----------
        text_list : List[str]
            Capability text per candidate (length ``N``).
        list_of_candidate_signals : List[dict]
            Raw ``redrob_signals`` dicts per candidate (length ``N``).

        Returns
        -------
        dict  (same schema as ``build_composite_matrix``)
        """
        if len(text_list) != len(list_of_candidate_signals):
            raise ValueError(
                f"Mismatch: text_list has {len(text_list)} entries but "
                f"list_of_candidate_signals has {len(list_of_candidate_signals)} elements."
            )

        capability_matrix: np.ndarray = self.encode_capability(text_list)
        intent_matrix: np.ndarray = self.encode_intent_from_signals(list_of_candidate_signals)

        composite_matrix: np.ndarray = np.concatenate(
            [capability_matrix, intent_matrix.astype(np.float32)],
            axis=1,
        )

        logger.info(
            "Composite (signals path) built -- capability: %s | intent: %s | composite: %s",
            capability_matrix.shape,
            intent_matrix.shape,
            composite_matrix.shape,
        )

        return {
            "capability": capability_matrix,
            "intent": intent_matrix,
            "composite": composite_matrix,
            "embedding_dim": self.embedding_dim,
        }


# ---------------------------------------------------------------------------
# Execution block — end-to-end verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("   Dual-Signal Encoder — End-to-End Verification")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Mock candidate data
    # ------------------------------------------------------------------
    candidate_texts: List[str] = [
        "Senior software engineer with 8 years experience in python rest apis cloud computing docker kubernetes",
        "Junior developer skilled in react typescript dot net development and network administration",
        "Data scientist focused on machine learning nlp data analysis statistical analysis",
        "Tech lead with expertise in java development cloud computing it support and cybersecurity",
        "",  # Candidate with no experience text (tests zero-vector fallback)
    ]

    candidate_behaviours: pd.DataFrame = pd.DataFrame({
        "candidate_id":             [101, 102, 103, 104, 105],
        "profile_update_frequency": [3.5, 1.0, 2.2, 4.8, 0.0],    # updates/month
        "submission_timestamp_index": [0.95, 0.40, 0.72, 0.88, 0.10],  # recency score
        "interaction_history_index":  [120.0, 30.0, 75.0, 200.0, 5.0],  # engagement score
    })

    # ------------------------------------------------------------------
    # Initialise encoder
    # ------------------------------------------------------------------
    encoder = DualSignalEncoder(
        model_name="all-MiniLM-L6-v2",
        batch_size=16,
        normalize_embeddings=True,
    )

    # ------------------------------------------------------------------
    # Capability embeddings
    # ------------------------------------------------------------------
    print("\n--- 1. Capability Embeddings ---")
    cap_matrix: np.ndarray = encoder.encode_capability(candidate_texts)
    print(f"  Shape : {cap_matrix.shape}")
    print(f"  dtype : {cap_matrix.dtype}")
    print(f"  L2 norms (expect ~1.0 for non-empty): "
          f"{np.linalg.norm(cap_matrix, axis=1).round(4).tolist()}")

    # ------------------------------------------------------------------
    # Intent matrix
    # ------------------------------------------------------------------
    print("\n--- 2. Intent (Behavioural) Matrix ---")
    intent_matrix: np.ndarray = encoder.encode_intent(candidate_behaviours)
    print(f"  Shape : {intent_matrix.shape}")
    print(f"  dtype : {intent_matrix.dtype}")
    intent_df = pd.DataFrame(
        intent_matrix,
        columns=encoder._intent_feature_names,
        index=candidate_behaviours["candidate_id"],
    )
    print(intent_df.round(4).to_string())

    # ------------------------------------------------------------------
    # Composite matrix
    # ------------------------------------------------------------------
    print("\n--- 3. Composite Matrix ---")
    result: dict = encoder.build_composite_matrix(candidate_texts, candidate_behaviours)
    comp = result["composite"]
    emb_dim = result["embedding_dim"]
    print(f"  Composite shape       : {comp.shape}")
    print(f"  Capability slice [:, :{emb_dim}] shape : {comp[:, :emb_dim].shape}")
    print(f"  Intent slice     [:, {emb_dim}:] shape : {comp[:, emb_dim:].shape}")
    print(f"  Global min / max      : {comp.min():.4f} / {comp.max():.4f}")
    print(f"  NaN count             : {int(np.isnan(comp).sum())}")

    # ------------------------------------------------------------------
    # Pairwise cosine distance between capability vectors
    # ------------------------------------------------------------------
    print("\n--- 4. Pairwise Capability Cosine Similarity (non-empty candidates) ---")
    cap_valid = result["capability"][:4]  # exclude empty-text candidate
    cosine_sim = cap_valid @ cap_valid.T
    sim_df = pd.DataFrame(
        cosine_sim.round(4),
        index=[f"C{i+101}" for i in range(4)],
        columns=[f"C{i+101}" for i in range(4)],
    )
    print(sim_df.to_string())

    print("\nDual-Signal Encoder verification completed successfully.")
    sys.exit(0)
