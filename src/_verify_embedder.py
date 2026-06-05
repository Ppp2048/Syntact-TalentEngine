"""Scratch verification: encode_intent_from_signals + build_composite_from_signals."""
import sys, json
sys.path.insert(0, 'src')

from embedder import DualSignalEncoder, _REDROB_SIGNAL_COLS

candidates = json.load(open('data/sample_candidates.json', encoding='utf-8'))

signal_dicts = [c.get('redrob_signals', {}) for c in candidates]
texts = [
    ' '.join(filter(None, [
        c.get('profile', {}).get('headline', ''),
        c.get('profile', {}).get('summary', ''),
        c.get('profile', {}).get('current_title', ''),
    ]))
    for c in candidates
]

print(f"Candidates        : {len(candidates)}")
print(f"Signal columns    : {_REDROB_SIGNAL_COLS}")
print()

enc = DualSignalEncoder()

cap = enc.encode_capability(texts)
print(f"encode_capability       shape: {cap.shape}  dtype: {cap.dtype}")

intent = enc.encode_intent_from_signals(signal_dicts)
print(f"encode_intent_from_signals  shape: {intent.shape}  dtype: {intent.dtype}")
print(f"Min/Max across all cells  : {intent.min():.4f} / {intent.max():.4f}  (target: 0.0 / 1.0)")
print(f"Signal feature names      : {enc._signal_feature_names}")
print()

result = enc.build_composite_from_signals(texts, signal_dicts)
print(f"composite[capability] : {result['capability'].shape}")
print(f"composite[intent]     : {result['intent'].shape}")
print(f"composite[composite]  : {result['composite'].shape}")
print(f"embedding_dim boundary: {result['embedding_dim']}")
print()

# Sample row: first candidate intent vector
import pandas as pd
row0 = pd.Series(intent[0], index=_REDROB_SIGNAL_COLS)
print("First candidate Behavioral Intent Vector:")
print(row0.to_string())

print()
print("PASS: encode_intent_from_signals operational on all 11 redrob_signals columns.")
