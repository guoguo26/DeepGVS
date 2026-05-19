#!/usr/bin/env python3
"""CDS / k-mer 等核酸特征提取（Methods: CDS feature extraction）。"""

from __future__ import annotations

import argparse
import math
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


_COMP = str.maketrans("ACGT", "TGCA")
_LUT_NT = np.full(256, -1, dtype=np.int16)
_LUT_NT[ord("A")] = 0
_LUT_NT[ord("C")] = 1
_LUT_NT[ord("G")] = 2
_LUT_NT[ord("T")] = 3
_COMP4 = np.array([3, 2, 1, 0], dtype=np.int64)
_STOPS = {"TAA", "TAG", "TGA"}
_START = "ATG"


def reverse_complement(seq: str) -> str:
    return seq.upper().translate(_COMP)[::-1]


def iter_kmers_fixed(k: int) -> List[str]:
    return ["".join(p) for p in product("ACGT", repeat=k)]


_KMER_CACHE: Dict[int, List[str]] = {k: iter_kmers_fixed(k) for k in (2, 3, 4, 5)}


def seq_to_codes(seq: str) -> np.ndarray:
    b = seq.upper().encode("ascii", errors="ignore")
    raw = np.frombuffer(b, dtype=np.uint8)
    return np.take(_LUT_NT, raw.astype(np.int64))


def shannon_entropy_nt_from_codes(codes: np.ndarray) -> float:
    if codes.size == 0:
        return 0.0
    x = codes.astype(np.int16)
    x = np.where(x < 0, np.int16(4), x)
    bc = np.bincount(x.astype(np.int64), minlength=5).astype(np.float64)
    tot = float(bc.sum())
    if tot <= 0:
        return 0.0
    h = 0.0
    for v in bc:
        if v <= 0:
            continue
        p = v / tot
        h -= p * math.log2(p)
    return float(h)


def base_counts_from_codes(codes: np.ndarray) -> np.ndarray:
    valid = codes[codes >= 0]
    if valid.size == 0:
        return np.zeros(4, dtype=np.float64)
    bc = np.bincount(valid.astype(np.int64), minlength=4).astype(np.float64)
    return bc


def kmer_freq_vectorized(codes: np.ndarray, k: int) -> np.ndarray:
    size = 4**k
    n = int(codes.shape[0])
    if n < k:
        return np.zeros(size, dtype=np.float64)

    if k == 2:
        a, b = codes[:-1], codes[1:]
        valid = (a >= 0) & (b >= 0)
        idx = (a[valid] * 4 + b[valid]).astype(np.int64)
    elif k == 3:
        a, b, c = codes[:-2], codes[1:-1], codes[2:]
        valid = (a >= 0) & (b >= 0) & (c >= 0)
        idx = (a[valid] * 16 + b[valid] * 4 + c[valid]).astype(np.int64)
    elif k == 4:
        a, b, c, d = codes[:-3], codes[1:-2], codes[2:-1], codes[3:]
        valid = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
        idx = (a[valid] * 64 + b[valid] * 16 + c[valid] * 4 + d[valid]).astype(np.int64)
    elif k == 5:
        a, b, c, d, e = codes[:-4], codes[1:-3], codes[2:-2], codes[3:-1], codes[4:]
        valid = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0) & (e >= 0)
        idx = (a[valid] * 256 + b[valid] * 64 + c[valid] * 16 + d[valid] * 4 + e[valid]).astype(np.int64)
    else:
        raise ValueError("Only k=2,3,4,5 are supported.")

    counts = np.bincount(idx, minlength=size).astype(np.float64)
    tot = float(counts.sum())
    if tot > 0:
        counts /= tot
    return counts


def _kmer_counts_map(seq: str, k: int, *, collapse_rc: bool = False) -> Dict[str, int]:
    seq = seq.upper()
    out: Dict[str, int] = {}
    for i in range(0, len(seq) - k + 1):
        mer = seq[i : i + k]
        if all(c in "ACGT" for c in mer):
            key = min(mer, reverse_complement(mer)) if collapse_rc else mer
            out[key] = out.get(key, 0) + 1
    return out


def palindrome_4mer_fraction_from_codes(codes: np.ndarray) -> float:
    if codes.shape[0] < 4:
        return 0.0
    a, b, c, d = codes[:-3], codes[1:-2], codes[2:-1], codes[3:]
    valid = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    pal = (a == _COMP4[d]) & (b == _COMP4[c]) & valid
    tot = int(valid.sum())
    if tot == 0:
        return 0.0
    return float(pal.sum()) / float(tot)


def max_tandem_repeat_units(seq: str, unit_size: int) -> Tuple[int, int]:
    seq = seq.upper()
    n = len(seq)
    if n < unit_size:
        return 1, 0
    longest = 1
    regions_ge3 = 0
    i = 0
    while i <= n - unit_size:
        unit = seq[i : i + unit_size]
        if any(c not in "ACGT" for c in unit):
            i += 1
            continue
        j = i + unit_size
        reps = 1
        while j + unit_size <= n and seq[j : j + unit_size] == unit:
            reps += 1
            j += unit_size
        if reps >= 3:
            regions_ge3 += 1
        longest = max(longest, reps)
        if reps > 1:
            i = max(i + 1, j - unit_size + 1)
        else:
            i += 1
    return int(longest), int(regions_ge3)


def longest_homopolymer_runs(seq: str) -> Tuple[int, int, int, int]:
    seq = seq.upper()
    best = {"A": 0, "T": 0, "C": 0, "G": 0}
    if not seq:
        return 0, 0, 0, 0
    cur_ch = seq[0]
    cur_len = 1
    if cur_ch in best:
        best[cur_ch] = 1
    for c in seq[1:]:
        if c == cur_ch:
            cur_len += 1
            if c in best:
                best[c] = max(best[c], cur_len)
        else:
            cur_ch = c
            cur_len = 1
            if c in best:
                best[c] = max(best[c], cur_len)
    return best["A"], best["T"], best["C"], best["G"]


def _orfs_in_frame(s: str, frame: int) -> List[int]:
    s = s[frame:]
    n = (len(s) // 3) * 3
    orfs: List[int] = []
    i = 0
    while i + 2 < n:
        codon = s[i : i + 3]
        if len(codon) < 3:
            break
        if any(b not in "ACGT" for b in codon):
            i += 3
            continue
        if codon != _START:
            i += 3
            continue
        j = i + 3
        while j + 2 < n:
            c2 = s[j : j + 3]
            if len(c2) < 3:
                i += 3
                break
            if any(b not in "ACGT" for b in c2):
                i += 3
                break
            if c2 in _STOPS:
                n_codons = (j - i) // 3
                if n_codons >= 2:
                    orfs.append(n_codons)
                i = j + 3
                break
            j += 3
        else:
            i += 3
    return orfs


def orf_count_and_max_length(seq: str, min_codons: int = 15, max_len_nt: int = 8192) -> Tuple[int, int]:
    seq = seq.upper()
    if len(seq) > max_len_nt:
        seq = seq[:max_len_nt]
    rc = reverse_complement(seq)
    all_lens: List[int] = []
    for sub in (seq, rc):
        for frame in range(3):
            all_lens.extend(_orfs_in_frame(sub, frame))
    filtered = [L for L in all_lens if L >= min_codons]
    if not filtered:
        return 0, 0
    return len(filtered), int(max(filtered))


def _window_start_positions(n: int, window: int, step: int) -> List[int]:
    if n <= 0:
        return [0]
    if n <= window:
        return [0]
    step = max(1, step)
    starts = list(range(0, max(1, n - window + 1), step))
    last = n - window
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def base_composition_features(seq: str, *, orf_max_len_nt: int = 8192) -> Dict[str, float]:
    seq = seq.upper()
    codes = seq_to_codes(seq)
    L = max(len(seq), 1)
    n_a = seq.count("A")
    n_c = seq.count("C")
    n_g = seq.count("G")
    n_t = seq.count("T")
    n_n = seq.count("N")
    gc = (n_c + n_g) / L
    at = (n_a + n_t) / L
    n_frac = n_n / L
    ent = shannon_entropy_nt_from_codes(codes)
    orf_n, orf_max = orf_count_and_max_length(seq, max_len_nt=orf_max_len_nt)

    gc_skew = (n_g - n_c) / max(1, n_g + n_c)
    at_skew = (n_a - n_t) / max(1, n_a + n_t)

    return {
        "length": float(len(seq)),
        "log_length": float(math.log1p(len(seq))),
        "gc_frac": float(gc),
        "at_frac": float(at),
        "n_frac": float(n_frac),
        "gc_at_ratio": float((n_c + n_g + 1.0) / (n_a + n_t + 1.0)),
        "gc_skew": float(gc_skew),
        "at_skew": float(at_skew),
        "nt_shannon_entropy": float(ent),
        "orf_count_min15aa": float(orf_n),
        "orf_max_codons": float(orf_max),
    }


def global_kmer_features(seq: str, k: int) -> Dict[str, float]:
    codes = seq_to_codes(seq)
    vec = kmer_freq_vectorized(codes, k)
    kmers = _KMER_CACHE[k]
    return {f"k{k}_{m}": float(v) for m, v in zip(kmers, vec)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract CDS DNA features from FASTA (same schema as reference features CSV).")
    p.add_argument("--input-fasta", type=str, required=True, help="Input nucleotide FASTA (CDS sequences).")
    p.add_argument(
        "--reference-feature-csv",
        type=str,
        required=True,
        help="Reference CSV: feature column names and order (columns other than sequence_id, label, split, source).",
    )
    p.add_argument("--output-csv", type=str, required=True, help="Output CSV path (sequence_id + feature columns).")
    p.add_argument("--local-window-size", type=int, default=128)
    p.add_argument("--local-window-step", type=int, default=128)
    p.add_argument("--orf-max-len-nt", type=int, default=8192)
    p.add_argument("--progress-every", type=int, default=200, help="Print progress every N sequences (0 disables).")
    return p.parse_args()


def load_reference_feature_columns(ref_csv: Path) -> List[str]:
    df = pd.read_csv(ref_csv, nrows=1)
    meta = {"sequence_id", "label", "split", "source"}
    feat_cols = [c for c in df.columns if c not in meta]
    if not feat_cols:
        raise ValueError(f"No feature columns in reference: {ref_csv}")
    return feat_cols


def compute_local_mean_std(seq: str, window: int, step: int) -> Dict[str, float]:
    starts = _window_start_positions(len(seq), window, step)
    gc_vals: List[float] = []
    ent_vals: List[float] = []

    for s in starts:
        sub = seq[s : s + window]
        codes = seq_to_codes(sub)
        bc = base_counts_from_codes(codes)
        valid_tot = float(bc.sum())
        if valid_tot > 0:
            n_c = float(bc[1])
            n_g = float(bc[2])
            gc = float((n_c + n_g) / valid_tot)
            ent = float(shannon_entropy_nt_from_codes(codes))
        else:
            gc = 0.0
            ent = 0.0
        gc_vals.append(gc)
        ent_vals.append(ent)

    return {
        "local_gc_mean": float(np.mean(gc_vals)) if gc_vals else 0.0,
        "local_gc_std": float(np.std(gc_vals)) if gc_vals else 0.0,
        "local_entropy_mean": float(np.mean(ent_vals)) if ent_vals else 0.0,
        "local_entropy_std": float(np.std(ent_vals)) if ent_vals else 0.0,
    }


def compute_unique_4mer_ratio(seq: str) -> float:
    counts = _kmer_counts_map(seq, 4, collapse_rc=False)
    tot = float(sum(counts.values()))
    if tot <= 0:
        return 0.0
    return float(len(counts)) / tot


def compute_feature_row(
    seq_id: str,
    seq: str,
    feat_cols: List[str],
    local_window_size: int,
    local_window_step: int,
    orf_max_len_nt: int,
) -> Dict[str, float | int | str]:
    seq_u = (seq or "").upper()
    codes_all = seq_to_codes(seq_u)

    base = base_composition_features(seq_u, orf_max_len_nt=orf_max_len_nt)
    base_keep = {
        "length": base["length"],
        "gc_frac": base["gc_frac"],
        "at_frac": base["at_frac"],
        "n_frac": base["n_frac"],
        "nt_shannon_entropy": base["nt_shannon_entropy"],
        "orf_count_min15aa": base["orf_count_min15aa"],
        "orf_max_codons": base["orf_max_codons"],
    }

    km2 = global_kmer_features(seq_u, 2)
    km3 = global_kmer_features(seq_u, 3)
    km4 = global_kmer_features(seq_u, 4)

    local_feats = compute_local_mean_std(seq_u, local_window_size, local_window_step)

    longest_a, longest_t, longest_c, longest_g = longest_homopolymer_runs(seq_u)
    repeat_max_units = max_tandem_repeat_units(seq_u, unit_size=2)[0]

    repeat_feats = {
        "longest_run_A": float(longest_a),
        "longest_run_C": float(longest_c),
        "longest_run_G": float(longest_g),
        "longest_run_T": float(longest_t),
        "max_tandem_dinuc_repeats": float(repeat_max_units),
    }

    pal_frac = float(palindrome_4mer_fraction_from_codes(codes_all))
    uniq4 = compute_unique_4mer_ratio(seq_u)
    tail_feats = {
        "palindrome_4mer_frac": pal_frac,
        "unique_4mer_ratio": float(uniq4),
    }

    val_map: Dict[str, float] = {}
    val_map.update(base_keep)
    val_map.update(km2)
    val_map.update(km3)
    val_map.update(km4)
    val_map.update(local_feats)
    val_map.update(repeat_feats)
    val_map.update(tail_feats)

    out: Dict[str, float | int | str] = {"sequence_id": seq_id}
    missing = [c for c in feat_cols if c not in val_map]
    if missing:
        raise RuntimeError(
            f"Reference expects {len(missing)} columns not produced by this extractor "
            f"(e.g. {missing[:6]}). Use a reference CSV from the same pipeline, or extend compute_feature_row."
        )
    for c in feat_cols:
        out[c] = float(val_map[c])
    return out


def read_fasta_first_token_as_id(path: Path) -> List[tuple[str, str]]:
    seqs: List[tuple[str, str]] = []
    cur_id: str | None = None
    cur_seq: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs.append((cur_id, "".join(cur_seq)))
                raw = line[1:].strip()
                cur_id = raw.split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        seqs.append((cur_id, "".join(cur_seq)))
    return seqs


def main() -> None:
    args = parse_args()
    input_fasta = Path(args.input_fasta).resolve()
    ref_csv = Path(args.reference_feature_csv).resolve()
    out_csv = Path(args.output_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not input_fasta.is_file():
        raise FileNotFoundError(input_fasta)
    if not ref_csv.is_file():
        raise FileNotFoundError(ref_csv)

    feat_cols = load_reference_feature_columns(ref_csv)
    records = read_fasta_first_token_as_id(input_fasta)
    if not records:
        raise RuntimeError("No FASTA records.")

    rows: List[Dict[str, float | int | str]] = []
    for i, (seq_id, seq) in enumerate(records, start=1):
        row = compute_feature_row(
            seq_id=seq_id,
            seq=seq,
            feat_cols=feat_cols,
            local_window_size=args.local_window_size,
            local_window_step=args.local_window_step,
            orf_max_len_nt=args.orf_max_len_nt,
        )
        rows.append(row)
        pe = int(args.progress_every)
        if pe > 0 and i % pe == 0:
            print(f"[progress] {i}/{len(records)}")

    df = pd.DataFrame(rows)
    df = df[["sequence_id"] + feat_cols]
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv} shape={df.shape} (n_features={len(feat_cols)})")


if __name__ == "__main__":
    main()
