#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
wccjc_sports_filter_v2.py

Two-stage sports-domain filter for WCC-JC (Japanese–Chinese) corpus:
1) Keyword-based coarse recall
2) Optional semantic filtering using a sentence-transformers encoder and a sports centroid.

Usage examples:

  # Train centroid from corpus (only needs to be done once)
  python wccjc_sports_filter_v2.py train --root_dir wccjc --out_prefix sports_model --max_pos 8000

  # Scan corpus and output sports sentence pairs
  python wccjc_sports_filter_v2.py scan --root_dir wccjc --model_prefix sports_model --out_tsv sports_pairs.tsv --threshold 0.3
"""

import argparse
import os
import re
import json
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# -------------------- Encoding helper -------------------- #

def open_with_guess_encoding(path: str):
    """
    Simplified for your case: we KNOW all corpus files are UTF-8 with CRLF,
    so we just open them as UTF-8 and let errors surface explicitly.
    """
    return open(path, "r", encoding="utf-8", errors="strict")


# -------------------- Sports keyword filters -------------------- #

SPORTS_KEYWORDS_ZH: List[str] = [
    "篮球", "籃球",
    "足球",
    "排球",
    "网球", "網球",
    "乒乓球",
    "羽毛球",
    "棒球",
    "冰球",
    "橄榄球", "橄欖球",
    "高尔夫", "高爾夫",
    "滑雪", "滑冰",
    "田径",
    "马拉松", "馬拉松",
    "奥运", "奧運",
    "亚运", "亞運",
    "世界杯",
    "联赛", "聯賽",
    "中超", "中甲", "中乙",
    "主场", "客场",
    "赛季",
    "体育", "體育",
    "运动会", "運動會",
    "球赛", "球賽",
    "比赛", "比賽",
    "进球", "得分",
    "加时赛", "加時賽",
    "决赛", "決賽",
    "半决赛", "半決賽",
    "季后赛", "季後賽",
    "教练", "教練",
    "球员", "選手",
    "球队", "球隊",
    "冠军", "冠軍",
    "射门", "罰球", "罚球",
    "点球", "點球",
    "罚球线", "罰球線",
    "三分球", "三分线",
    "篮筐", "籃筐",
]

SPORTS_KEYWORDS_JA: List[str] = [
    "バスケ", "バスケットボール",
    "サッカー",
    "フットボール",
    "野球",
    "テニス",
    "バレーボール",
    "卓球",
    "バドミントン",
    "ゴルフ",
    "ラグビー",
    "ホッケー",
    "スキー",
    "スノーボード",
    "陸上",  # athletics
    "マラソン",
    "オリンピック",
    "ワールドカップ",
    "リーグ",
    "シーズン",
    "Jリーグ",
    "プロ野球",
    "甲子園",
    "選手",
    "監督",
    "コーチ",
    "試合",
    "試合開始",
    "前半", "後半",
    "延長戦",
    "決勝",
    "準決勝",
    "プレーオフ",
    "優勝",
    "得点",
    "ゴール",
    "シュート",
    "フリースロー",
    "スリーポイント",
]

def is_sports_sentence_zh(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return any(kw in t for kw in SPORTS_KEYWORDS_ZH)


def is_sports_sentence_ja(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return any(kw in t for kw in SPORTS_KEYWORDS_JA)


def is_sports_pair_by_keywords(ja: str, zh: str) -> bool:
    return is_sports_sentence_ja(ja) or is_sports_sentence_zh(zh)


# -------------------- File format helpers -------------------- #

LANG_SUFFIX_RE = re.compile(r"\.(ja|jp|jpn|ja_JP|jpn_JP|zh|zh_CN|zh_TW|cn|chi)$", re.IGNORECASE)

def detect_parallel_files(root_dir: str) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """
    Walk the tree and detect two kinds of sources:
      1) Split files: base.xxx.ja + base.xxx.zh (mapped via LANG_SUFFIX_RE)
      2) Single TSV/TXT files that *look* like parallel data (contains a delimiter between JA and ZH).
    Returns:
      split_pairs: base_key -> { "ja": path, "zh": path }
      tsv_files: list of candidate TSV/TXT paths
    """
    split_candidates: Dict[str, Dict[str, str]] = {}
    tsv_files: List[str] = []

    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            path = os.path.join(dirpath, fname)
            # Skip obvious non-text stuff
            lower = fname.lower()
            if lower.endswith((".zip", ".tar", ".gz", ".bz2", ".xz", ".7z")):
                continue

            m = LANG_SUFFIX_RE.search(fname)
            if m:
                lang = m.group(1).lower()
                base = LANG_SUFFIX_RE.sub("", path)
                entry = split_candidates.setdefault(base, {})
                # normalize language key
                if lang.startswith("ja"):
                    entry["ja"] = path
                else:
                    entry["zh"] = path
                continue

            # Other text-y files
            if lower.endswith((".txt", ".tsv", ".csv")):
                tsv_files.append(path)

    split_pairs = {k: v for k, v in split_candidates.items() if "ja" in v and "zh" in v}
    return split_pairs, tsv_files


def iter_split_pair(ja_path: str, zh_path: str) -> Iterable[Tuple[str, str]]:
    """Yield (ja, zh) pairs from two separate files, line-aligned."""
    with open_with_guess_encoding(ja_path) as f_ja, open_with_guess_encoding(zh_path) as f_zh:
        for ja_line, zh_line in zip(f_ja, f_zh):
            yield ja_line.rstrip("\n\r"), zh_line.rstrip("\n\r")


def detect_tsv_delimiter(sample_lines: List[str]) -> Optional[str]:
    """
    Detect a delimiter for a TSV-like parallel file.
    We check a few common patterns.
    """
    # Ignore empty lines for detection
    lines = [ln for ln in sample_lines if ln.strip()]
    if not lines:
        return None

    # Count occurrences
    if any("\t" in ln for ln in lines):
        return "\t"

    if any("|||" in ln for ln in lines):
        return "|||"

    if any("@@@" in ln for ln in lines):
        return "@@@"

    return None


def iter_tsv_parallel(path: str) -> Iterable[Tuple[str, str]]:
    """
    Yield (ja, zh) pairs from a single TSV/TXT-like file.

    We try to detect a delimiter; if none is found, we skip the file.
    """
    try:
        f = open_with_guess_encoding(path)
    except OSError:
        return

    # Peek a few lines to detect delimiter
    sample: List[str] = []
    try:
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            sample.append(line)
    except UnicodeDecodeError:
        f.close()
        return

    delim = detect_tsv_delimiter(sample)
    if delim is None:
        # Not a parallel TSV we can understand
        f.close()
        return

    # Re-open to iterate from the beginning
    f.close()
    f = open_with_guess_encoding(path)

    for line in f:
        line = line.rstrip("\n\r")
        if not line.strip():
            continue
        if delim not in line:
            continue
        parts = line.split(delim)
        if len(parts) < 2:
            continue
        # Heuristic: assume first is JA, last is ZH
        ja = parts[0]
        zh = parts[-1]
        yield ja, zh

    f.close()


def iter_all_pairs(root_dir: str) -> Iterable[Tuple[str, str]]:
    """
    Iterate over all detected parallel JA/ZH sentence pairs in root_dir.
    """
    split_pairs, tsv_files = detect_parallel_files(root_dir)

    # 1) Split files
    for base, langs in split_pairs.items():
        ja_path = langs["ja"]
        zh_path = langs["zh"]
        for ja, zh in iter_split_pair(ja_path, zh_path):
            yield ja, zh

    # 2) TSV/TXT-like files
    for path in tsv_files:
        for ja, zh in iter_tsv_parallel(path):
            yield ja, zh


# -------------------- Length filter -------------------- #

def length_ok(ja: str, zh: str, min_len: int = 3, max_len: int = 200, max_ratio: float = 5.0) -> bool:
    """
    Basic length / length-ratio filtering to remove garbage pairs.
    """
    len_ja = len(ja.strip())
    len_zh = len(zh.strip())
    if len_ja < min_len or len_zh < min_len:
        return False
    if len_ja > max_len or len_zh > max_len:
        return False
    if len_ja == 0 or len_zh == 0:
        return False
    ratio = max(len_ja, len_zh) / min(len_ja, len_zh)
    if ratio > max_ratio:
        return False
    return True


# -------------------- Embedding helpers -------------------- #

def load_encoder(model_name: str):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed. Please `pip install sentence-transformers`.")
    return SentenceTransformer(model_name)


def embed_texts(encoder, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode texts into L2-normalized vectors.
    """
    embeddings = encoder.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    return embeddings / norms


# -------------------- Train command -------------------- #

def cmd_train(args):
    root_dir = args.root_dir
    out_prefix = args.out_prefix
    max_pos = args.max_pos
    encoder_name = args.encoder_name

    print(f"[INFO] Training sports centroid from root_dir={root_dir}")
    print(f"[INFO] Max positive sports pairs used: {max_pos}")

    pos_pairs: List[Tuple[str, str]] = []

    total_pairs = 0
    for ja, zh in iter_all_pairs(root_dir):
        total_pairs += 1
        if not length_ok(ja, zh, args.min_len, args.max_len, args.max_ratio):
            continue
        if is_sports_pair_by_keywords(ja, zh):
            pos_pairs.append((ja, zh))
            if len(pos_pairs) >= max_pos:
                break

    print(f"[INFO] Total parallel pairs seen (before early stop): {total_pairs}")
    print(f"[INFO] Collected {len(pos_pairs)} positive sports pairs based on keywords.")

    if not pos_pairs:
        print("[ERROR] No sports pairs were found with given keywords. "
              "You may need to expand the keyword list or check encodings.")
        return

    # Build training texts: concatenate JA and ZH
    train_texts = [f"{ja} || {zh}" for ja, zh in pos_pairs]

    print(f"[INFO] Loading sentence-transformers encoder: {encoder_name}")
    encoder = load_encoder(encoder_name)

    print("[INFO] Computing embeddings for positive sports pairs...")
    emb = embed_texts(encoder, train_texts, batch_size=args.batch_size)

    centroid = emb.mean(axis=0)
    # Normalize centroid
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    # Inspect similarity distribution on the same positives (for guidance)
    sims = emb @ centroid
    print(f"[INFO] Cosine similarity stats on positive pairs:")
    print(f"       min={sims.min():.4f}, max={sims.max():.4f}, mean={sims.mean():.4f}, median={np.median(sims):.4f}")

    # Save centroid and config
    centroid_path = out_prefix + "_centroid.npy"
    config_path = out_prefix + "_config.json"
    np.save(centroid_path, centroid)
    config = {
        "encoder_name": encoder_name,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "max_ratio": args.max_ratio,
        "sports_keywords_zh": SPORTS_KEYWORDS_ZH,
        "sports_keywords_ja": SPORTS_KEYWORDS_JA,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved centroid to {centroid_path}")
    print(f"[INFO] Saved config to   {config_path}")
    print("[INFO] For scanning, pick a threshold between (roughly) the min/median of these sims, e.g. 0.3~0.4.")


# -------------------- Scan command -------------------- #

def cmd_scan(args):
    root_dir = args.root_dir
    model_prefix = args.model_prefix
    out_tsv = args.out_tsv
    threshold = args.threshold

    centroid_path = model_prefix + "_centroid.npy"
    config_path = model_prefix + "_config.json"

    if not os.path.exists(centroid_path) or not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find centroid/config with prefix {model_prefix}")

    centroid = np.load(centroid_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    encoder_name = config["encoder_name"]
    min_len = config["min_len"]
    max_len = config["max_len"]
    max_ratio = config["max_ratio"]

    print(f"[INFO] Scanning root_dir={root_dir}")
    print(f"[INFO] Using encoder={encoder_name}")
    print(f"[INFO] Using centroid from {centroid_path}")
    print(f"[INFO] Length filter: min_len={min_len}, max_len={max_len}, max_ratio={max_ratio}")
    print(f"[INFO] Cosine similarity threshold: {threshold}")

    encoder = load_encoder(encoder_name)

    out_f = open(out_tsv, "w", encoding="utf-8")
    kept = 0
    keyword_candidates = 0
    total_pairs = 0

    batch_texts: List[str] = []
    batch_pairs: List[Tuple[str, str]] = []

    def flush_batch():
        nonlocal kept
        if not batch_texts:
            return
        emb = embed_texts(encoder, batch_texts, batch_size=args.batch_size)
        sims = emb @ centroid
        for (ja, zh), sim in zip(batch_pairs, sims):
            if sim >= threshold:
                out_f.write(f"{ja}\t{zh}\n")
                kept += 1
        batch_texts.clear()
        batch_pairs.clear()

    for ja, zh in iter_all_pairs(root_dir):
        total_pairs += 1
        if not length_ok(ja, zh, min_len, max_len, max_ratio):
            continue
        if not is_sports_pair_by_keywords(ja, zh):
            continue
        keyword_candidates += 1
        batch_pairs.append((ja, zh))
        batch_texts.append(f"{ja} || {zh}")

        if len(batch_texts) >= args.batch_size:
            flush_batch()

    flush_batch()
    out_f.close()

    print(f"[INFO] Total parallel pairs scanned: {total_pairs}")
    print(f"[INFO] Keyword-stage sports candidates: {keyword_candidates}")
    print(f"[INFO] Final kept pairs after semantic filtering: {kept}")
    print(f"[INFO] Output written to: {out_tsv}")
    if keyword_candidates < 1000:
        print("[WARN] Very few keyword candidates were found. "
              "This is suspicious for a ~7e5-sentence training set; "
              "please double-check that we are scanning the correct root_dir "
              "and that the encoding detection is working.")


# -------------------- Main CLI -------------------- #

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Sports-domain sentence mining for WCC-JC (JA-ZH). "
                    "Supports keyword-based coarse recall + semantic filtering."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Train sports centroid from corpus (uses keyword positives only).")
    p_train.add_argument("--root_dir", type=str, required=True, help="Path to wccjc root directory.")
    p_train.add_argument("--out_prefix", type=str, default="sports_model", help="Prefix for centroid/config files.")
    p_train.add_argument("--max_pos", type=int, default=8000, help="Maximum number of sports pairs used to build centroid.")
    p_train.add_argument("--min_len", type=int, default=3)
    p_train.add_argument("--max_len", type=int, default=200)
    p_train.add_argument("--max_ratio", type=float, default=5.0)
    p_train.add_argument("--encoder_name", type=str, default="paraphrase-multilingual-MiniLM-L12-v2",
                         help="Name of sentence-transformers model.")
    p_train.add_argument("--batch_size", type=int, default=64)
    p_train.set_defaults(func=cmd_train)

    # scan
    p_scan = sub.add_parser("scan", help="Scan corpus and output sports pairs.")
    p_scan.add_argument("--root_dir", type=str, required=True, help="Path to wccjc root directory.")
    p_scan.add_argument("--model_prefix", type=str, default="sports_model", help="Prefix used when training centroid.")
    p_scan.add_argument("--out_tsv", type=str, default="wccjc_sports.tsv",
                        help="Output TSV with JA<TAB>ZH sports pairs.")
    p_scan.add_argument("--threshold", type=float, default=0.3,
                        help="Cosine similarity threshold for keeping a pair.")
    p_scan.add_argument("--batch_size", type=int, default=64)
    p_scan.set_defaults(func=cmd_scan)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
