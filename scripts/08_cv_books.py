"""按作品分组的 K 折交叉验证。

为什么需要它：
  单次划分只能留出少量作品当测试集（panel 子集里 COMICS 只有 21 本书，
  单次 8:1:1 划分后测试集只分到 2 本书），据此判断泛化能力噪声极大——
  换个随机种子结果可能差十几个百分点。

  交叉验证让每个作品组都轮流当一次测试集，训练时又从未见过它，
  因此得到的是「对未见过的作品」的准确率，且用上了全部数据。

用法:
  python scripts/08_cv_books.py --subset panel --folds 5
  python scripts/08_cv_books.py --subset cover --folds 5 --epochs 12
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import random

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets" / "comic_style"
CLEAN = DS / "clean"
CV = DS / "_cv"
OUT = DS / "_eval"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def group_of(path: Path) -> str:
    """从清洗后的文件名 <tag>__<作品组>__<原名>.jpg 里取作品组。

    不能用 path.parent.name —— 那是类别目录名（japanese/western），不是作品。
    作品名本身可能含 '__'，所以取首尾之间的整段。
    """
    parts = path.stem.split("__")
    if len(parts) > 2:
        return "__".join(parts[1:-1])
    return parts[1] if len(parts) > 1 else path.stem


def read_manifest(subset):
    import csv
    with (CLEAN / "manifest.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["subset"] == subset]
    return rows


def cap_per_group(rows, cap, rng):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    out = []
    for g, items in groups.items():
        if cap and len(items) > cap:
            items = rng.sample(items, cap)
        out += items
    return out


def make_folds(rows, k, rng):
    """按作品分组做分层 K 折：每一类内部把作品组均分到各折。"""
    folds = [[] for _ in range(k)]
    for cls in sorted({r["class"] for r in rows}):
        groups = defaultdict(list)
        for r in rows:
            if r["class"] == cls:
                groups[r["group"]].append(r)
        keys = list(groups)
        rng.shuffle(keys)
        keys.sort(key=lambda g: -len(groups[g]))  # 大组优先，便于均衡
        sizes = [0] * k
        for g in keys:
            t = sizes.index(min(sizes))
            folds[t] += groups[g]
            sizes[t] += len(groups[g])
    return folds


def build_fold_dir(fold_idx, rows_all, test_rows, folds_n):
    d = CV / f"fold{fold_idx}"
    if d.exists():
        shutil.rmtree(d)
    test_keys = {id(r) for r in test_rows}
    for split, items in (("train", [r for r in rows_all if id(r) not in test_keys]),
                         ("val", test_rows),
                         ("test", test_rows)):
        for r in items:
            dst = d / split / r["class"] / Path(r["file"]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CLEAN / r["file"], dst)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="panel")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--model", default=str(ROOT / "yolo11n-cls.pt"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--cap", type=int, default=100, help="单作品最多取多少张")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from ultralytics import YOLO

    rng = random.Random(args.seed)
    rows = read_manifest(args.subset)
    rows = cap_per_group(rows, args.cap, rng)
    print(f"子集={args.subset}  样本={len(rows)}  "
          f"作品组={len({r['group'] for r in rows})}  折数={args.folds}")

    folds = make_folds(rows, args.folds, rng)
    for i, f in enumerate(folds):
        print(f"  fold{i}: {len(f)} 张, "
              f"{len({r['group'] for r in f})} 组, "
              f"类别 {dict((c, sum(1 for r in f if r['class'] == c)) for c in sorted({r['class'] for r in f}))}")

    per_book = defaultdict(lambda: defaultdict(int))  # book -> (true,pred) -> n
    fold_acc = []
    all_true, all_pred = [], []

    for i, test_rows in enumerate(folds):
        d = build_fold_dir(i, rows, test_rows, args.folds)
        print(f"\n===== fold {i + 1}/{args.folds} =====", flush=True)
        model = YOLO(args.model)
        model.train(data=str(d), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, patience=args.patience,
                    workers=args.workers, device="cpu", seed=args.seed,
                    project=str(ROOT / "runs" / "classify"), name=f"cv_{args.subset}_f{i}",
                    exist_ok=True, verbose=False)

        names = model.names
        files = [(f, c.name) for c in sorted(p for p in (d / "test").iterdir() if p.is_dir())
                 for f in sorted(c.iterdir()) if f.suffix.lower() in IMG_EXT]
        preds = [names[int(r.probs.top1)] for r in
                 model.predict([str(f) for f, _ in files], imgsz=args.imgsz,
                               verbose=False, stream=True)]
        ok = sum(1 for (_, t), p in zip(files, preds) if t == p)
        acc = ok / len(files)
        fold_acc.append(acc)
        print(f"  fold{i} 测试准确率 = {acc:.4f}  (n={len(files)})")

        for (f, t), p in zip(files, preds):
            per_book[group_of(f)][(t, p)] += 1
            all_true.append(t)
            all_pred.append(p)

    classes = sorted(set(all_true) | set(all_pred))
    total = len(all_true)
    oof = sum(1 for t, p in zip(all_true, all_pred) if t == p) / total

    print(f"\n{'=' * 60}\n按作品交叉验证结果（{args.folds} 折，每个作品都当过测试集）")
    print(f"折准确率: {[round(a, 4) for a in fold_acc]}")
    print(f"平均 {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
    print(f"合并后的样本外(OOF)准确率 = {oof:.4f}   (n={total})")

    print(f"\n各作品表现（OOF）:")
    print(f"{'book':<10}{'n':>5}{'acc':>8}   错判")
    for b in sorted(per_book, key=lambda k: -sum(per_book[k].values())):
        counts = per_book[b]
        n = sum(counts.values())
        ok = sum(v for (t, p), v in counts.items() if t == p)
        errs = " ".join(f"{t}->{p}:{v}" for (t, p), v in sorted(counts.items()) if t != p)
        print(f"{b[:9]:<10}{n:>5}{ok / n:>8.2f}   {errs}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"cv_{args.subset}.json").write_text(json.dumps({
        "subset": args.subset, "folds": args.folds, "cap": args.cap,
        "fold_acc": fold_acc, "mean": float(np.mean(fold_acc)),
        "std": float(np.std(fold_acc)), "oof_acc": oof, "n": total,
        "per_book": {b: {f"{t}->{p}": v for (t, p), v in c.items()}
                     for b, c in per_book.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT / f'cv_{args.subset}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
