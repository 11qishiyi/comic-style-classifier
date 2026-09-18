"""在**外部数据集**上验证模型——训练时完全没见过的来源。

与 06_eval.py 的区别：
  06_eval.py 跑的是本项目自己的测试集（同来源、按作品划分），衡量的是「跨作品泛化」。
  本脚本跑的是外部来源的数据，衡量的是「跨来源泛化」，是更严格的检验。

外部集通常是单一类别的（例如 Manga109 全是日漫、Garfield 全是美漫），
所以指标是「预期类别的召回率」——有多大比例被判成了它本该属于的类。

用法:
  python scripts/09_eval_external.py --model release/models/comic-style-panel-yolo11n-cls.onnx \
      --dir datasets/external/manga109 --expect japanese
  python scripts/09_eval_external.py --model release/models/comic-style-cover-yolo11n-cls.onnx \
      --dir datasets/external/garfield --expect western --limit 300
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "comic_style" / "_eval_external"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SIZE = 224


def imread_u(path, flags=cv2.IMREAD_COLOR):
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, flags) if buf.size else None


def imwrite_u(path, img, quality=92):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(str(path))
    return ok


def collect_images(d, limit, seed=0):
    files = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT)
    if limit and len(files) > limit:
        random.Random(seed).shuffle(files)
        files = sorted(files[:limit])
    return files


def montage(items, out_path, cols=8, rows=5, cell=200, label=None):
    """items: [(path, true_label, pred_label)]；在图上标注预测结果"""
    if not items:
        return
    picks = items[: cols * rows]
    sheet = np.full((rows * cell, cols * cell, 3), 32, np.uint8)
    for i, (p, t, pr) in enumerate(picks):
        img = imread_u(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = min(cell / w, cell / h)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y0 = r * cell + (cell - img.shape[0]) // 2
        x0 = c * cell + (cell - img.shape[1]) // 2
        sheet[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        color = (0, 200, 0) if t == pr else (0, 0, 255)
        cv2.rectangle(sheet, (c * cell, r * cell), (c * cell + cell - 1, r * cell + cell - 1), color, 3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_u(out_path, sheet)
    print(f"  对比图 -> {out_path.name} ({len(picks)} 张，绿框=判对，红框=判错)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", required=True, help="外部数据集目录")
    ap.add_argument("--expect", required=True, help="该外部集应有类别，如 japanese / western")
    ap.add_argument("--limit", type=int, default=400, help="最多取多少张（0=全部）")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    src = Path(args.dir)
    if not src.is_absolute():
        src = ROOT / src

    sys.path.insert(0, str(ROOT / "release"))
    from predict import OnnxClassifier, TorchClassifier, load_classes

    classes = load_classes(model_path)
    clf = OnnxClassifier(model_path) if model_path.suffix == ".onnx" else TorchClassifier(model_path)
    files = collect_images(src, args.limit)
    if not files:
        print(f"没找到图片: {src}")
        return 1
    if args.expect not in classes:
        print(f"注意：预期类别 {args.expect} 不在模型类别 {classes} 中")

    tag = args.tag or f"{src.name}_{model_path.stem}"
    print(f"模型: {model_path.name}  类别: {classes}")
    print(f"外部集: {src}")
    print(f"预期类别: {args.expect}   样本数: {len(files)}")
    print()

    results, preds, confs = [], Counter(), []
    for f in files:
        probs = np.asarray(clf(f), dtype=float)
        top = int(np.argmax(probs))
        pred = classes[top]
        preds[pred] += 1
        confs.append(float(probs[top]))
        results.append((f, args.expect, pred))

    n = len(results)
    hit = sum(1 for _, t, p in results if t == p)
    acc = hit / n

    print(f"预测分布: {dict(preds.most_common())}")
    print(f"平均置信度: {np.mean(confs):.4f}")
    print(f"\n>>> 判为「{args.expect}」的比例 = {hit}/{n} = {acc:.4f}")
    if preds:
        print(f">>> 最主要的误判方向: " +
              "、".join(f"{k} {v} 张 ({v/n:.1%})" for k, v in preds.most_common(3)))

    wrong = [r for r in results if r[1] != r[2]]
    montage(wrong, OUT / f"ext_{tag}_wrong.jpg", label="错判")
    montage(results, OUT / f"ext_{tag}_all.jpg", label="全部")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ext_{tag}.json").write_text(json.dumps({
        "tag": tag, "model": model_path.name, "external_dir": str(src),
        "expect": args.expect, "n": n, "acc": acc,
        "pred_dist": dict(preds), "mean_conf": float(np.mean(confs)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT / f'ext_{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
