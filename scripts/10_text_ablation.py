"""文字依赖性的量化与消融实验。

用户提的质疑：模型会不会是在"读文字"（日文假名 vs 英文字母）而不是看画风？
本脚本用两步回答：

  1. 量化：估计训练/测试集里文字区域的面积占比。
     漫画文字有两个稳定特征——(a) 位于白色气泡/留白里，(b) 由大量大小相仿的小连通域组成。
     用连通域统计可以在不依赖 OCR 的前提下把文字区域圈出来。

  2. 消融：把圈出来的文字区域涂白，重新推理，比较准确率变化。
     - 若准确率基本不变 → 模型不依赖文字
     - 若大幅下降     → 模型确实在靠文字作弊

用法:
  python scripts/10_text_ablation.py --dir datasets/comic_style/panel/test --debug 12
  python scripts/10_text_ablation.py --model release/models/comic-style-panel-yolo11n-cls.onnx \
      --dir datasets/comic_style/panel/test --erase
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "comic_style" / "_text_ablation"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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


def text_mask(img):
    """估计文字区域。

    漫画文字的特征（按重要性排序）：
      1. **位于大片白色区域之内**——对话气泡和旁白框都是白底黑字。
         这一条最关键：它能排除脸部睫毛、头发线条、网点贴网等细密线条，
         那些虽然也是"小连通域"，但底色是灰的或有网点，不是纯白大块。
      2. 由大量尺寸相仿的小连通域成行排列构成。

    第一版实现只用第 2 条，结果把眼睫毛和贴网圈成了文字，却漏掉真正的对白框。
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    area_img = h * w

    # 候选笔画：局部二值化后的小连通域
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return np.zeros((h, w), np.uint8)

    cand = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, bw_, bh_, area = stats[i]
        if area < 4 or area > area_img * 0.002:
            continue
        ar = bw_ / max(bh_, 1)
        fill = area / max(bw_ * bh_, 1)
        if 0.1 < ar < 8 and fill > 0.2 and max(bw_, bh_) < 0.06 * max(h, w):
            cand[y:y + bh_, x:x + bw_] = 255
    if not cand.any():
        return cand

    # 把笔画连成块，再按"是否处在白色区域内"筛选
    k = max(3, int(0.006 * min(h, w)) | 1)
    dil = cv2.dilate(cand, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)

    # 先算出整张图的"白"掩码，用于判断块的周围是否有大片留白
    white = (gray > 200).astype(np.uint8)
    out = np.zeros((h, w), np.uint8)
    for i in range(1, n2):
        x, y, bw_, bh_, area = st2[i]
        if area < area_img * 0.0015:
            continue
        # 块的外接矩形往外扩一圈，看这一圈里白像素的比例
        pad = max(3, int(0.02 * min(h, w)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw_ + pad), min(h, y + bh_ + pad)
        ring = white[y0:y1, x0:x1].copy()
        inner = np.zeros_like(ring)
        inner[y - y0:y - y0 + bh_, x - x0:x - x0 + bw_] = 1
        ring = ring[inner == 0]                    # 只看外圈
        if ring.size == 0 or ring.mean() < 0.55:   # 外圈不够白 → 不是气泡/文字框
            continue
        out[lab2 == i] = 255
    return out


def coverage(files, limit=150, seed=0):
    files = list(files)
    random.Random(seed).shuffle(files)
    files = files[:limit]
    covs = []
    for f in files:
        img = imread_u(f)
        if img is None:
            continue
        m = text_mask(img)
        covs.append(float((m > 0).mean()))
    return np.mean(covs) if covs else 0.0, len(covs)


def run_coverage(sets):
    print(f"{'集合':<34}{'样本':>6}{'估计文字面积占比':>18}")
    print("-" * 62)
    res = {}
    for name, files in sets.items():
        if not files:
            print(f"{name:<34}{'(无数据)':>6}")
            continue
        c, n = coverage(files)
        res[name] = {"coverage": c, "n": n}
        print(f"{name:<34}{n:>6}{c * 100:>17.1f}%")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "release/models/comic-style-panel-yolo11n-cls.onnx"))
    ap.add_argument("--dir", help="要做消融的数据集目录（含类别子目录）")
    ap.add_argument("--erase", action="store_true", help="执行涂白消融并比较准确率")
    ap.add_argument("--coverage-only", action="store_true", help="只统计文字面积占比")
    ap.add_argument("--debug", type=int, default=0, help="输出前 N 张的文字掩码可视化")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.coverage_only or not args.dir:
        DS = ROOT / "datasets" / "comic_style"
        ext = ROOT / "datasets" / "external"
        sets = {}
        for split in ("train",):
            for cls in ("japanese", "chinese", "western"):
                sets[f"cover/{split}/{cls}"] = list((DS / "cover" / split / cls).glob("*.jpg"))
        for cls in ("japanese", "western"):
            sets[f"panel/{split}/{cls}"] = list((DS / "panel" / split / cls).glob("*.jpg"))
        sets["外部 钢炼（已去文字）"] = list((ext / "whitened-manga-panels").glob("*.png"))
        sets["外部 Manga109 封面"] = list((ext / "manga109").rglob("*.png"))
        sets["外部 Garfield"] = list((ext / "garfield/extracted").rglob("*.jpeg"))
        res = run_coverage(sets)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "coverage.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> {OUT / 'coverage.json'}")
        if args.debug:
            print("（--debug 需配合 --dir 使用）")
        return 0

    src = Path(args.dir)
    if not src.is_absolute():
        src = ROOT / src
    classes = sorted(p for p in src.iterdir() if p.is_dir())
    items = [(f, c.name) for c in classes for f in sorted(c.iterdir())
             if f.suffix.lower() in IMG_EXT]
    if not items:
        print(f"没找到图片: {src}")
        return 1
    if args.limit:
        items = items[: args.limit]

    if args.debug:
        rows, cols, cell = 3, 4, 270
        dbg = []
        for f, _ in items[: args.debug]:
            img = imread_u(f)
            if img is None:
                continue
            m = text_mask(img)
            vis = img.copy()
            vis[m > 0] = (0, 0, 255)
            vis = cv2.addWeighted(img, 0.5, vis, 0.5, 0)
            h, w = vis.shape[:2]
            s = min((cell - 12) / w, (cell - 12) / h)   # 必须装得进 cell，否则拼接越界
            dbg.append(cv2.resize(vis, (max(1, int(w * s)), max(1, int(h * s))),
                                  interpolation=cv2.INTER_AREA))
        if dbg:
            sheet = np.full((rows * cell, cols * cell, 3), 32, np.uint8)
            for i, im in enumerate(dbg[: rows * cols]):
                r, c = divmod(i, cols)
                y0, x0 = r * cell + 6, c * cell + 6
                sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            OUT.mkdir(parents=True, exist_ok=True)
            imwrite_u(OUT / "text_mask_debug.jpg", sheet)
            print(f"文字掩码可视化（红色=判定为文字）-> {OUT / 'text_mask_debug.jpg'}")

    if not args.erase:
        return 0

    sys.path.insert(0, str(ROOT / "release"))
    from predict import OnnxClassifier, TorchClassifier, load_classes

    mp = Path(args.model)
    if not mp.is_absolute():
        mp = ROOT / mp
    clf = OnnxClassifier(mp) if mp.suffix == ".onnx" else TorchClassifier(mp)
    cls_names = load_classes(mp)

    print(f"\n模型: {mp.name}  类别: {cls_names}")
    print(f"数据: {src}   样本: {len(items)}")

    per = defaultdict(lambda: {"n": 0, "orig": 0, "erased": 0, "cov": []})
    for i, (f, truth) in enumerate(items):
        img = imread_u(f)
        if img is None:
            continue
        m = text_mask(img)
        cov = float((m > 0).mean())
        erased = img.copy()
        erased[m > 0] = 255          # 涂白
        p_orig = cls_names[int(np.argmax(clf(f)))]
        tmp = OUT / "_tmp.jpg"
        OUT.mkdir(parents=True, exist_ok=True)
        imwrite_u(tmp, erased)
        p_er = cls_names[int(np.argmax(clf(tmp)))]
        d = per[truth]
        d["n"] += 1
        d["orig"] += (p_orig == truth)
        d["erased"] += (p_er == truth)
        d["cov"].append(cov)
        if i % 100 == 0:
            print(f"  ...{i}/{len(items)}", flush=True)
    if (OUT / "_tmp.jpg").exists():
        (OUT / "_tmp.jpg").unlink()

    print(f"\n{'类别':<12}{'n':>5}{'文字占比':>10}{'原图准确率':>12}{'涂白后':>10}{'变化':>9}")
    print("-" * 60)
    tot = {"n": 0, "orig": 0, "erased": 0}
    for c in sorted(per):
        d = per[c]
        tot["n"] += d["n"]; tot["orig"] += d["orig"]; tot["erased"] += d["erased"]
        print(f"{c:<12}{d['n']:>5}{np.mean(d['cov']) * 100:>9.1f}%"
              f"{d['orig'] / d['n']:>12.4f}{d['erased'] / d['n']:>10.4f}"
              f"{(d['erased'] - d['orig']) / d['n'] * 100:>+8.1f}分")
    print("-" * 60)
    print(f"{'合计':<12}{tot['n']:>5}{'':>10}{tot['orig'] / tot['n']:>12.4f}"
          f"{tot['erased'] / tot['n']:>10.4f}"
          f"{(tot['erased'] - tot['orig']) / tot['n'] * 100:>+8.1f}分")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"erase_{src.parent.name}_{src.name}.json").write_text(json.dumps({
        "dir": str(src), "model": mp.name, "total": tot,
        "per_class": {k: {"n": v["n"], "orig": v["orig"], "erased": v["erased"],
                          "text_coverage": float(np.mean(v["cov"]))}
                      for k, v in per.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
