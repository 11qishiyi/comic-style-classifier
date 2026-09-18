"""分类模型的评估与消融实验。

除了常规的混淆矩阵，这里实现了三个针对本任务特有风险的对照实验：

  --gray          把测试图转成灰度再推理。
                  三类漫画里日漫是黑白印刷、国漫和欧美漫是彩色，模型可能只是在判「有没有颜色」。
                  准确率跌幅直接量化了它对色彩线索的依赖。

  --downscale N   把测试图短边压到 N 像素再推理。
                  cover 子集里中日封面是 190x247 缩略图、欧美封面是 474x640，
                  面积差 5.8 倍，模型可能靠清晰度作弊。降采样后如果性能崩塌，说明确实如此。

  --classes A B   只保留指定类别（用于跨域实验，例如把日漫当正类、欧美漫当负类）。

用法:
  python scripts/06_eval.py --model runs/classify/cover/weights/best.pt --data datasets/comic_style/cover
  python scripts/06_eval.py --model runs/classify/cover/weights/best.pt --data datasets/comic_style/cover --gray
  python scripts/06_eval.py --model runs/classify/panel/weights/best.pt --data datasets/comic_style/cover --classes japanese western --tag cross_cover2panel
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets" / "comic_style"
OUT = DS / "_eval"
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


def prepare(images, ops, tag):
    """按指定的图像变换序列生成一份测试图副本，返回 [(路径, 真实类别)]。"""
    if not ops:
        return images
    out_dir = OUT / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = []
    for path, cls in images:
        img = imread_u(path)
        if img is None:
            continue
        for op in ops:
            if op == "gray":
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)  # 保持三通道，避免影响预处理
            elif op.startswith("down"):
                short = int(op[4:])
                h, w = img.shape[:2]
                scale = short / min(h, w)
                img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                                 interpolation=cv2.INTER_AREA)
        dst = out_dir / f"{cls}__{path.name}"
        imwrite_u(dst, img)
        prepared.append((dst, cls))
    return prepared


def confusion(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), np.int64)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def report(m, classes):
    total = m.sum()
    acc = np.trace(m) / total if total else 0.0
    print(f"\n{'':<12}" + "".join(f"{c:>12}" for c in classes) + f"{'recall':>10}")
    for i, c in enumerate(classes):
        row = "".join(f"{v:>12}" for v in m[i])
        rec = m[i, i] / m[i].sum() if m[i].sum() else 0.0
        print(f"{c:<12}{row}{rec:>10.3f}")
    prec_row, f1s = [], []
    for j in range(len(classes)):
        col = m[:, j].sum()
        p = m[j, j] / col if col else 0.0
        r = m[j, j] / m[j].sum() if m[j].sum() else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        prec_row.append(p)
        f1s.append(f1)
    print(f"{'precision':<12}" + "".join(f"{v:>12.3f}" for v in prec_row))
    print(f"{'f1':<12}" + "".join(f"{v:>12.3f}" for v in f1s))
    print(f"\n总体准确率 = {acc:.4f}   (n={total})")
    return acc


def draw_matrix(m, classes, path, title):
    cell, pad = 90, 130
    h = pad + cell * len(classes) + 40
    w = pad + cell * len(classes) + 60
    img = np.full((h, w, 3), 255, np.uint8)
    mx = max(m.max(), 1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = m[i, j]
            # 越深表示数量越多
            shade = int(255 - 200 * (v / mx))
            cv2.rectangle(img, (pad + j * cell, pad + i * cell),
                          (pad + (j + 1) * cell - 2, pad + (i + 1) * cell - 2),
                          (shade, shade, shade), -1)
            cv2.putText(img, str(v), (pad + j * cell + 30, pad + i * cell + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for i, c in enumerate(classes):
        cv2.putText(img, c, (10, pad + i * cell + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, c, (pad + i * cell + 5, pad - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "true ->", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, title, (pad, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    img[pad - 5:pad - 3, pad:pad + cell * len(classes)] = 0
    img[pad:pad + cell * len(classes), pad - 5:pad - 3] = 0
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_u(path, img)


def save_error_sheet(images, y_true, y_pred, pairs, path, cell=200, cols=8):
    """把指定混淆对（真实->预测）的错判样本拼成一张图。"""
    items = []
    for (img_path, _), t, p in zip(images, y_true, y_pred):
        if (t, p) in pairs:
            img = imread_u(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            s = min(cell / w, cell / h)
            img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                             interpolation=cv2.INTER_AREA)
            items.append(img)
    if not items:
        print(f"没有 {pairs} 的错判样本")
        return
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 32, np.uint8)
    for i, img in enumerate(items):
        r, c = divmod(i, cols)
        h, w = img.shape[:2]
        y0 = r * cell + (cell - h) // 2
        x0 = c * cell + (cell - w) // 2
        sheet[y0:y0 + h, x0:x0 + w] = img
    path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_u(path, sheet)
    print(f"错判样本对比图 -> {path.name} ({len(items)} 张)")


def collect(data_dir, keep_classes):
    items = []
    for split in ("test", "val"):
        d = data_dir / split
        if not d.exists():
            continue
        for cls_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            if keep_classes and cls_dir.name not in keep_classes:
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    items.append((f, cls_dir.name))
        if items:
            break  # 优先用 test，没有则退回 val
    return items


def load_groups():
    """从 manifest 读「文件名 -> 作品分组」，比从文件名反解可靠。"""
    man = DS / "clean" / "manifest.csv"
    out = {}
    if man.exists():
        import csv as _csv
        with man.open(encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                out[Path(r["file"]).name] = r["group"]
    return out


def report_per_group(images, y_true, y_pred, groups):
    """按作品分组统计准确率。

    按作品划分时，若某个组在测试集里占比过高，整体准确率会退化成「这一个作品的得分」。
    这个分解能暴露这种情况，也能看出模型是否只在部分作品上work。
    """
    per = defaultdict(lambda: defaultdict(int))
    cls_of = {}
    for (path, _), t, p in zip(images, y_true, y_pred):
        g = groups.get(path.name, path.name)
        per[g][(t, p)] += 1
        cls_of.setdefault(g, set()).add(t)

    print(f"\n按作品分组的准确率（共 {len(per)} 组，只列 >=3 张的组）:")
    print(f"{'group':<28}{'class':<22}{'n':>5}{'acc':>8}   错误明细")
    rows = sorted(per.items(), key=lambda kv: -sum(kv[1].values()))
    for g, counts in rows:
        n = sum(counts.values())
        if n < 3:
            continue
        ok = sum(v for (t, p), v in counts.items() if t == p)
        errs = " ".join(f"{t}->{p}:{v}" for (t, p), v in sorted(counts.items()) if t != p)
        print(f"{g[:27]:<28}{'/'.join(sorted(cls_of[g])):<22}{n:>5}{ok / n:>8.2f}   {errs}")
    # 最大组占测试集的比例，用来判断整体指标是否被单组绑架
    if rows:
        top_n = sum(rows[0][1].values())
        total = sum(sum(c.values()) for _, c in rows)
        print(f"最大组占比: {top_n}/{total} = {top_n / total:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--classes", nargs="*", default=None,
                    help="只评估这些类别（跨域实验用）")
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--downscale", type=int, default=0, help="短边压到此像素数")
    ap.add_argument("--tag", default=None, help="输出文件名后缀")
    ap.add_argument("--per-group", action="store_true",
                    help="按作品分组分解准确率，检查是否被单个作品主导")
    ap.add_argument("--save-errors", action="store_true",
                    help="把判错的样本拼成对比图，便于人工判断是模型错还是标注噪声")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    tag = args.tag or f"{data_dir.name}_{model_path.parent.parent.name}"
    images = collect(data_dir, set(args.classes) if args.classes else None)
    if not images:
        print(f"没找到测试图: {data_dir}")
        return 1

    ops = []
    tag = args.tag or f"{data_dir.name}_{model_path.parent.parent.name}"
    if args.gray:
        ops.append("gray")
        tag += "_gray"
    if args.downscale:
        ops.append(f"down{args.downscale}")
        tag += f"_ds{args.downscale}"

    classes = sorted({c for _, c in images})
    print(f"模型: {model_path.name}")
    print(f"数据: {data_dir}  类别: {classes}  样本: {len(images)}")
    print(f"变换: {'+'.join(ops) if ops else 'none'}")

    images = prepare(images, ops, tag)
    model = YOLO(str(model_path))
    names = model.names  # {idx: 类别名}

    paths = [str(p) for p, _ in images]
    y_true = [c for _, c in images]
    y_pred = []
    # 模型输出类别顺序取自训练时的数据集，这里按名字映射回标签
    for r in model.predict(paths, imgsz=224, verbose=False, stream=True):
        y_pred.append(names[int(r.probs.top1)])

    # 跨域评估时模型可能预测出真实标签里没有的类别，矩阵要按两者的并集建
    classes = sorted(set(y_true) | set(y_pred))
    m = confusion(y_true, y_pred, classes)
    acc = report(m, classes)
    draw_matrix(m, classes, OUT / f"cm_{tag}.jpg", tag)
    (OUT / f"metrics_{tag}.json").write_text(
        json.dumps({"tag": tag, "acc": acc, "classes": classes,
                    "matrix": m.tolist(), "n": len(images)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n混淆矩阵图 -> {OUT / f'cm_{tag}.jpg'}")
    if args.per_group:
        report_per_group(images, y_true, y_pred, load_groups())
    if args.save_errors:
        err_pairs = {(t, p) for t, p in zip(y_true, y_pred) if t != p}
        for i, pr in enumerate(sorted(err_pairs)):
            save_error_sheet(images, y_true, y_pred, {pr},
                             OUT / f"errors_{tag}_{pr[0]}2{pr[1]}.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
