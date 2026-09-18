"""生成一个灰度版本的数据集，用于「去掉颜色后还能不能分辨画风」的严格实验。

为什么不能只用 06_eval.py --gray 代替：
  用彩色图训练出来的模型去推理灰度图，是分布外(OOD)输入，掉分可能只是因为
  模型没见过灰度图，而不一定说明它原本只靠颜色。
  直接用灰度图训练一个模型才能公平地回答：抛开颜色，三类画风是否可分。

用法:
  python scripts/07_make_gray.py --src datasets/comic_style/cover --dst datasets/comic_style/cover_gray
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def imread_u(path):
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def imwrite_u(path, img, quality=92):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(str(path))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_absolute():
        src = ROOT / src
    if not dst.is_absolute():
        dst = ROOT / dst

    # 必须先清空：源划分变了以后，旧文件名和新文件名不一致，
    # 直接写入会把两套划分混在一个目录里，且数量对不上
    if dst.exists():
        shutil.rmtree(dst)
        print(f"已清空旧目录 {dst}")

    n = 0
    splits = [d for d in src.iterdir() if d.is_dir()]
    for split in splits:
        for cls_dir in sorted(p for p in split.iterdir() if p.is_dir()):
            out_dir = dst / split.name / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() not in IMG_EXT:
                    continue
                img = imread_u(f)
                if img is None:
                    continue
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 保持三通道，避免 Ultralytics 预处理的通道数差异引入额外变量
                out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                imwrite_u(out_dir / f.name, out)
                n += 1

    print(f"灰度数据集已生成: {n} 张 -> {dst}")
    for split in sorted(p.name for p in dst.iterdir() if p.is_dir()):
        line = []
        for cls_dir in sorted(p for p in (dst / split).iterdir() if p.is_dir()):
            line.append(f"{cls_dir.name}={len(list(cls_dir.iterdir()))}")
        print(f"  {split}: {'  '.join(line)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
