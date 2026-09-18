"""清洗各原始来源的图片，输出到 clean/{subset}/{class}/ 并生成 manifest.csv。

处理内容:
  1. 剔除无法解码、过小、长宽比异常的图
  2. 剔除「纯文字页 / 广告页」——用连通域分析判断前景是否由大量细碎文本块构成
  3. 感知哈希(phash)去重，跨来源的同一张图只保留一份
  4. 统一 EXIF 方向、转 RGB JPEG、长边限制到 640

被剔除的图会输出对比图到 _inspect/rejected_*.jpg，便于人工复核过滤器是否误杀。
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets" / "comic_style"
RAW = DS / "raw"
CLEAN = DS / "clean"
INSPECT = DS / "_inspect"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_SIDE = 640
MIN_SIDE = 150
PHASH_MAX_DIST = 6

ACN = RAW / "chauaoe999" / "Comic_classification_by_Country"

# (subset, class, 源目录, 来源标签)
SPECS = [
    ("cover", "japanese", ACN / "manga", "acn_manga"),
    ("cover", "chinese", ACN / "manhua", "acn_manhua"),
    ("cover", "western", RAW / "kaustubhrastogi17" / "images" / "classic", "k_classic"),
    ("panel", "japanese", RAW / "manga_panels", "jp_panel"),
    ("panel", "western", RAW / "comics_panels", "us_panel"),
]


def imread_u(path, flags=cv2.IMREAD_COLOR):
    """Windows 下 OpenCV 的 imread 读不了非 ASCII 路径，用 fromfile+imdecode 替代。"""
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


def phash_bits(gray, hash_size=8, factor=4):
    size = hash_size * factor
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small.astype(np.float32))[:hash_size, :hash_size]
    med = np.median(dct[1:, 1:])
    return int("".join("1" if v > med else "0" for v in dct.flatten()), 2)


def text_page_score(gray):
    """估计「这是一页文字/广告而非漫画画面」的程度。

    思路: 二值化后做连通域分析。文字由大量小而密的笔画块组成，
    前景像素大多落在小连通域里；漫画画面有大量成片填充区域。
    返回 (小连通域前景占比, 每百万像素的小连通域数)。
    """
    h, w = gray.shape
    # 自适应阈值应对扫描件的明暗不均
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return 1.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    total = areas.sum()
    if total <= 0:
        return 1.0, 0.0
    small_cut = 0.0005 * h * w  # 小于图像面积 0.05% 视为「小碎块」
    small = areas < small_cut
    frac_small = float(areas[small].sum() / total)
    density = float(small.sum() / (h * w / 1e6))
    return frac_small, density


def group_key(tag, path):
    """推导「作品」分组键，供划分时避免同一作品跨 train/val。"""
    if tag == "us_panel":
        return path.parent.name  # 按 COMICS 的书目 id 分组
    stem = path.stem
    if tag in ("acn_manga", "acn_manhua"):
        # ACN 文件名形如 "<越南语标题>-<话号>"，去掉尾部数字得到作品名
        parts = stem.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
    return stem


def montage(items, out_path, cols=6, rows=4, cell=240):
    if not items:
        return
    sheet = np.full((rows * cell, cols * cell, 3), 32, np.uint8)
    for i, (_, img) in enumerate(items[: cols * rows]):
        h, w = img.shape[:2]
        scale = min(cell / w, cell / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        thumb = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y0 = r * cell + (cell - nh) // 2
        x0 = c * cell + (cell - nw) // 2
        sheet[y0:y0 + nh, x0:x0 + nw] = thumb
    imwrite_u(out_path, sheet)


def load_for_clean(path):
    """用 PIL 读入以正确处理 EXIF 方向与非 ASCII 路径，返回 BGR ndarray。"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im).convert("RGB")
        else:
            im = im.convert("RGB")
        arr = np.asarray(im)[:, :, ::-1]  # RGB -> BGR
    return np.ascontiguousarray(arr)


def resize_max(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def process(specs):
    CLEAN.mkdir(parents=True, exist_ok=True)
    INSPECT.mkdir(parents=True, exist_ok=True)
    rows = []
    rejected = {}
    seen = []  # (phash, subset, class) 用于跨来源去重
    stats = {}

    for subset, cls, src, tag in specs:
        if not src.exists():
            print(f"[skip] {tag}: {src} 不存在")
            continue
        out_dir = CLEAN / subset / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
        keep = drop_small = drop_ratio = drop_text = drop_dup = drop_bad = 0

        for p in files:
            try:
                img = load_for_clean(p)
            except Exception:
                drop_bad += 1
                continue
            if img is None or img.size == 0:
                drop_bad += 1
                continue
            h, w = img.shape[:2]
            if min(h, w) < MIN_SIDE:
                drop_small += 1
                continue
            ar = w / h
            if ar < 0.2 or ar > 5.0:
                drop_ratio += 1
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frac_small, density = text_page_score(gray)
            if frac_small > 0.62 and density > 90:
                drop_text += 1
                rejected.setdefault(f"{subset}_{cls}_text", []).append(
                    (p.name, cv2.resize(img, (240, 240)))
                )
                continue

            ph = phash_bits(gray)
            dup_of = None
            for prev, psub, pcls in seen:
                if (ph ^ prev).bit_count() <= PHASH_MAX_DIST:
                    dup_of = f"{psub}/{pcls}"
                    break
            if dup_of is not None:
                drop_dup += 1
                continue
            seen.append((ph, subset, cls))

            img = resize_max(img)
            gk = group_key(tag, p)
            # 用完整原始文件名：同名不同扩展名的文件可能是不同图片，只用 stem 会互相覆盖
            out_name = f"{tag}__{gk}__{p.name}"
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in out_name)
            # Windows 路径上限 260 字符，长名截断并用短哈希保唯一
            if len(safe) > 110:
                digest = hashlib.md5(out_name.encode("utf-8")).hexdigest()[:10]
                safe = f"{safe[:100]}__{digest}"
            safe += ".jpg"
            out_path = out_dir / safe
            imwrite_u(out_path, img)
            keep += 1
            rows.append({
                "subset": subset, "class": cls, "source": tag,
                "group": gk, "file": f"{subset}/{cls}/{safe}",
                "orig": str(p.relative_to(RAW)),
                "w": img.shape[1], "h": img.shape[0],
                "text_frac_small": round(frac_small, 4),
                "text_density": round(density, 1),
            })

        stats[(subset, cls, tag)] = (len(files), keep, drop_bad, drop_small,
                                     drop_ratio, drop_text, drop_dup)

    # manifest
    man = CLEAN / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["subset"])
        wtr.writeheader()
        wtr.writerows(rows)

    print(f"\n{'subset/class':<18}{'source':<12}{'total':>7}{'keep':>7}"
          f"{'bad':>6}{'small':>7}{'ratio':>7}{'text':>6}{'dup':>6}")
    for (subset, cls, tag), (tot, k, b, s, r, t, d) in stats.items():
        print(f"{subset + '/' + cls:<18}{tag:<12}{tot:>7}{k:>7}{b:>6}{s:>7}{r:>7}{t:>6}{d:>6}")

    for key, items in rejected.items():
        montage(items, INSPECT / f"rejected_{key}.jpg")
        print(f"被判定为文字页/广告页的样本 -> rejected_{key}.jpg ({len(items)} 张)")
    print(f"\nmanifest -> {man}  ({len(rows)} 条)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="只处理指定来源标签")
    args = ap.parse_args()
    specs = [s for s in SPECS if not args.only or s[3] in args.only]
    return process(specs)


if __name__ == "__main__":
    sys.exit(main())
