"""把整页漫画切成单格(panel)。

日漫页的版式特征是格子之间有白色留白/装订线，因此用递归 XY-cut：
先在整幅范围内找「全白且宽度超过阈值」的水平带，切；再在每块里找垂直带，切；
递归到切不动为止，剩下的就是单格。

带 --debug 会输出画了切分框的对比图，用于人工确认切分是否正确。
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets" / "comic_style"
RAW = DS / "raw"
INSPECT = DS / "_inspect"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

WHITE_LEVEL = 200     # 灰度高于此值算「白」（仅用于裁掉页边）
INK_LEVEL = 128       # 灰度低于此值算「墨」
LINE_INK_FRAC = 0.012  # 一行/一列墨迹占比低于此值 = 留白装订线
FRAME_INK_FRAC = 0.85  # 墨迹占比高于此值且很薄 = 格子边框线
FRAME_MAX_THICK = 9   # 边框线最大厚度(px)
MIN_GAP = 6           # 装订线至少多宽
MIN_PANEL = 60        # 单格最小边长
MAX_DEPTH = 6


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


def find_gutters(cuttable, min_gap):
    """在布尔数组（True=该行/列可切）中找出长度 >= min_gap 的连续段。"""
    runs, start = [], None
    for i, w in enumerate(cuttable):
        if w and start is None:
            start = i
        elif not w and start is not None:
            if i - start >= min_gap:
                runs.append((start, i))
            start = None
    if start is not None and len(cuttable) - start >= min_gap:
        runs.append((start, len(cuttable)))
    return runs


def find_cuts(ink_frac, min_gap):
    """在「逐行/逐列墨迹占比」曲线上找出所有可切位置。

    两类切点:
      A. 留白装订线——墨迹占比极低，宽度不限
      B. 格子边框线——墨迹占比极高且很薄（黑白漫画相邻格子常只靠一条黑线分隔）
    """
    cuts = []
    # A: 低墨迹带
    for s, e in find_gutters(ink_frac < LINE_INK_FRAC, min_gap):
        cuts.append((s, e))
    # B: 细的满墨线
    for s, e in find_gutters(ink_frac > FRAME_INK_FRAC, 1):
        if e - s <= FRAME_MAX_THICK:
            cuts.append((s, e))
    cuts.sort()
    # 合并重叠/相邻的切点带
    merged = []
    for s, e in cuts:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def split_axis(ink, x, y, w, h, axis, depth):
    sub = ink[y:y + h, x:x + w]
    profile = sub.mean(axis=1) if axis == "h" else sub.mean(axis=0)
    cuts = find_cuts(profile, MIN_GAP)
    if not cuts:
        return None

    out, prev = [], 0
    for s, e in cuts:
        if s - prev >= MIN_PANEL:
            if axis == "h":
                out += xy_cut(ink, x, y + prev, w, s - prev, depth + 1)
            else:
                out += xy_cut(ink, x + prev, y, s - prev, h, depth + 1)
        prev = e
    length = h if axis == "h" else w
    if length - prev >= MIN_PANEL:
        if axis == "h":
            out += xy_cut(ink, x, y + prev, w, length - prev, depth + 1)
        else:
            out += xy_cut(ink, x + prev, y, length - prev, h, depth + 1)
    if not out:
        return None
    # 切分没有实际减少面积，说明是无意义的切法
    if len(out) == 1:
        return None
    return out


def xy_cut(ink, x, y, w, h, depth=0):
    """递归切分。ink 为整页的墨迹掩码(bool)，坐标用绝对像素。"""
    if depth >= MAX_DEPTH or w < MIN_PANEL or h < MIN_PANEL:
        return [(x, y, w, h)]

    for axis in ("h", "v"):
        res = split_axis(ink, x, y, w, h, axis, depth)
        if res:
            return res

    return [(x, y, w, h)]


def page_to_panels(img, min_panel=MIN_PANEL):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 页边（扫描黑边/灰边）先裁掉，否则会干扰切分
    white = (gray > WHITE_LEVEL).astype(np.uint8)
    ys, xs = np.where(white > 0)
    if len(xs) == 0:
        return []
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img[y0:y1, x0:x1]
    if min(crop.shape[:2]) < min_panel:
        return []

    cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink = cg < INK_LEVEL
    boxes = xy_cut(ink, 0, 0, ink.shape[1], ink.shape[0])

    panels = []
    for (x, y, w, h) in boxes:
        if w < min_panel or h < min_panel:
            continue
        # 太狭长的多半是切分残留，丢弃
        if max(w, h) / max(min(w, h), 1) > 6:
            continue
        panels.append(crop[y:y + h, x:x + w])
    return panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(RAW / "kaustubhrastogi17" / "images" / "manga"))
    ap.add_argument("--out", default=str(RAW / "manga_panels"))
    ap.add_argument("--debug", type=int, default=0, help="输出前 N 页的切分可视化")
    ap.add_argument("--min-panel", type=int, default=MIN_PANEL)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    INSPECT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)

    debug_items, total, n_pages, n_panels = [], 0, 0, 0
    for p in files:
        img = imread_u(p)
        if img is None:
            continue
        n_pages += 1
        panels = page_to_panels(img, args.min_panel)
        for i, pan in enumerate(panels):
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in p.stem)
            imwrite_u(out / f"{safe}__p{i}.jpg", pan)
            n_panels += 1
        total += len(panels)

        if args.debug and len(debug_items) < args.debug:
            vis = img.copy()
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            wmask = (g > WHITE_LEVEL).astype(np.uint8)
            ys, xs = np.where(wmask > 0)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            for (x, y, w, h) in xy_cut(g[y0:y1, x0:x1] < INK_LEVEL, 0, 0,
                                       x1 - x0, y1 - y0):
                cv2.rectangle(vis, (x0 + x, y0 + y), (x0 + x + w, y0 + y + h),
                              (0, 0, 255), 3)
            debug_items.append((p.name, cv2.resize(vis, (300, 420))))

    print(f"pages={n_pages}  panels={total}  avg={total / max(n_pages, 1):.1f} panels/page")
    print(f"-> {out}")
    if debug_items:
        sheet = np.full((2 * 430, 4 * 310, 3), 32, np.uint8)
        for i, (_, im) in enumerate(debug_items[:8]):
            r, c = divmod(i, 4)
            h, w = im.shape[:2]
            sheet[r * 430 + 5:r * 430 + 5 + h, c * 310 + 5:c * 310 + 5 + w] = im
        imwrite_u(INSPECT / "panelize_debug.jpg", sheet)
        print(f"切分可视化 -> {INSPECT / 'panelize_debug.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
