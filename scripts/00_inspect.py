"""抽样生成对比图（contact sheet），用于人工核验各来源的实际图像形态。"""
import sys
from pathlib import Path

import cv2
import numpy as np

RAW = Path(__file__).resolve().parent.parent / "datasets" / "comic_style" / "raw"
OUT = Path(__file__).resolve().parent.parent / "datasets" / "comic_style" / "_inspect"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def imread_u(path, flags=cv2.IMREAD_COLOR):
    """OpenCV 的 imread 在 Windows 上无法处理非 ASCII 路径，这里绕开它。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, flags) if buf.size else None


def imwrite_u(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))
    return ok


def list_images(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)


def describe(paths):
    sizes, modes, sats = [], [], []
    for p in paths:
        img = imread_u(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        sizes.append((w, h))
        # 单通道 = 灰度图；三通道算饱和度均值，用来区分彩印 vs 黑白印刷
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 2):
            modes.append("gray")
            sats.append(0.0)
        else:
            hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
            modes.append("color")
            sats.append(float(hsv[:, :, 1].mean()))
    if not sizes:
        return None
    ws = np.array([s[0] for s in sizes])
    hs = np.array([s[1] for s in sizes])
    ar = ws / np.maximum(hs, 1)
    return {
        "n": len(sizes),
        "w": (int(ws.min()), int(np.median(ws)), int(ws.max())),
        "h": (int(hs.min()), int(np.median(hs)), int(hs.max())),
        "ar_med": round(float(np.median(ar)), 3),
        "portrait_frac": round(float((ar < 1).mean()), 3),
        "grey_frac": round(float(np.mean([m == "gray" for m in modes])), 3),
        "sat_mean": round(float(np.mean(sats)), 1),
    }


def montage(paths, out_path, cols=6, rows=4, cell=260):
    picks = paths[: cols * rows]
    if not picks:
        return
    sheet = np.full((rows * cell, cols * cell, 3), 32, np.uint8)
    for i, p in enumerate(picks):
        img = imread_u(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(cell / w, cell / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y0 = r * cell + (cell - nh) // 2
        x0 = c * cell + (cell - nw) // 2
        sheet[y0:y0 + nh, x0:x0 + nw] = img
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_u(out_path, sheet)
    print(f"  montage -> {out_path.name}")


TARGETS = {
    "acn_manga": RAW / "chauaoe999" / "Comic_classification_by_Country" / "manga",
    "acn_manhua": RAW / "chauaoe999" / "Comic_classification_by_Country" / "manhua",
    "acn_manhwa": RAW / "chauaoe999" / "Comic_classification_by_Country" / "manhwa",
    "k_manga": RAW / "kaustubhrastogi17" / "images" / "manga",
    "k_classic": RAW / "kaustubhrastogi17" / "images" / "classic",
    "comics_panels": RAW / "comics_panels",
}


def main():
    for name, d in TARGETS.items():
        if not d.exists():
            print(f"{name}: NOT FOUND ({d})")
            continue
        paths = list_images(d)
        stats = describe(paths)
        print(f"\n[{name}] {d}")
        print(f"  {stats}")
        montage(paths, OUT / f"{name}.jpg")
    print(f"\nOK -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
