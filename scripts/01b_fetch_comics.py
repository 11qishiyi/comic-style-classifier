"""流式抽取 COMICS (UMD) 数据集的分格图。

数据集本体 61.2 GiB，但服务端支持 Range 请求，所以这里用流式 tar 顺序读取，
只解出目标数量的图就断开连接，不必下载整包。

来源: https://obj.umiacs.umd.edu/comics/  (美国黄金时代漫画，1938-1954，原始作品已进入公有领域)
"""
import argparse
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

URL = "https://obj.umiacs.umd.edu/comics/raw_panel_images.tar.gz"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "datasets" / "comic_style" / "raw" / "comics_panels"
VALID_EXT = {".jpg", ".jpeg", ".png"}


def _stream_once(target: int, timeout: int):
    """跑一遍流式解包。返回本次解出的新图数量；连接中断会抛异常。"""
    req = urllib.request.Request(URL, headers={"User-Agent": "comic-style-dataset/0.1"})
    resp = urllib.request.urlopen(req, timeout=timeout)

    saved = 0
    books = set()
    t0 = time.time()
    last_log = t0

    with tarfile.open(fileobj=resp, mode="r|gz") as tf:
        for member in tf:
            if saved >= target:
                break
            if not member.isfile():
                continue
            name = Path(member.name)
            if name.suffix.lower() not in VALID_EXT:
                continue

            # member.name 形如 raw_panel_images/<book_id>/<page>_<panel>.jpg
            parts = name.parts
            book_id = parts[-2] if len(parts) >= 2 else "unknown"
            out_dir = DEST / book_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / name.name
            if not out_path.exists():
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                out_path.write_bytes(fh.read())
            saved += 1
            books.add(book_id)

            now = time.time()
            if now - last_log >= 15:
                rate = saved / max(now - t0, 1e-6)
                eta = (target - saved) / max(rate, 1e-6)
                print(
                    f"  {saved}/{target} panels | {len(books)} books | "
                    f"{rate:.1f}/s | eta {eta / 60:.1f} min",
                    flush=True,
                )
                last_log = now

    resp.close()
    return saved, books


def count_existing():
    files = list(DEST.rglob("*.jpg")) if DEST.exists() else []
    books = {p.parent.name for p in files}
    return len(files), len(books)


def fetch(target: int, timeout: int = 120, max_retries: int = 200):
    """流式抽取，带断线重试。

    tar.gz 无法从中间字节位置恢复解压，所以重试只能从流的开头重新读。
    但已保存的图会跳过，因此进度不会丢失，重试代价只是重下前缀部分的字节。
    """
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"streaming {URL}\n  target = {target} panels -> {DEST}")

    for attempt in range(1, max_retries + 1):
        have, n_books = count_existing()
        if have >= target:
            print(f"\ndone: {have} panels from {n_books} books")
            print(f"-> {DEST}")
            return True

        print(f"\n[attempt {attempt}/{max_retries}] 已有 {have} 张 / {n_books} 本书，继续读流...",
              flush=True)
        try:
            saved, books = _stream_once(target, timeout)
            have, n_books = count_existing()
            print(f"  本轮读到 {saved} 个成员，共 {len(books)} 本书")
            if have >= target:
                print(f"\ndone: {have} panels from {n_books} books")
                print(f"-> {DEST}")
                return True
        except Exception as e:
            print(f"  中断: {type(e).__name__}: {e}", flush=True)
            wait = min(30 * attempt, 180)
            print(f"  {wait}s 后重试", flush=True)
            time.sleep(wait)

    have, n_books = count_existing()
    print(f"\n重试次数用尽，当前 {have} 张 / {n_books} 本书")
    return have >= target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=1500, help="目标分格图数量")
    args = ap.parse_args()
    fetch(args.num)
    return 0


if __name__ == "__main__":
    sys.exit(main())
