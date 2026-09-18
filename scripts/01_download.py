"""Step 1: 下载所有原始数据（可复现入口）。

包含两部分:
  A. Kaggle 数据集（需要 Kaggle 账号 token）
  B. COMICS (UMD) 分格图的流式抽取（支持 HTTP Range，只取需要的部分）

前置条件:
  Kaggle token 放在 ~/.kaggle/access_token，或设置环境变量 KAGGLE_API_TOKEN。
  获取方式: https://www.kaggle.com/settings/api  (Generate New Token)

用法:
  python scripts/01_download.py              # 全部
  python scripts/01_download.py --skip-comics
  python scripts/01_download.py --comics-n 3000
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "comic_style" / "raw"

# (Kaggle 数据集 ref, 本地目录名, 用途)
KAGGLE_SETS = [
    # 中/日/韩 漫画封面三分类，本方案里「中国系」唯一的合规图像来源
    ("chauaoe999/comic-classification-by-country", "chauaoe999", "cover: japanese + chinese"),
    # 日漫 vs 欧美经典漫画，欧美封面的主要来源
    ("kaustubhrastogi17/manga-and-classic-comic-arts", "kaustubhrastogi17", "cover: japanese + western"),
]


def check_kaggle():
    if not shutil.which("kaggle"):
        print("未找到 kaggle CLI，请先运行: pip install kaggle")
        return False
    token = Path.home() / ".kaggle" / "access_token"
    if not token.exists() and not (Path.home() / ".kaggle" / "kaggle.json").exists():
        print(f"未找到 Kaggle 凭据。请把 API token 写入 {token}")
        print("或设置环境变量 KAGGLE_API_TOKEN")
        return False
    return True


def download_kaggle():
    for ref, dest_name, purpose in KAGGLE_SETS:
        dest = RAW / dest_name
        dest.mkdir(parents=True, exist_ok=True)
        # 已有内容就跳过，避免重复下载
        existing = [p for p in dest.rglob("*") if p.is_file()]
        if existing:
            print(f"[skip] {ref} 已存在 ({len(existing)} 个文件)")
            continue
        print(f"[get ] {ref}  ->  {dest_name}   ({purpose})")
        cmd = ["kaggle", "datasets", "download", "-d", ref,
               "-p", str(dest), "--unzip"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = (r.stdout or "")[-300:]
        if r.returncode != 0:
            print(f"  失败: {(r.stderr or tail)[-300:]}")
        else:
            print("  完成")


def fetch_comics(n):
    script = Path(__file__).resolve().parent / "01b_fetch_comics.py"
    print(f"[comics] 流式抽取 {n} 张分格图（COMICS 数据集本体 61 GiB，只下需要的部分）")
    subprocess.run([sys.executable, str(script), "-n", str(n)], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-kaggle", action="store_true")
    ap.add_argument("--skip-comics", action="store_true")
    ap.add_argument("--comics-n", type=int, default=6000)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    if not args.skip_kaggle:
        if check_kaggle():
            download_kaggle()
        else:
            print("跳过 Kaggle 下载")

    if not args.skip_comics:
        fetch_comics(args.comics_n)

    print("\n原始数据就绪。接下来依次运行:")
    print("  python scripts/04_panelize.py            # 日漫整页 -> 单格")
    print("  python scripts/02_clean.py               # 清洗 + 去重")
    print("  python scripts/03_build_split.py --clean  # 按作品划分 train/val/test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
