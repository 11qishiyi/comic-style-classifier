"""训练 YOLO 分类模型。

用法:
  python scripts/05_train_cls.py --data datasets/comic_style/cover --name cover
  python scripts/05_train_cls.py --data datasets/comic_style/panel --name panel
"""
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "yolo12n-cls.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="数据集目录（Ultralytics 分类格式）")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--name", required=True, help="本次运行名")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=8, help="早停耐心值")
    ap.add_argument("--workers", type=int, default=4, help="dataloader 进程数；CPU 训练时可调小以降低争抢")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    if not data.is_absolute():
        data = ROOT / data
    if not data.exists():
        print(f"数据集不存在: {data}")
        return 1

    model = YOLO(args.model)
    print(f"模型: {args.model} (task={model.task})")
    print(f"数据: {data}")
    print(f"类别: {sorted(p.name for p in data.iterdir() if p.is_dir())}")

    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        project=str(ROOT / "runs" / "classify"),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    metrics = model.val()
    print(f"\n[{args.name}] top1={metrics.top1:.4f}  top5={metrics.top5:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
