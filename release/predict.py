"""漫画画风分类推理脚本。

支持两种权重，输出完全一致（已验证逐位相同）：
  * ONNX（推荐发布用）：只需 onnxruntime + pillow + numpy，不依赖 ultralytics/torch
  * .pt            ：需要装 ultralytics

用法:
  python predict.py release/models/comic-style-cover-yolo11n-cls.onnx 图片或目录 [更多...]
  python predict.py release/models/comic-style-panel-yolo11n-cls.onnx panels/

  # 只看 Top-1
  python predict.py release/models/comic-style-cover-yolo11n-cls.onnx a.jpg --topk 1

  # 输出 JSON，便于脚本调用
  python predict.py release/models/comic-style-cover-yolo11n-cls.onnx a.jpg --json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SIZE = 224


def preprocess(path, size=SIZE):
    """必须与 ultralytics.data.augment.classify_transforms(size=224) 完全一致，
    否则精度会掉。顺序是：

        T.Resize(224, BILINEAR)   # 短边缩放到 224，保持长宽比
        T.CenterCrop(224)         # 中心裁剪 224x224
        T.ToTensor()              # 除以 255
        T.Normalize((0,0,0),(1,1,1))   # 等于不做归一化

    注意两个常见坑：
      1. 不是 ImageNet 那套「短边缩到 256 再裁 224」——ultralytics 的 crop_fraction 已废弃。
      2. 不要减 ImageNet 均值、不要除标准差；ultralytics 分类用的是 0~1 原始像素。
    实测这两个坑会让同一张图给出不同的置信度分布。
    """
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = size / min(w, h)
    im = im.resize((max(size, round(w * scale)), max(size, round(h * scale))),
                   Image.BILINEAR)          # 短边 >= size，长边按比例
    w, h = im.size
    # 必须用 round 而不是整除：torchvision 的 CenterCrop 是
    #   int(round((w - size) / 2.0))
    # 当差值为奇数时（例如 291-224=67），round(33.5)=34 而 67//2=33，
    # 裁切位置会差 1 个像素，在小图上足以让置信度偏移 0.3。
    left, top = int(round((w - size) / 2.0)), int(round((h - size) / 2.0))
    im = im.crop((left, top, left + size, top + size))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None]   # NCHW


def to_probs(x):
    """把模型输出统一成概率。

    Ultralytics 导出的 ONNX 分类模型**已经把 softmax 包含在计算图里**，
    输出就是概率；再套一层 softmax 会把分布压平（例如 [1,0,0] 会变成 [0.576,0.212,0.212]），
    看起来还能用但置信度全错。这里按「是否已是合法概率分布」自动判别，
    这样无论上游导出带不带 softmax 都能正确处理。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.min() >= 0.0 and abs(x.sum() - 1.0) < 1e-3:
        return x
    e = np.exp(x - x.max())
    return e / e.sum()


class OnnxClassifier:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def __call__(self, path):
        return to_probs(self.sess.run(None, {self.input: preprocess(path)})[0][0])


class TorchClassifier:
    def __init__(self, path):
        from ultralytics import YOLO
        self.model = YOLO(str(path))
        self.classes = [self.model.names[i] for i in sorted(self.model.names)]

    def __call__(self, path):
        return np.asarray(
            self.model.predict(str(path), imgsz=SIZE, verbose=False)[0].probs.data.numpy(),
            dtype=np.float64,
        )


def load_classes(model_path):
    """类别名从同目录的 classes.json 读；.pt 可直接从权重里读。

    先精确匹配文件名主干，再退化到子串匹配——这样重命名权重文件
    （例如加版本号后缀）不会让脚本失效。
    """
    base = Path(model_path).stem
    cf = Path(model_path).parent / "classes.json"
    if cf.exists():
        d = json.loads(cf.read_text(encoding="utf-8"))
        if base in d:
            return d[base]
        # 子串匹配：取最长的匹配键，避免 "cover" 抢先匹配到 "comic-style-cover-gray-..."
        cands = [k for k in d if k in base or base in k]
        if cands:
            return d[max(cands, key=len)]
    if str(model_path).endswith(".pt"):
        from ultralytics import YOLO
        m = YOLO(str(model_path))
        return [m.names[i] for i in sorted(m.names)]
    raise SystemExit(f"找不到 {cf} 里与 {base} 对应的类别定义")


def gather(inputs):
    files = []
    for it in inputs:
        p = Path(it)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        elif p.is_file():
            files.append(p)
        else:
            print(f"跳过（不存在）: {it}", file=sys.stderr)
    return files


def main():
    ap = argparse.ArgumentParser(description="漫画画风分类推理")
    ap.add_argument("model", help=".onnx 或 .pt 权重路径")
    ap.add_argument("inputs", nargs="+", help="图片或目录")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本调用")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"权重不存在: {model_path}")

    classes = load_classes(model_path)
    clf = OnnxClassifier(model_path) if model_path.suffix == ".onnx" else TorchClassifier(model_path)
    files = gather(args.inputs)
    if not files:
        raise SystemExit("没有找到任何图片")

    out = []
    for f in files:
        probs = np.asarray(clf(f), dtype=float)
        order = np.argsort(probs)[::-1][: args.topk]
        rec = {
            "file": str(f),
            "pred": classes[int(order[0])],
            "confidence": round(float(probs[order[0]]), 4),
            "topk": [{"class": classes[int(i)], "prob": round(float(probs[i]), 4)} for i in order],
        }
        out.append(rec)
        if not args.json:
            print(f"{f.name}")
            print(f"  -> {rec['pred']}  ({rec['confidence']:.4f})")
            for t in rec["topk"][1:]:
                print(f"     {t['class']:<10} {t['prob']:.4f}")

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
