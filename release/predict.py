"""漫画画风分类推理脚本。

最简单的用法——只给一张图，模型自动选：

    python predict.py 我的图.jpg

其他用法：

    python predict.py 图片或目录             # 用默认的 cover 三分类模型
    python predict.py --model panel 图.jpg   # 指定模型
    python predict.py --self-test            # 不需要图片，验证环境
    python predict.py --list                 # 列出可用模型

支持两种权重，输出完全一致（已验证逐位相同）：
  * ONNX（推荐）：只需 onnxruntime + pillow + numpy，不依赖 ultralytics/torch
  * .pt          ：需要装 ultralytics
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
WEIGHT_EXT = {".pt", ".onnx"}
SIZE = 224

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

# 昵称 -> 文件名主干，少打字
ALIASES = {
    "cover": "comic-style-cover-yolo11n-cls",
    "panel": "comic-style-panel-yolo11n-cls",
    "cover-gray": "comic-style-cover-gray-yolo11n-cls",
    "panel-gray": "comic-style-panel-gray-yolo11n-cls",
}
DEFAULT_ALIAS = "cover"

USAGE_HINT = """
──────────────────────────────────────────────────────────────
 漫画画风分类 · 用法
──────────────────────────────────────────────────────────────

 最少只需给一张图：

     python predict.py 我的图.jpg
     python predict.py 某个文件夹/

 换个模型（cover 认封面/单幅插画，panel 认漫画单格）：

     python predict.py --model panel 我的图.jpg

 先确认环境装好了没（不需要任何图片）：

     python predict.py --self-test

 查看有哪些模型：

     python predict.py --list

 没有图片可测？随手截一张漫画封面或动画截图存成 jpg 即可，
 手机拍一本漫画书封面也行。

──────────────────────────────────────────────────────────────
"""


def resolve_model(name, prefer="onnx"):
    """把用户给的模型名解析成实际文件路径。

    接受昵称（cover）、文件名主干（comic-style-cover-yolo11n-cls）或完整路径。
    相对路径会同时按「当前工作目录」和「脚本所在目录」解析，
    因此在任何目录下执行都能找到模型。
    """
    if not name:
        name = DEFAULT_ALIAS
    stem = ALIASES.get(name, name)

    p = Path(name)
    cands_path = [p] if p.is_absolute() else [Path.cwd() / p, HERE / p]
    for cand in cands_path:
        if cand.exists() and cand.is_file():
            return cand

    for ext in (f".{prefer}", ".onnx", ".pt"):
        f = MODELS_DIR / f"{stem}{ext}"
        if f.exists():
            return f
    loose = [f for f in MODELS_DIR.glob(f"*{stem}*") if f.suffix in WEIGHT_EXT]
    if loose:
        loose.sort(key=lambda f: (f.suffix != f".{prefer}", len(f.name)))
        return loose[0]
    return None


def list_models():
    if not MODELS_DIR.exists():
        print(f"找不到模型目录: {MODELS_DIR}")
        return 1
    print(f"可用模型（位于 {MODELS_DIR}）:\n")
    print(f"  {'昵称':<13}{'类别数':<7}{'识别类别':<28}文件")
    print("  " + "-" * 72)
    for alias, stem in ALIASES.items():
        files = sorted(f.suffix.lstrip(".") for f in MODELS_DIR.glob(f"{stem}.*"))
        if not files:
            continue
        try:
            cls = load_classes(MODELS_DIR / f"{stem}.onnx")
            n, names = len(cls), "/".join(cls)
        except Exception:
            n, names = "?", "?"
        mark = "   ← 默认" if alias == DEFAULT_ALIAS else ""
        print(f"  {alias:<13}{n:<7}{names:<28}{'+'.join(files)}{mark}")
    return 0


def self_test():
    """不依赖任何图片，验证「依赖装齐 + 模型能加载 + 推理链路通」。"""
    print("自检中...\n")
    ok = True
    for mod in ("numpy", "PIL", "onnxruntime"):
        try:
            m = __import__(mod)
            print(f"  [OK] {mod} {getattr(m, '__version__', '')}")
        except ImportError:
            ok = False
            print(f"  [缺失] {mod}  →  请运行: pip install pillow numpy onnxruntime")
    if not ok:
        return 1

    model = resolve_model(DEFAULT_ALIAS, prefer="onnx")
    if model is None:
        print(f"\n  [缺失] 找不到模型文件，请确认 {MODELS_DIR} 下有 .onnx 文件")
        return 1
    print(f"  [OK] 模型 {model.name} ({model.stat().st_size / 1e6:.1f} MB)")

    try:
        classes = load_classes(model)
        print(f"  [OK] 类别 {classes}")
    except Exception as e:
        print(f"  [失败] 读类别出错: {e}")
        return 1

    # 造一张合成图跑通链路（这张图没有语义，只为验证流程能走完）
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    for i in range(SIZE):
        img[i, :] = (i % 256, (i * 2) % 256, (i * 3) % 256)
    tmp = HERE / "_selftest.jpg"
    Image.fromarray(img[:, :, ::-1]).save(tmp)
    try:
        probs = np.asarray(clf_call(model, tmp), dtype=float)
        print(f"  [OK] 推理链路通，输出 {len(probs)} 个概率，和 = {probs.sum():.4f}")
        print(f"       预测 = {classes[int(np.argmax(probs))]}（合成图无意义，仅验证流程）")
    except Exception as e:
        print(f"  [失败] 推理出错: {type(e).__name__}: {e}")
        return 1
    finally:
        tmp.unlink(missing_ok=True)

    print("\n环境正常，可以跑真实图片了：")
    print("    python predict.py 我的图.jpg")
    return 0


def clf_call(model_path, image_path):
    """加载并推理一次（.onnx 或 .pt 自动分派）。"""
    if Path(model_path).suffix == ".onnx":
        return OnnxClassifier(model_path)(image_path)
    return TorchClassifier(model_path)(image_path)


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
    files, missing = [], []
    for it in inputs:
        p = Path(it)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        elif p.is_file():
            files.append(p)
        else:
            missing.append(it)
    return files, missing


def main():
    ap = argparse.ArgumentParser(
        description="漫画画风分类推理",
        epilog="不确定怎么用就直接运行 python predict.py，会打印用法示例。",
    )
    ap.add_argument("model", nargs="?", help="模型：昵称(cover/panel)或 .pt/.onnx 路径。省略则用默认 cover 模型")
    ap.add_argument("inputs", nargs="*", help="图片文件或目录")
    ap.add_argument("--model", dest="model_opt", help="显式指定模型，等价于第一个位置参数")
    ap.add_argument("--topk", type=int, default=3, help="显示前几个类别（默认 3）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本调用")
    ap.add_argument("--list", action="store_true", help="列出可用模型后退出")
    ap.add_argument("--self-test", action="store_true", help="自检环境，不需要任何图片")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.list:
        return list_models()

    # 允许「只给一张图」——此时第一个位置参数其实是图片，不是模型
    model_token, inputs = args.model, list(args.inputs)
    if args.model_opt:
        model_token = args.model_opt
    elif model_token and Path(model_token).suffix.lower() not in WEIGHT_EXT:
        inputs = [model_token] + inputs
        model_token = None

    # 完全没给参数 → 打印友好用法，而不是抛 argparse 错误
    if not model_token and not inputs:
        print(USAGE_HINT)
        return 0

    model_path = resolve_model(model_token)
    if model_path is None:
        print(f"找不到模型 {model_token!r}。可用模型：\n")
        list_models()
        return 1

    if not inputs:
        print(USAGE_HINT)
        return 0

    classes = load_classes(model_path)
    clf = OnnxClassifier(model_path) if model_path.suffix == ".onnx" else TorchClassifier(model_path)
    files, missing = gather(inputs)

    for m in missing:
        print(f"  [跳过] 路径不存在: {m}", file=sys.stderr)
    if not files:
        print(
            f"\n没有找到任何图片。你给的路径是: {inputs}\n"
            "  提示：把 '我的图.jpg' 换成你电脑上真实的图片路径，例如\n"
            "        python predict.py C:/Users/你/Desktop/某张封面.jpg\n"
            "        或给一个文件夹: python predict.py C:/Users/你/Pictures/漫画\n",
            file=sys.stderr,
        )
        return 1

    if not args.json:
        print(f"模型: {model_path.name}   识别类别: {classes}\n")

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
