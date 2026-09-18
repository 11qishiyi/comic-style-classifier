# 漫画画风分类模型（可发布的权重包）

区分漫画作品是**日本 manga / 中国 manhua / 欧美 comic** 画风。

**本目录只含权重与推理代码，不含任何训练图片。** 数据集的构建流程在仓库根目录，
原图因版权原因不可分发（详见 `MODEL_CARD.md` 第 8 节）。

## 快速开始

```bash
pip install pillow numpy

# ONNX 路径：不需要 ultralytics / torch，装上 onnxruntime 即可
pip install onnxruntime
python predict.py models/comic-style-cover-yolo11n-cls.onnx 你的图.jpg
python predict.py models/comic-style-cover-yolo11n-cls.onnx 某个目录/

# PyTorch 路径：需要 ultralytics
pip install ultralytics
python predict.py models/comic-style-cover-yolo11n-cls.pt 你的图.jpg
```

输出示例：

```
你的图.jpg
  -> western  (0.9918)
     japanese   0.0082
     chinese    0.0000
```

加 `--json` 输出机器可读结果，加 `--topk 1` 只看第一名。

## 该用哪个模型

| 输入 | 用哪个 |
|---|---|
| 封面、单幅插画 | `models/comic-style-cover-yolo11n-cls.onnx`（三分类） |
| 漫画单格（分格） | `models/comic-style-panel-yolo11n-cls.onnx`（二分类） |
| 整页漫画 | 两个都不适合，需先切分格 |

`cover_gray` / `panel_gray` 是消融实验的对照模型（训练时输入已转灰度），
用来证明模型不是靠颜色分类，日常使用请忽略。

## 在 Python 里调用

```python
import sys
sys.path.insert(0, "release")          # 或把 predict.py 放到你的项目里
from predict import OnnxClassifier, load_classes

clf = OnnxClassifier("models/comic-style-cover-yolo11n-cls.onnx")
classes = load_classes("models/comic-style-cover-yolo11n-cls.onnx")
probs = clf("test.jpg")
print(classes[int(probs.argmax())], probs.max())
```

## 文件说明

| 文件 | 说明 |
|---|---|
| `models/*.pt` | PyTorch 权重，需 `ultralytics` 加载 |
| `models/*.onnx` | ONNX 权重，只需 `onnxruntime`，可用 C#/Java/JS/C++/移动端 |
| `models/classes.json` | 各模型的类别名与顺序（ONNX 输出是索引，靠它映射成标签） |
| `predict.py` | 推理脚本，`.pt` / `.onnx` 两条路径共用 |
| `manifest.csv` | 训练集构成清单：每张图的来源、作品分组、尺寸、过滤指标。**只含文件名与元数据，不含图片** |
| `MODEL_CARD.md` | 模型卡：指标、预处理规格、已知限制、许可 |

## 指标

| 模型 | 测试准确率 | 按作品交叉验证 |
|---|---|---|
| `cover` | 0.8627 | — |
| `panel` | 0.9693 | 0.9824 ± 0.0101 |

详细指标、混淆矩阵、消融实验与已知限制见 **[MODEL_CARD.md](MODEL_CARD.md)**。

## 两个必须知道的前提

1. **中↔日边界不可直接采信。** 训练标签是「作品来源国」而非「画风」，
   日本轻小说会用国漫风封面、中国网文会用日式动画风封面，两者本就交叉。
   western 类几乎零错误（召回 1.000），真正难分辨的是中↔日。
2. **预处理必须严格照搬。** 不是 ImageNet 的「缩到 256 再裁 224」，
   而是「短边缩到 224 → 中心裁剪 224 → 除以 255 → 不做归一化」。
   细节见 MODEL_CARD 第 3 节，`predict.py` 里已实现。

## 许可

模型基于 Ultralytics YOLO11 训练，受 **AGPL-3.0** 约束。
个人学习、研究、开源项目可直接使用；商业闭源或 SaaS 部署需向 Ultralytics 购买企业许可。
