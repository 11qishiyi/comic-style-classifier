# 漫画画风分类模型（可发布的权重包）

区分漫画作品是**日本 manga / 中国 manhua / 欧美 comic** 画风。

**本目录只含权重与推理代码，不含任何训练图片。**
数据集的构建流程在仓库根目录，原图因版权原因不可分发
（详见 [MODEL_CARD.md](MODEL_CARD.md) 第 8 节）。

---

## 五分钟跑起来

### 1. 装依赖

需要 Python 3.9+。只装三个，**不需要 PyTorch，也不需要 ultralytics**：

```bash
pip install pillow numpy onnxruntime
```

### 2. 自检（不需要图片）

```bash
python predict.py --self-test
```

它会检查依赖、加载模型、跑通推理链路。看到 `环境正常` 就说明装好了。

### 3. 给一张图

**本包不含任何图片**，请自己准备一张：截一张漫画/动画封面、拍一本漫画书封面，
或任何手边的图片都行。假设存成了 `我的图.jpg`。

### 4. 跑

```bash
python predict.py 我的图.jpg
```

输出：

```
模型: comic-style-cover-yolo11n-cls.onnx   识别类别: ['chinese', 'japanese', 'western']

我的图.jpg
  -> western  (0.9918)
     japanese   0.0082
     chinese    0.0000
```

批量跑就给一个文件夹：

```bash
python predict.py 某个文件夹/
```

> 命令在任何工作目录下都能执行——模型路径按脚本自身位置解析，不需要先 `cd` 进来。
> 若你已经在仓库根目录，把命令写成 `python release/predict.py ...` 即可。

---

## 用哪个模型

| 你的输入是什么 | 用哪个 | 命令 |
|---|---|---|
| 封面、单幅插画 | `cover`（三分类，**默认**） | `python predict.py 图.jpg` |
| 漫画的**单格**（分格） | `panel`（二分类） | `python predict.py --model panel 图.jpg` |
| 整页漫画 | 两个都不适合 | 需先把整页切成单格 |

`cover-gray` / `panel-gray` 是消融实验用的对照模型（训练时输入已转灰度），
用来证明模型不是靠颜色分类，**日常使用请忽略**。

查看全部可用模型及其识别类别：

```bash
python predict.py --list
```

---

## 命令行参数

```bash
python predict.py                           # 打印用法示例
python predict.py 图片或目录                 # 默认用 cover 模型
python predict.py --model panel 图片或目录    # 指定模型
python predict.py --topk 1 图片              # 只看第一名
python predict.py --json 图片                # 输出 JSON，便于脚本调用
python predict.py --list                    # 列出模型
python predict.py --self-test               # 环境自检
```

---

## 在 Python 里调用

```python
import sys
sys.path.insert(0, "release")          # 或把 predict.py 复制到你的项目里
from predict import OnnxClassifier, load_classes

clf = OnnxClassifier("release/models/comic-style-cover-yolo11n-cls.onnx")
classes = load_classes("release/models/comic-style-cover-yolo11n-cls.onnx")

probs = clf("我的图.jpg")
print(classes[int(probs.argmax())], float(probs.max()))
```

---

## 文件说明

| 文件 | 说明 |
|---|---|
| `models/*.onnx` | ONNX 权重，只需 `onnxruntime`；也能给 C#/Java/JS/C++/移动端用 |
| `models/*.pt` | PyTorch 权重，需 `pip install ultralytics`；两者预测结果已逐位验证一致 |
| `models/classes.json` | 各模型的类别名与顺序（ONNX 输出是索引，靠它映射成标签） |
| `predict.py` | 推理脚本，`.pt` / `.onnx` 两条路径共用 |
| `manifest.csv` | 训练集构成清单：每张图的来源、作品分组、尺寸、过滤指标。**只含文件名与元数据，不含图片** |
| `MODEL_CARD.md` | 模型卡：指标、预处理规格、已知限制、许可 |

---

## 指标

| 模型 | 本项目测试集 | 按作品交叉验证 |
|---|---|---|
| `cover` | 0.8627 | — |
| `panel` | 0.9693 | 0.9824 ± 0.0101 |

---

## ⚠️ 使用前必读：这个模型能用在哪

外部数据集验证（训练时从未见过的来源）显示**能力边界比测试集数字窄得多**：

| 外部集 | 预期 | 用哪个模型 | 判对率 |
|---|---|---|---|
| 《钢之炼金术师》分格（文字已抹除） | japanese | `panel` | **1.000** |
| Manga109，109 部日漫封面 | japanese | `panel` | **0.862** |
| Manga109（同一批图） | japanese | `cover` | **0.101** |
| Garfield 美国报纸连环画 | western | `cover` | 0.193 |

**`cover` 模型学到的是「封面设计年代感」，不是「民族画风」。**
在真实的日本漫画封面（Manga109，1970s–2010s 复古设计）上只有 10.1% 判对，
因为那些封面在视觉统计上更接近黄金时代美漫。

| 场景 | 可用性 |
|---|---|
| 黑白漫画分格判日漫 / 经典美漫 | ✅ `panel` 可用，已跨来源验证 |
| 现代日系封面 / 国漫封面 / 经典美漫封面 | ✅ `cover` 可用（仅其训练域内） |
| 现代美漫、报纸连环画 | ❌ 两个模型都会判错 |
| 复古日漫单行本封面 | ❌ `cover` 判成欧美 |

完整分析见仓库 `docs/external-validation.md` 与 `docs/limitations.md`。

---

## 两个常见疑问

**「模型是不是在靠文字判断？」** 不是。训练数据未做文字擦除，但涂白消融显示
擦掉文字后 panel 仍有 94.3%、cover 不降反升 1.3 分；且 Garfield（英文）被判成日漫、
Manga109（日文）被判成欧美——文字语言与预测方向相反。
详见仓库 `docs/text-dependency.md`。

**「预处理要怎么写？」** 必须与 Ultralytics 的 `classify_transforms(size=224)` 一致：
短边缩到 **224**（不是 256）→ 中心裁剪 224 → 除以 255 → **不做归一化**。
`predict.py` 里已实现，直接用即可。详见 `MODEL_CARD.md` 第 3 节。

---

## 许可

模型基于 Ultralytics YOLO11 训练，受 **AGPL-3.0** 约束。
个人学习、研究、开源项目可直接使用；商业闭源或 SaaS 部署需向 Ultralytics 购买企业许可。
