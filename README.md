# 漫画画风分类数据集与模型

区分漫画作品的绘画风格来源：**日本 manga / 中国 manhua / 欧美 comic**。

用 Ultralytics YOLO11-cls 训练，数据集遵循
[Ultralytics 分类格式](https://docs.ultralytics.com/datasets/classify/)
（`train/val/test` + 类别子目录，无需 `data.yaml`）。

> **本项目不含训练图片。** 原图含可识别的商业连载漫画，
> 受版权保护不可分发。仓库提供完整的构建脚本，可自行复现数据集。
> 详见 [docs/compliance.md](docs/compliance.md)。

## 结果

| 子集 | 任务 | 测试集规模 | 准确率 |
|---|---|---|---|
| `cover/` | 三分类（封面级） | 153 张 | **0.8627** |
| `panel/` | 二分类（分格级，日 vs 欧美） | 228 张 / **65 个作品组** | **0.9693** |

按作品 5 折交叉验证：**0.9824 ± 0.0101**，
59 个作品组中 **56 个（95%）在完全未见过的前提下准确率 ≥ 90%**，43 个满分。

## 三条主要结论

**1. 画风是真实学会的，不是靠颜色作弊。**
用灰度图**重新训练**（完全剔除颜色信息）后，panel 测试准确率仍有 **0.9605**，
比彩色的 0.9693 只低 0.9 分。模型靠的是线条、网点、字形这些真正的画风线索。

**2. 分格级的泛化能力经过验证。**
5 折交叉验证让每本书都当过一次测试集，56/59 个作品组 ≥ 90%，
不存在「只在少数几本书上 work」的情况。

**3. 中↔日是唯一难点，且受标注口径限制。**
cover 的 19 个错误里 18 个落在 chinese↔japanese 之间，western 零错误。
逐张核对后发现根因是**「作品来源国」与「封面画风」不一致**——
日本轻小说会用厚涂国漫风封面，中国网文会用日式动画风封面。
**若你的需求是按画风而非按国别分类，中↔日边界不可直接采信。**

**4. 但模型的能力边界比测试集数字窄得多 —— 外部验证揭示的。**
在训练时从未使用过、全网另找的外部数据集上验证：`panel` 模型跨来源泛化良好
（《钢之炼金术师》分格 **100%**、Manga109 **86.2%**），
但 **`cover` 模型只有 10.1%**——在 109 部真实日漫封面上，87% 被判成了欧美。
根因是它学到的是「封面设计年代感」而非「民族画风」。
**不要把它当作能识别各国漫画画风的通用模型用。**
详见 [docs/external-validation.md](docs/external-validation.md)。

完整实验与消融见 [docs/experiments.md](docs/experiments.md)。

## 快速开始

```bash
pip install pillow numpy onnxruntime

python release/predict.py release/models/comic-style-cover-yolo11n-cls.onnx 你的图.jpg
```

```
你的图.jpg
  -> western  (0.9918)
     japanese   0.0082
     chinese    0.0000
```

`.onnx` 路径**不需要 ultralytics / torch**，也可用 C#/Java/JS/C++/移动端加载。
需要逐位复现时用 `.pt` 路径（`pip install ultralytics`）。

| 输入 | 用哪个模型 |
|---|---|
| 封面、单幅插画 | `comic-style-cover-yolo11n-cls` |
| 漫画单格（分格） | `comic-style-panel-yolo11n-cls` |
| 整页漫画 | 两个都不适合，需先切分格 |

模型详情、预处理规格与限制见 [release/MODEL_CARD.md](release/MODEL_CARD.md)。

## 目录结构

```
├── release/                    # 可发布的权重包（39 MB）
│   ├── models/                 # 4 个模型 × (.pt, .onnx) + classes.json
│   ├── predict.py              # 推理脚本
│   ├── MODEL_CARD.md           # 模型卡
│   └── manifest.csv            # 训练集构成清单（不含图片）
├── scripts/                    # 数据构建与训练全流程
│   ├── 00_inspect.py           # 抽样核验图像形态
│   ├── 01_download.py          # 下载 Kaggle 数据集 + COMICS 流式抽取
│   ├── 01b_fetch_comics.py     # HTTP Range 流式抽取（带断线重试）
│   ├── 02_clean.py             # 清洗、文字页过滤、phash 去重
│   ├── 03_build_split.py       # 按作品分层划分
│   ├── 04_panelize.py          # 漫画整页 XY-cut 切分格
│   ├── 05_train_cls.py         # 训练
│   ├── 06_eval.py              # 评估与消融
│   ├── 07_make_gray.py         # 生成灰度版数据集
│   ├── 08_cv_books.py          # 按作品交叉验证
│   ├── 09_eval_external.py     # 外部数据集验证
│   └── 10_text_ablation.py     # 文字涂白消融
├── docs/                       # 详细文档
│   ├── dataset.md              # 数据来源、许可、统计、划分设计
│   ├── experiments.md          # 完整实验结果与消融
│   ├── limitations.md          # 已知限制
│   ├── troubleshooting.md      # 9 个工程坑
│   ├── compliance.md           # 上传 GitHub 的合规要点
│   ├── external-validation.md  # 外部数据集验证（揭示能力边界）
│   └── text-dependency.md      # 文字依赖性检验（模型是否在"读文字"）
└── requirements.txt
```

## 复现数据集

```bash
pip install -r requirements.txt

# Kaggle token 放到 ~/.kaggle/access_token
# (https://www.kaggle.com/settings/api 生成)

python scripts/01_download.py --comics-n 18000   # 下载原始数据（COMICS 很耗时）
python scripts/04_panelize.py --debug 8          # 日漫整页 → 单格
python scripts/02_clean.py                       # 清洗 + 去重
python scripts/03_build_split.py --clean --max-ratio 3.0 --min-groups 8 --max-per-group 120
python scripts/05_train_cls.py --model yolo11n-cls.pt \
    --data datasets/comic_style/cover --name cover --epochs 30 --patience 12
```

数据来源、划分设计与踩坑细节见 [docs/dataset.md](docs/dataset.md)。
`yolo11n-cls.pt` 的官方下载地址在国内会 SSL 失败，可用代理前缀
`https://ghfast.top/<原始URL>`。

## 数据来源

| 子集 | 类别 | 来源 | 数量 |
|---|---|---|---|
| cover | japanese / chinese | Kaggle `chuaaoe999/comic-classification-by-country` | 各 630 封面 |
| cover | western | Kaggle `kaustubhrastogi17/manga-and-classic-comic-arts` | 310 张 |
| panel | japanese | 同上 `manga/` 整页切分格 | 311 页 → 1,314 格 |
| panel | western | COMICS (UMD) 黄金时代美漫（公有领域） | 18,006 格 / 60 本书 |

**主动排除**：Kaggle `cenkbircanoglu/comic-books-classification`（52,156 张内页扫描）
文件路径含盗版发布组标记，许可证为上传者自填不可信，本项目未使用。

## 已知限制（速览）

1. **国漫只有封面，没有内页** → `panel/` 副集只能做日 vs 欧美二分类
2. **中↔日受标注口径限制** → 按画风而非按国别分类的需求不适用
3. **`cover/` 三类成像条件不统一** → 中日封面 190×247 缩略图 vs 欧美 474×640
4. **`western` 类偏向经典欧美漫画**（1938–1970 年代），与当代美漫有差距
5. **跨形态不可用** → panel 模型判封面 0.822，cover 模型判分格 0.807

详见 [docs/limitations.md](docs/limitations.md)。

## 许可

- 模型基于 **Ultralytics YOLO11** 训练，受 **AGPL-3.0** 约束
  （依据：`yolo11n-cls.pt` 权重内嵌元数据 `license: AGPL-3.0 License`）。
  个人学习、研究、开源项目可直接使用；商业闭源或 SaaS 部署需购买企业许可。
- 训练数据另有版权，**不可随模型分发原图**。
  详见 [docs/compliance.md](docs/compliance.md)。

---

**环境**：Python 3.14 / torch 2.14.0+cpu / ultralytics 8.4.154。
全部实验在 CPU（i5-12500H）上完成，未使用 GPU。
