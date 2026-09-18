# Model Card — 漫画画风三分类 / 二分类

## 1. 概览

| 模型 | 任务 | 类别 | 测试准确率 | 文件 |
|---|---|---|---|---|
| `cover` | 三分类（封面级） | chinese / japanese / western | **0.8627** (n=153) | `comic-style-cover-yolo11n-cls.{pt,onnx}` |
| `panel` | 二分类（分格级） | japanese / western | **0.9693** (n=228) | `comic-style-panel-yolo11n-cls.{pt,onnx}` |
| `cover_gray` | 三分类（灰度，消融用） | chinese / japanese / western | 0.8497 (n=153) | `comic-style-cover-gray-yolo11n-cls.{pt,onnx}` |
| `panel_gray` | 二分类（灰度，消融用） | japanese / western | 0.9605 (n=228) | `comic-style-panel-gray-yolo11n-cls.{pt,onnx}` |

以上文件均位于 `models/` 目录下。

**`cover_gray` / `panel_gray` 是消融实验用的对照模型**，训练时输入图已转灰度。
它们的用途是回答「模型是不是只靠颜色在分类」——结论是不靠，
`panel` 去掉全部颜色信息只掉 0.9 分。日常使用请用 `cover` 和 `panel`。

## 2. 架构与训练

- 基座：**YOLO11n-cls**（Ultralytics），ImageNet-1k 预训练，1.7M 参数，4.5 GFLOPs
- 输入：224×224 RGB
- 训练：CPU（i5-12500H），batch 32，cover 训练 30 epochs、panel 18 epochs
- 分类头按本数据集类别数重建，骨干网络沿用 ImageNet 特征

> 官方分类权重 `yolo11n-cls.pt` 的下载地址是 GitHub Releases，国内直连会 SSL 失败，
> 可用代理前缀：`https://ghfast.top/<原始URL>`。

## 3. 预处理规格（重要）

推理预处理**必须**与 `ultralytics.data.augment.classify_transforms(size=224)` 一致：

```
Resize(224, BILINEAR)   # 短边缩放到 224，保持长宽比
CenterCrop(224)         # 中心裁剪 224×224，偏移量 = int(round((边长-224)/2.0))
ToTensor()              # 除以 255
Normalize((0,0,0),(1,1,1))   # 等于不做归一化
```

**两个容易踩的坑**（都实测过会造成明显精度偏差）：

1. **不是 ImageNet 那套「短边缩到 256 再裁 224」**。
   Ultralytics 的 `crop_fraction` 参数已废弃，现在直接缩到 224。
   按老惯例写会让同一张图给出完全不同的置信度分布。
2. **不做 ImageNet 均值方差归一化**。Ultralytics 分类用的是 0~1 原始像素。
3. **中心裁剪要用 `round` 而非整除**。当 `边长-224` 为奇数时（例如 291-224=67），
   `round(33.5)=34` 而 `67//2=33`，裁切位置差 1 像素，在小图上足以让置信度偏移 0.3。

`predict.py` 已按上述规格实现并验证。

## 4. ONNX 与 PyTorch 路径的一致性

在本项目测试集上逐张比对（同样的预处理规格）：

| 模型 | 预测一致率 | 准确率差异 | 概率最大偏差 |
|---|---|---|---|
| `cover` | 153/153 (100%) | 0.00 分 | 0.0055 |
| `panel` | 227/228 (99.6%) | 0.44 分 | 0.1243 |
| `cover_gray` | 153/153 (100%) | 0.00 分 | — |
| `panel_gray` | 228/228 (100%) | 0.00 分 | — |

`panel` 的 1 张差异出现在决策边界上（438×391 的图被 `.pt` 判为 western、被 `.onnx` 判为 japanese），
根因是 PIL 与 torchvision 在大幅下采样时的抗锯齿核不同。
**需要逐位复现请用 `.pt` 路径**；追求轻量依赖用 `.onnx` 即可，差异可忽略。

另外提醒：Ultralytics 导出的 ONNX 分类模型**计算图里已含 softmax**，输出就是概率。
再套一层 softmax 会把分布压平（`[1,0,0]` → `[0.576,0.212,0.212]`），
类别看起来还对但置信度全错。`predict.py` 里做了自动判别。

## 5. 训练数据

| 子集 | 类别 | 来源 | 数量 |
|---|---|---|---|
| cover | japanese | Kaggle `chauaoe999/comic-classification-by-country` (`manga/`) | 630 封面 |
| cover | chinese | 同上 (`manhua/`) | 630 封面 |
| cover | western | Kaggle `kaustubhrastogi17/manga-and-classic-comic-arts` (`classic/`) | 310 张 |
| panel | japanese | 同上 `manga/` 的整页，XY-cut 切分格 | 311 页 → 1,314 格 |
| panel | western | COMICS (UMD) `raw_panel_images.tar.gz` | 18,006 格 / 60 本书 |

划分按**作品分层**（同一部作品不跨 train/val/test），测试集覆盖 65 个作品组。

完整构建流程见仓库根目录的 `README.md` 与 `scripts/`。

## 6. 性能细节

### cover（三分类）

| 类别 | 召回率 | 精确率 |
|---|---|---|
| chinese | 0.825 | 0.852 |
| japanese | 0.841 | 0.828 |
| western | **1.000** | 0.964 |

**全部 19 个错误里 18 个落在 chinese↔japanese 之间，western 零错误。**

### panel（二分类）

| 类别 | 召回率 |
|---|---|
| japanese | 0.926 |
| western | 0.983 |

按作品 5 折交叉验证（每本书都当过测试集）：**0.9824 ± 0.0101**，
59 个作品组中 **56 个（95%）在完全未见过的前提下准确率 ≥ 90%**，43 个满分。

## 7. 已知限制（使用前请读）

1. **中↔日区分受标注口径限制，不是模型能力问题。**
   训练数据的标签是**作品来源国**，而封面画风与来源国是交叉的：
   日本轻小说会用厚涂国漫风封面，中国网文会用日式动画风封面。
   逐张核对错判样本后确认，模型的「错误」多数是画风与国籍不一致。
   **如果你的需求是「按画风分类」而不是「按来源国分类」，本模型的中↔日边界不可直接采信。**
2. **`western` 类偏向经典欧美漫画**（1938–1970 年代），与当代欧美漫画有差距。
3. **跨形态不可用**。用 `panel` 模型判封面只有 0.822，用 `cover` 模型判分格只有 0.807。
   请按输入形态选择对应模型：
   - 单幅封面 / 插画 → `cover`
   - 漫画单格 → `panel`
   - 整页漫画 → 两个都不合适，需先切分格或自行微调
4. **国漫只有封面级数据**，`chinese` 类没有内页样本。
5. 输入图像很小（如 190×247 的缩略图）时置信度会偏低，但仍可用。

## 8. 许可

- 本模型基于 **Ultralytics YOLO11** 训练，受 **AGPL-3.0** 约束。
  Ultralytics 官方说明：AGPL-3.0 免费档适用于学术研究、个人项目与开源应用，
  要求以 AGPL-3.0 公开整个衍生项目的源码；商业闭源或 SaaS 部署需购买企业许可。
  依据见 `yolo11n-cls.pt` 权重内嵌的元数据 `license: AGPL-3.0 License`。
- **训练数据本身另有版权，不可随模型一起分发原图**：
  日漫分格含可识别的商业连载作品（Bleach、进击的巨人、Naruto 等），
  Kaggle 上传者声明的 MIT / Apache-2.0 覆盖不了这些图片的原始版权。
  详见根目录 `README.md` 的合规说明。
- 训练权重的法律地位在多数法域尚无定论（涉及用受版权保护的作品训练模型）。
  本项目仅作个人学习与交流用途。
