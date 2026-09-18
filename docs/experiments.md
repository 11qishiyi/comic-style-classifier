# 实验结果与消融

模型：`yolo11n-cls.pt`（Ultralytics 官方 ImageNet-1k 预训练分类权重，5.79 MB / 1000 类）
训练配置：尺寸 224、batch 32、CPU（i5-12500H）、cover 30 epochs / panel 18 epochs

## 1. 主结果

| 模型 | 训练数据 | 测试数据 | 准确率 | 各类召回率 |
|---|---|---|---|---|
| cover | cover | cover（153 张） | **0.8627** | chinese 0.825 / japanese 0.841 / western **1.000** |
| panel | panel | panel（228 张 / 65 组） | **0.9693** | japanese 0.926 / western 0.983 |
| cover 灰度 | cover_gray | cover_gray | 0.8497 | chinese 0.778 / japanese 0.873 / western 0.963 |
| panel 灰度 | panel_gray | panel_gray | **0.9605** | — |

混淆矩阵见 `datasets/comic_style/_eval/cm_*.jpg`（需本地跑过实验才有）。

## 2. 消融：彩色模型推理被改造过的测试图

| 条件 | cover | panel |
|---|---|---|
| 基线 | 0.8627 | 0.9693 |
| 测试图转灰度 | 0.8039 (−5.9) | 0.6447 (−32.5) |
| 测试图短边压到 190 | 0.8431 (−2.0) | — |
| 灰度 + 降采样 | 0.7843 (−7.8) | — |

**这组数字有个陷阱，不要直接用来下结论。**

用彩色图训练的模型去推理灰度图，是分布外（OOD）输入，
掉分可能只是因为模型没见过灰度图，不能推断「它原本只靠颜色分类」。

panel 掉了 32.5 分看着很吓人，但**用灰度图重新训练后准确率回到 0.9605**
（只比彩色低 0.9 分），说明那 32.5 分几乎全是分布外伪影。

**要下颜色相关的结论，请用第 1 节的灰度训练结果，而不是本节。**
这也是 `scripts/07_make_gray.py` 存在的理由。

## 3. 跨域：形态不匹配时无法迁移

| 方向 | 准确率 |
|---|---|
| panel 模型 → cover 图（日 vs 欧美，90 张） | 0.8222 |
| cover 模型 → panel 图（日 vs 欧美，228 张） | 0.8070 |

两个方向都明显低于同域的 0.96–0.97，说明模型学到的是
**与图像形态绑定的线索**，而不是一个可迁移的「画风」概念。
这是整页 / 分格 / 封面不能混着训的直接证据。

## 4. 错误分析：中↔日混淆的本质是标签口径问题

cover 测试集的 19 个错误里，**18 个落在 chinese↔japanese 之间**
（另 1 个 japanese→western），western 类 27 张全部正确。

把错判样本导出来逐张看（`scripts/06_eval.py --save-errors`）：

- 标注为「日漫」却判成「国漫」的，包括《THE BEGINNING AFTER END》、
  《極黒のブリュンヒルデ》等**确凿的日本作品**，但它们的封面是厚涂写实半 3D 风格
- 标注为「国漫」却判成「日漫」的，包括《极品医圣》（国产网文）等，
  但封面用的是**日式动画画风**

**根因**：ACN-Cover 按**作品来源国**标注，而封面画风与来源国是交叉的——
日本轻小说会用厚涂国漫风封面，中国网文会用日式动画风封面。
模型的「错误」很多时候是画风与国籍不一致，而不是识别失败。

这意味着**本数据集的上限受「来源国」与「画风」的相关性限制**。
想进一步提升中↔日区分度，需要改成按画风而非按国别标注。

## 5. 按作品交叉验证：泛化能力到底有多强

单次划分的测试集虽已覆盖 65 个作品组，但每组只有 3–20 张。
为彻底回答「换个没见过的作品还行不行」，用 `scripts/08_cv_books.py` 做了 5 折交叉验证：
把 60 本书 + 531 个日漫分格按作品分成 5 折，每折的模型都**从未见过**该折的任何作品。

| 指标 | 值 |
|---|---|
| 各折准确率 | 0.9674 / 0.9734 / 0.9878 / 0.9913 / 0.9923 |
| 折间平均 | **0.9824 ± 0.0101** |
| 合并样本外 (OOF) | **0.9773**（n=4,049） |
| 按类别 | western 0.99（3,518 张）/ japanese 0.93（531 张） |

按作品拆开看（59 个作品组，每组 60 张）：

| 准确率 | 作品组数 |
|---|---|
| = 1.00 | **43** |
| ≥ 0.95 | 54 |
| ≥ 0.90 | **56 / 59** |
| < 0.90 | 3（789 号书 0.67、1462 号 0.87、1128 号 0.87） |

**结论**：59 个作品组里有 56 个（95%）在从未见过的情况下准确率 ≥ 90%，
不存在「只在少数几本书上 work」的情况。

尤其值得注意的是 956 号书——它在最早那个坏划分下只有 0.58，
修正划分后达到 0.93，印证了当时的问题出在划分而非模型。

失败集中在 3 本画风特殊的书上（789 号书 0.67 最差）。
结合第 4 节的分析，这类书多半是幽默动物漫画等与主流超级英雄漫画差异较大的题材。

## 6. 复现命令

```bash
# 训练
python scripts/05_train_cls.py --model yolo11n-cls.pt \
    --data datasets/comic_style/cover --name cover --epochs 30 --patience 12
python scripts/05_train_cls.py --model yolo11n-cls.pt \
    --data datasets/comic_style/panel --name panel --epochs 18 --patience 6

# 灰度对照
python scripts/05_train_cls.py --model yolo11n-cls.pt \
    --data datasets/comic_style/cover_gray --name cover_gray --epochs 30 --patience 12
python scripts/05_train_cls.py --model yolo11n-cls.pt \
    --data datasets/comic_style/panel_gray --name panel_gray --epochs 18 --patience 6

# 评估与消融
python scripts/06_eval.py --model runs/classify/cover/weights/best.pt \
    --data datasets/comic_style/cover --per-group --save-errors
python scripts/06_eval.py --model runs/classify/cover/weights/best.pt \
    --data datasets/comic_style/cover --gray
python scripts/06_eval.py --model runs/classify/cover/weights/best.pt \
    --data datasets/comic_style/cover --downscale 190

# 交叉验证
python scripts/08_cv_books.py --subset panel --folds 5 --epochs 5 --patience 2 --cap 60
```

> **超参注意**：patience 设太小会过早早停。曾用 patience=5，在 epoch 9 就停了，
> 测试准确率只有 0.785；改用 patience=12 后升到 0.9111。
> 验证集只有一百多张时早停判据噪声很大，**建议 patience ≥ 10**。
