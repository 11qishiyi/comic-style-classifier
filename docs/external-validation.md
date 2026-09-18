# 外部数据集验证

## 为什么必须做这一步

项目自带的测试集（cover 153 张、panel 228 张）虽然按作品分层划分、且 panel 过了 5 折交叉验证，
但它们和训练数据**来源相同**。测试集准确率衡量的是「跨作品泛化」，
而**不是**「跨来源泛化」。

用户提出的质疑是对的：*"panel 测试集只有两本书？这泛化能力也太弱了吧"*。
修掉划分缺陷后测试集覆盖了 65 个作品组，但仍不足以回答一个更根本的问题——
**换个完全不同的来源，模型还行不行？**

于是在全网随机找了训练时**从未使用过**的外部数据集做验证。
脚本：`scripts/09_eval_external.py`。

## 用了哪些外部数据

| 外部集 | 内容 | 数量 | 与训练数据的关系 |
|---|---|---|---|
| [whitened-manga-panels](https://www.kaggle.com/datasets/tonystarktony/whitened-manga-panels) | 《钢之炼金术师》漫画分格，**文字气泡已被抹除** | 100 张 | 独立来源、独立作品 |
| [Manga109](https://www.kaggle.com/datasets/guansuo/manga109) | 109 部日漫的单行本封面（1970s–2010s） | 109 张 | 学术界标准日漫数据集，独立来源 |
| [Garfield Comic Strips](https://www.kaggle.com/datasets/navinkasa/garfield-comic-strips-2017-to-2023-public) | 美国报纸连环画（2017–2023） | 400 张（抽样） | 独立来源、独立体裁 |

外部集都是单一类别，所以指标是「**预期类别的判对率**」。

## 结果

| 外部集 | 预期 | 模型 | 判对率 | 主要误判方向 |
|---|---|---|---|---|
| whitened-manga-panels | japanese | `panel` | **1.000** (100/100) | — |
| Manga109 | japanese | `panel` | **0.862** (94/109) | western 15 张 |
| Manga109 | japanese | `cover` | **0.101** (11/109) | **western 95 张 (87%)** |
| Garfield | western | `cover` | **0.193** (77/400) | japanese 322 张 (80%) |
| Garfield | western | `panel` | **0.323** (129/400) | japanese 271 张 (68%) |

## 结论一：panel 模型跨来源泛化良好

`panel` 模型在两个完全独立的日漫来源上分别拿到 **100%** 和 **86.2%**。

《钢之炼金术师》那 100 张的**文字气泡已被抹除**，模型没有文字可依赖，
仍全部判对（平均置信度 0.9947）——说明它学的是线条、网点、字形之外的**画面结构**，
不是"有没有日文"。

## 结论二：cover 模型跨来源泛化失败

**同一批 Manga109 图片，两个模型给出完全相反的结论**：

- `panel` 模型：86.2% 判为日漫 ✓
- `cover` 模型：只有 10.1% 判为日漫，**87.2% 判成了欧美** ✗

Manga109 是确凿的日本漫画（109 部，含《あくはむ》《ホットロード》等），
`cover` 模型却几乎全判成欧美。

### 根因：cover 模型学的是「封面设计年代感」，不是「民族画风」

拆开看训练数据的两端：

| 训练数据 | 视觉特征 |
|---|---|
| `cover/western` = k_classic | 1938–1970 年代美漫封面：**高饱和原色、粗墨线、粗体标题字、网点印刷** |
| `cover/japanese` = ACN-Cover | 现代网络时代日系封面：**柔和淡彩、角色特写、动漫插画风** |

Manga109 的封面虽然出自日本，但设计是**复古浓烈平涂 + 粗体标题**——
在视觉统计上更靠近「黄金时代美漫封面」那一端，于是被判成欧美。

模型区分的是**两段时代的封面设计语言**，而不是「日本画风」和「欧美画风」。

### Garfield 失败的原因相同

Garfield 与现代报纸连环画是**淡彩平涂 + 干净稀疏黑线 + 大量留白**。
量化对比（各 120 张采样，尺寸统一到 224）：

| 集合 | 饱和度 | 明度 | 墨迹占比 |
|---|---|---|---|
| 训练集 `panel/western` | 89.9 | 161.9 | **0.418** |
| 训练集 `panel/japanese` | 44.4 | 172.4 | 0.358 |
| 外部 Garfield | 65.8 | **189.9** | **0.228** |
| 外部《钢炼》 | 0.0 | 194.6 | 0.221 |

Garfield 是所有集合里**最亮、墨迹最少**的，墨迹占比比训练集的 japanese 类还低。
模型的 `western` 概念建立在「高饱和、高墨迹密度的老式印刷」上，
遇到现代干净线条的欧美漫画就直接失效。

## 结论三：已发布模型的适用边界

**`cover` 模型不能用来说明"模型能识别各国漫画画风"。** 它的 0.8627 测试集准确率是
**特定域内**的指标（现代日系封面 vs 中国网络漫画封面 vs 经典美漫封面），
离开这个域就崩。

相比之下 `panel` 模型更接近真正的画风识别，但它的 `western` 类同样偏向经典美漫印刷风格。

| 使用场景 | 是否可用 |
|---|---|
| 判别现代日系封面 / 国漫封面 / 经典美漫封面 | ✅ `cover` 可用（其训练域内） |
| 判别黑白漫画分格是日漫还是经典美漫 | ✅ `panel` 可用，跨来源验证过 |
| 判别现代美漫 / 报纸连环画 | ❌ 两个模型都会判错 |
| 判别复古日漫单行本封面 | ❌ `cover` 会判成欧美 |

## 如何改进

1. **扩充 `western` 类的时代跨度**：加入现代美漫、报纸连环画、欧洲漫画（BD），
   而不是只有 1938–1970 年代的黄金/白银时代作品。
2. **扩充 `japanese` 类的时代跨度**：加入 1970–1990 年代的复古单行本封面，
   目前只有网络时代的现代封面。
3. **按画风而非按来源国标注**：这是 [experiments.md 第 4 节](experiments.md#4-错误分析中日混淆的本质是标签口径问题)
   已经指出的根本问题，外部验证进一步印证了它。
4. **报告指标时标明域**：不要给单一准确率数字，而应说明"在哪个域上"。

## 复现命令

```bash
# 下载外部数据
kaggle datasets download -d tonystarktony/whitened-manga-panels -p datasets/external/wmp --unzip
kaggle datasets download -d guansuo/manga109 -p datasets/external/manga109 --unzip
kaggle datasets download -d navinkasa/garfield-comic-strips-2017-to-2023-public -p datasets/external/garfield

# 逐个验证（脚本会输出判对率、误判方向、错判样本对比图）
python scripts/09_eval_external.py --model release/models/comic-style-panel-yolo11n-cls.onnx \
    --dir datasets/external/whitened-manga-panels --expect japanese --tag wmp_panel
python scripts/09_eval_external.py --model release/models/comic-style-cover-yolo11n-cls.onnx \
    --dir datasets/external/manga109 --expect japanese --limit 0 --tag manga109_cover
python scripts/09_eval_external.py --model release/models/comic-style-panel-yolo11n-cls.onnx \
    --dir datasets/external/manga109 --expect japanese --limit 0 --tag manga109_panel
python scripts/09_eval_external.py --model release/models/comic-style-cover-yolo11n-cls.onnx \
    --dir "datasets/external/garfield/extracted" --expect western --limit 400 --tag garfield_cover
python scripts/09_eval_external.py --model release/models/comic-style-panel-yolo11n-cls.onnx \
    --dir "datasets/external/garfield/extracted" --expect western --limit 400 --tag garfield_panel
```

结果与错判样本保存在 `datasets/comic_style/_eval_external/`。
