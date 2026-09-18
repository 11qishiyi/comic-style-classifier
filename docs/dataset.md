# 数据集：来源、许可与构建

## 1. 调研结论：没有现成的三风格数据集

立项前检索了 HuggingFace（经镜像）、Kaggle（经公开 API 逐条核实）、Roboflow、
Papers with Code、arXiv 与中文学术库，
**不存在带「日漫 / 国漫 / 欧美漫」三分类标签的图像数据集**。

最接近的三个：

| 数据集 | 规模 | 标签 | 许可 | 缺口 |
|---|---|---|---|---|
| [ACN-Cover](https://www.kaggle.com/datasets/chauaoe999/comic-classification-by-country) | 1,890 张封面 | 中国 / 韩国 / 日本 三分类 | Apache-2.0 | 无欧美类，且是低分辨率缩略图 |
| [Anime vs Manga vs Cartoon vs Comics vs Human](https://www.kaggle.com/datasets/eissam0/anime-vs-manga-vs-cartoon-vs-comics-vs-human-image) | 5.0 GB | 5 类 | CC0 | 无国漫 |
| [Manga vs Classic Comic Arts](https://www.kaggle.com/datasets/kaustubhrastogi17/manga-and-classic-comic-arts) | 278 MB | 日漫 / 欧美 二分类 | MIT | 缺国漫 |

学术界的数据集（Manga109 日漫 21,142 页、COMICS 美漫约 120 万格、eBDtheque 法比漫画 100 页）
都是分格 / 文本检测任务的数据集，**不含跨文化风格标签**。因此本数据集由多来源拼装构建。

## 2. 数据来源

| 子集 | 类别 | 来源 | 形态 | 原始数量 | 许可 |
|---|---|---|---|---|---|
| cover | japanese | Kaggle `chauaoe999/comic-classification-by-country` (`manga/`) | 封面 190×247 | 630 | Apache-2.0（上传者声明） |
| cover | chinese | 同上 (`manhua/`) | 封面 190×247 | 630 | Apache-2.0（上传者声明） |
| cover | western | Kaggle `kaustubhrastogi17/manga-and-classic-comic-arts` (`classic/`) | 封面 + 部分内页 | 310 | MIT（上传者声明） |
| panel | japanese | 同上 (`manga/`) 的整页，XY-cut 切成单格 | 分格，黑白印刷 | 311 页 → 1,314 格 | MIT（上传者声明） |
| panel | western | [COMICS (UMD)](https://obj.umiacs.umd.edu/comics/) `raw_panel_images.tar.gz` | 分格，彩色印刷 | 18,006 格 / 60 本书 | 原始作品为公有领域 |

### 主动排除的来源

Kaggle `cenkbircanoglu/comic-books-classification`（52,156 张 DC/Marvel 内页扫描）
是规模最大的欧美漫画图像集，但其文件路径含 `GetComics.INFO`、
`(Digital) (Nahga-Empire)` 等**盗版发布组标记**，
许可证为上传者自填、不可信。**本项目未使用**。

### 国漫的先天短板

符合授权的现代中国漫画内页在公开渠道几乎不存在。
本项目只使用封面级素材，**未抓取任何商业漫画平台的付费或试读内容**。

## 3. 数据集统计

```
datasets/comic_style/
├── cover/          # 主集：封面级三分类（共 1,528 张）
│   ├── train/{japanese,chinese,western}/   502 / 504 / 216
│   ├── val/  {japanese,chinese,western}/    63 /  63 /  27
│   └── test/ {japanese,chinese,western}/    63 /  63 /  27
├── panel/          # 副集：分格级二分类（共 2,124 张）
│   ├── train/{japanese,western}/           423 / 1256
│   ├── val/  {japanese,western}/            54 /  163
│   └── test/ {japanese,western}/            54 /  174
├── cover_gray/  panel_gray/   # 上面两套的灰度版本
└── clean/                     # 清洗后全量 + manifest.csv（19,747 张）
```

类别名用 ASCII（`japanese` / `chinese` / `western`）而非中文，
规避 Windows 下非 ASCII 路径导致的读写失败。

## 4. 划分设计（本项目最大的一个坑）

划分 8:1:1，**按作品分层**——同一部作品 / 同一页漫画 / 同一本书不跨 train 与 val/test。

但仅仅「按作品分」还不够，**分配方式**同样关键：

早期实现按图片数贪心分配，结果 COMICS 里页数最多的 956 号书
**一本书就占了欧美测试集的 79%**，整体准确率退化成「这一本书的得分」，
panel 准确率只有 0.764。

改成 **`val`/`test` 优先用「小作品」填满配额**后：

| | 修复前 | 修复后 |
|---|---|---|
| panel 测试集 | 2 本书，最大组占 79% | **65 个作品组，最大组占 9.6%** |
| panel 准确率 | 0.764（假象） | **0.9693** |

相关参数：`--max-per-group 120`（单作品上限）、`--min-groups 8`（val/test 至少覆盖的作品数）、
`--max-ratio 3.0`（多数类下采样上限）。

## 5. 清洗流程

由 `scripts/02_clean.py` 完成：

1. 剔除无法解码、极小（短边 < 150px）、长宽比异常（< 0.2 或 > 5）的图
2. **剔除纯文字页 / 广告页**：用连通域分析判断前景是否由大量细碎文本块构成
   （判据：小连通域前景占比 > 0.62 且每百万像素的小连通域数 > 90）。
   被剔除的样本会输出成对比图供人工复核
3. **感知哈希（phash）跨来源去重**，重复图只保留一份
4. 统一 EXIF 方向、转 RGB JPEG、长边限制 640

## 6. 分格切分

日漫整页需要切成单格（`scripts/04_panelize.py`），用递归 XY-cut。

只用「白色装订线」判据时每页只能切出 **1.4 格**——因为大量相邻格子
仅靠一条**黑色边框线**分隔，黑线属于「高墨迹」带会被漏掉。
补上「细长贯穿黑线 = 边框线」判据后升到 **4.3 格/页**。

欧美分格图无需切分：COMICS 数据集自带分格标注，直接取 `raw_panel_images.tar.gz`。

## 7. 复现步骤

```bash
pip install -r requirements.txt

# Kaggle 凭据：API token 放到 ~/.kaggle/access_token
# (https://www.kaggle.com/settings/api 生成)

# 1. 下载原始数据（COMICS 需顺序读取 61 GiB 的 tar 包，很耗时）
python scripts/01_download.py --comics-n 18000

# 2. 日漫整页 -> 单格（--debug 输出切分可视化）
python scripts/04_panelize.py --debug 8

# 3. 清洗 + 去重
python scripts/02_clean.py

# 4. 按作品分层划分
python scripts/03_build_split.py --clean --max-ratio 3.0 --min-groups 8 --max-per-group 120

# 5. 灰度版本（消融实验用）
python scripts/07_make_gray.py --src datasets/comic_style/cover --dst datasets/comic_style/cover_gray
python scripts/07_make_gray.py --src datasets/comic_style/panel --dst datasets/comic_style/panel_gray
```

## 8. 遵守的 Ultralytics 分类格式

依据[官方文档](https://docs.ultralytics.com/datasets/classify/)：

- split-directory 结构：`train` + `val`（可选 `test`），每个目录下按类别建子目录
- **分类任务不需要 `data.yaml`**，`data` 参数直接指向数据集根目录
- 图像需唯一命名，用 JPEG / PNG 等常见格式

本项目全部符合。
