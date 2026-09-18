# 上传 GitHub 的合规要点

## 结论

| 内容 | 能否上传 | 原因 |
|---|---|---|
| 模型权重 | ✅ 可以 | 受 AGPL-3.0 约束，个人学习 / 研究 / 开源项目在许可范围内 |
| 训练 / 推理代码 | ✅ 可以 | AGPL-3.0 要求开源衍生作品源码，这正好满足 |
| 数据集原图 | ❌ **不可以** | 含可识别的商业连载漫画，Kaggle 上传者的声明覆盖不了原始版权 |
| manifest 等元数据 | ✅ 可以 | 只含文件名与统计信息，不含图片本体 |

仓库根目录的 `.gitignore` 已把 `datasets/`、`runs/`、`scratch/`、`.zcode/`
与预训练权重全部排除，Kaggle 凭据也不在仓库内。
**上传前务必 `git status --short` 再确认一遍。**

## 1. 模型：AGPL-3.0，合规做法

三条依据：

1. Ultralytics 仓库的 `LICENSE` 文件是 **GNU AGPL-3.0** 全文
2. **`yolo11n-cls.pt` 权重文件自身的元数据里写着**
   `license: AGPL-3.0 License (https://ultralytics.com/license)`
   ——这是嵌在权重里面的，不看文档也能验证
3. [官方许可页](https://www.ultralytics.com/license) 说明：
   AGPL-3.0 免费档适用于学术研究、个人项目与开源应用；
   合规意味着 *"publicly releasing the complete corresponding source code
   for the entire derivative work"*，包括脚本与配置文件、**某些情况下还包括模型权重**；
   **所有训练出的模型默认受 AGPL-3.0 约束**。
   商业产品、闭源软件、SaaS / API / 云部署需购买 Enterprise License。

**满足合规的做法：**

- 仓库根目录放 AGPL-3.0 的 `LICENSE` 全文（本项目已有）
- 训练脚本、配置一并开源（本项目 `scripts/` 已全套提供）
- README 写明「基于 Ultralytics YOLO11 训练，受 AGPL-3.0 约束」
- 留意 **AGPL 的网络条款**：若把模型部署为网络服务（网页 / API），
  也必须向使用者提供源码。这是 AGPL 比 GPL 更严的地方。

**一个无法回避的不确定性**：模型是从受版权保护的漫画图像上训练得到的。
训练权重的法律地位在多数法域尚无定论（美国几起相关诉讼未决）。
AGPL-3.0 是眼下唯一确定需要处理的许可问题，但原著作权人若主张，风险不为零。

## 2. 数据集：为什么不能传

核心法理是一句常被忽略的话：
**Kaggle 上传者勾选 MIT / Apache-2.0，不等于他有权给你这个许可**——
这些图不是他创作的。

逐源实测结果：

| 来源 | 数量 | 版权状况 |
|---|---|---|
| `kaustubhrastogi17` (manga) | 531 格 | **177 格可定位到具体商业连载作品**：Bleach 46、进击的巨人 50、Naruto 21、hng_m 37、dbm 23。上传者描述里自认 "webscraped" |
| ACN-Cover | 1,260 张 | 中日韩商业漫画封面（可见《DEATH NOTE》《七つの大罪》等），上传者无权授权 |
| `k_classic` | 270 张 | Superman / Captain America / Marvel 封面，上传者自认 webscraped |
| COMICS (UMD) | 17,688 格 | 源材料按美国法已入公有领域，但数据集页面无许可声明；且中国保护期为作者终生 + 50 年 |

**前三项明确不能上传**——它们是商业作品的复制品，GitHub 会受理 DMCA 下架请求。

**第四项（COMICS）是唯一有讨论空间的，但仍不建议：**

- 按**美国法**源材料已过版权期，在美国境内再分发合法
- 但**中国**《著作权法》保护期是作者终生 + 50 年，
  1938–1954 年的美国漫画若作者 1976 年后去世，在中国可能仍在保护期内
- Superman、Batman 这类角色，即使具体期刊版权失效，
  **商标权与角色形象权仍在 DC 手里**
- COMICS 数据集页面本身未声明任何许可，严格说并未授权给你

## 3. 推荐的仓库内容

把「图片」换成「生成图片的脚本」，学习价值几乎不损失，版权风险归零：

| 上传 | 不上传 |
|---|---|
| ✅ 模型权重（`release/models/`，4 个模型 × .pt/.onnx） | ❌ `datasets/comic_style/raw/` |
| ✅ `scripts/` 全部脚本 | ❌ `datasets/comic_style/{cover,panel,*_gray}/` 图片 |
| ✅ `release/manifest.csv`（来源、作品分组、尺寸，不含图片） | ❌ `datasets/comic_style/_inspect/` 抽样图 |
| ✅ 评估结果、混淆矩阵、指标 JSON | ❌ 错判样本的原图 |
| ✅ 错判样本的**文字描述**（如「《极品医圣》被判成日漫」） | |

这样仓库变成一份**完整可复现的「漫画画风三分类数据集构建方案」**——
别人 clone 下来跑一遍脚本就能得到同样的数据集，
而且对方需要自己去 Kaggle 接受各数据集的条款，责任边界清晰。

## 4. 上传前检查清单

```bash
git init
git add -A
git status --short          # 确认没有 datasets/、.kaggle/、scratch/ 混入
git diff --cached --stat    # 确认总大小（本项目约 38 MB）
```

检查是否有凭据泄漏：搜索 Kaggle token 的前缀（`KGAT_` 去掉下划线写成通配形式，
避免本文档被 GitHub 的推送保护误判），命中即说明有明文 token 被误提交。

- [ ] `datasets/` 未被提交
- [ ] 无 Kaggle token / 其他凭据
- [ ] `LICENSE`（AGPL-3.0）在根目录
- [ ] `README.md` 写明模型受 AGPL-3.0 约束
- [ ] 无单文件超过 50 MB（本项目最大 5.9 MB，无需 Git LFS）
