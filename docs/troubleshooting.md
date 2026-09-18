# 工程踩坑记录

复现本项目时最容易踩的 9 个坑，全部是实测踩过并修掉的。
前 4 个会在**不报错的情况下**给出错误结果，最危险。

## 1. Windows 非 ASCII 路径：`cv2.imread` 静默失败

工作目录含中文时，`cv2.imread` / `cv2.imwrite` 直接返回 `None` 或写入失败，
**不抛异常**。第一版抽样脚本因此得到全空的对比图。

```python
# 错误
img = cv2.imread(str(path))
cv2.imwrite(str(out), img)

# 正确
buf = np.fromfile(str(path), dtype=np.uint8)
img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
buf.tofile(str(out))
```

PIL 的 `Image.open` 不受影响，所以读图可以直接用 PIL。

## 2. 文件名冲突：同一 stem 多扩展名互相覆盖

`kaustubhrastogi17/images/classic/` 下存在 `1 (1).jpeg`、`1 (1).jpg`、`1 (1).png`
**三个不同内容的文件**。清洗脚本若只用 `stem` 作为输出名，三者会映射到同一个
`xxx.jpg` 互相覆盖——**静默丢失 16 张图，且 manifest 虚报数量**
（磁盘实际 254 个文件，manifest 记录 270 条）。

修法：把原始扩展名纳入输出名。

```python
out_name = f"{tag}__{group}__{p.name}"   # 用完整文件名，不用 stem
```

## 3. 路径长度超 Windows 260 字符上限

部分作品名很长，`tag + 作品名 + 原名` 拼接后超过 260 字符，
`tofile` 报 `OSError: [Errno 22] Invalid argument`。

修法：截断 + 短哈希保唯一。

```python
if len(safe) > 110:
    digest = hashlib.md5(out_name.encode("utf-8")).hexdigest()[:10]
    safe = f"{safe[:100]}__{digest}"
```

## 4. 单作品主导评估：影响结论最大的一个坑

按图片数贪心划分 train/val/test 时，COMICS 里页数最多的 956 号书
一本书就占了欧美测试集的 **79%**，整体准确率退化成「这一本书的得分」，
panel 准确率只有 0.764。

改成 **val/test 优先用小作品填满配额**后，测试集覆盖 65 个作品组、
最大组占比降到 9.6%，准确率回到 0.9693。

详见 [dataset.md 第 4 节](dataset.md#4-划分设计本项目最大的一个坑)。

## 5. 生成派生数据集前必须清空目标目录

源划分变了以后新旧文件名不一致，直接写入会把两套划分混在一个目录里。
曾导致灰度数据集 `train` 有 585 张（源划分只有 504 张），
数量对不上且极难排查。

修法：脚本开头 `shutil.rmtree(dst)`。

## 6. 漫画切格：只用「白色装订线」判据会漏掉一半格子

**现象**：每页只能切出 1.4 格（日漫一页通常 4–6 格）。

**原因**：大量相邻格子仅靠一条**黑色边框线**分隔，没有白色留白。
黑线属于「高墨迹」带，会被「低墨迹 = 可切」的判据排除。

**修法**：双判据——留白装订线（墨迹占比 < 1.2%）**或**
细长贯穿黑线（墨迹占比 > 85% 且厚度 ≤ 9px）都是切点。
修后升到 4.3 格/页。

可视化验证：

```bash
python scripts/04_panelize.py --debug 8   # 输出画了切分框的对比图
```

## 7. 单成员 gzip 无法断点续传

COMICS 的 `raw_panel_images.tar.gz`（61 GiB）是**单个 gzip 成员**，
网络中断后无法从中间字节恢复解压——每次重试都得从流开头重读。

我在多个字节偏移处采样探测 gzip 成员头（`1f 8b 08`），命中 0 次，确认了这一点。

**可用的缓解方案**：重试时从流开头读，但**跳过已保存的文件**，
进度不会丢失，代价只是重下前缀字节。`scripts/01b_fetch_comics.py` 就是这么做的：

```python
for attempt in range(max_retries):
    try:
        _stream_once(target)      # 已存在的文件跳过不写
    except Exception:
        time.sleep(...)           # 从流开头重读，进度保留
```

本项目靠这个机制在网络断续的情况下从 21 本书扩到了 60 本。

## 8. 网络归因要实测，别急着怪服务端

早期把 COMICS 抽取失败归因于「服务端不稳定」并写进了文档。
后来实测发现是**本地网络整体掉线**——同期 HuggingFace 镜像速度也降到 839 B/s。

**做法**：下载失败时先做对照测速，再判断问题在哪一侧。

## 9. ONNX 导出的两个坑（封装模型必踩）

### 9.1 Ultralytics 导出的 ONNX 已经含 softmax

分类模型的计算图里**已包含 softmax**，输出就是概率。
若再套一层，分布会被压平——`[1,0,0]` 变成 `[0.576, 0.212, 0.212]`。
**类别看起来还对，但置信度全错**，很难发现。

修法：按「是否已是合法概率分布」自动判别。

```python
if x.min() >= 0.0 and abs(x.sum() - 1.0) < 1e-3:
    return x                    # 已是概率
return softmax(x)               # 是 logits
```

### 9.2 预处理规格和中心裁剪取整

必须与 `ultralytics.data.augment.classify_transforms(size=224)` 完全一致：

```
Resize(224, BILINEAR)   # 短边缩到 224，不是 256！
CenterCrop(224)         # 偏移 = int(round((边长-224)/2.0))，不是整除
ToTensor()              # 除以 255
Normalize((0,0,0),(1,1,1))   # 等于不做归一化
```

两个具体问题：

- **不是 ImageNet 那套「缩到 256 再裁 224」**。Ultralytics 的 `crop_fraction`
  参数已废弃，现在直接缩到 224。按老惯例写会让同一张图给出完全不同的置信度分布。
- **中心裁剪要用 `round` 而非整除**。当 `边长-224` 为奇数时（例如 291-224=67），
  `round(33.5)=34` 而 `67//2=33`，裁切位置差 **1 个像素**，
  在小缩略图上足以让置信度偏移 0.3。

修正后 cover 的 ONNX/.pt 预测一致率从 **97.4% 升到 100%**。

### 9.3 残余差异

PIL 与 torchvision 在大幅下采样时的抗锯齿核不同，
导致 panel 有 1/228 张在决策边界上翻转（`.pt` 判 western、`.onnx` 判 japanese）。
需要逐位复现请用 `.pt` 路径。详见 [release/MODEL_CARD.md](../release/MODEL_CARD.md) 第 4 节。
