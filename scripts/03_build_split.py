"""把 clean/ 里的图按 Ultralytics 分类格式组装成 train/val/test。

关键点:
  * 按「作品」分层划分——同一部作品(或同一页漫画)的图不会同时出现在 train 和 val，
    否则模型可以靠记住同一部作品的画风而不是学会风格本身。
  * 类别不平衡时按 max-ratio 对多数类下采样，避免模型直接躺平预测多数类。

产出:
  datasets/comic_style/cover/{train,val,test}/{japanese,chinese,western}/
  datasets/comic_style/panel/{train,val,test}/{japanese,western}/
"""
import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets" / "comic_style"
CLEAN = DS / "clean"

CLASS_ORDER = ["japanese", "chinese", "western"]


def read_manifest():
    with (CLEAN / "manifest.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def balance(rows, max_ratio, rng):
    """按 max_ratio 对多数类下采样后再返回。"""
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["class"]].append(r)
    if not by_cls:
        return rows
    smallest = min(len(v) for v in by_cls.values())
    cap = int(smallest * max_ratio)
    out = []
    for cls, items in by_cls.items():
        if len(items) > cap:
            items = rng.sample(items, cap)
            print(f"    [balance] {cls}: 下采样到 {cap}")
        out += items
    return out


def cap_per_group(rows, max_per_group, rng):
    """限制单个作品分组的图片数。

    COMICS 里各本书的页数相差极大，不设上限时一本书就可能在 val/test 里占绝对多数，
    使评估结果退化成「这一本书的得分」。按组下采样可以让各作品对评估的贡献均衡。
    """
    if max_per_group <= 0:
        return rows, 0
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    out, capped = [], 0
    for g, items in groups.items():
        if len(items) > max_per_group:
            items = rng.sample(items, max_per_group)
            capped += 1
        out += items
    if capped:
        print(f"    [cap] {capped} 个作品组被截到 {max_per_group} 张")
    return out, capped


def split_by_group(rows, ratios, rng, min_groups=8):
    """按作品分组切分，保证同一组只落在一个集合里。

    关键设计：val/test 优先用**小组**去填满配额，而不是按图片数贪心。
    原因是 COMICS 里各本书页数悬殊，若只按图片数分配，
    val/test 会只分到 1-2 本书，评估就退化成「这两本书的得分」。
    从小往大挑可以让 10% 的测试配额覆盖尽可能多的作品，评估才有多样性。
    """
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)

    keys = list(groups.keys())
    rng.shuffle(keys)

    total = len(rows)
    tgt = {"train": total * ratios[0], "val": total * ratios[1], "test": total * ratios[2]}
    assigned = {"train": [], "val": [], "test": []}
    used = set()

    for split in ("test", "val"):
        # 从小到大挑组，避免一本书吃掉整个配额
        cand = sorted((k for k in keys if k not in used), key=lambda k: len(groups[k]))
        cur = 0
        for k in cand:
            if cur >= tgt[split] and len(assigned[split]) >= min_groups:
                break
            sz = len(groups[k])
            # 已满足最少组数后，不再接受会明显超出配额的组
            if cur + sz > tgt[split] * 1.3 and len(assigned[split]) >= min_groups:
                continue
            assigned[split] += groups[k]
            cur += sz
            used.add(k)

    for k in keys:
        if k not in used:
            assigned["train"] += groups[k]

    return assigned, len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", default="0.8,0.1,0.1")
    ap.add_argument("--max-ratio", type=float, default=2.0,
                    help="多数类相对少数类的最大倍数，超出则下采样")
    ap.add_argument("--max-per-group", type=int, default=120,
                    help="单个作品分组最多取多少张，防止一本书主导评估（0=不限）")
    ap.add_argument("--min-groups", type=int, default=8,
                    help="val/test 各至少要覆盖多少个作品分组")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", action="store_true", help="先删除已有输出目录")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ratios = tuple(float(x) for x in args.ratios.split(","))
    assert abs(sum(ratios) - 1.0) < 1e-6, "划分比例之和必须为 1"

    rows = read_manifest()
    print(f"manifest: {len(rows)} 张\n")

    if args.clean:
        for subset in ("cover", "panel"):
            d = DS / subset
            if d.exists():
                shutil.rmtree(d)
        print("已清空旧输出\n")

    summary = []
    for subset in ("cover", "panel"):
        sub_rows = [r for r in rows if r["subset"] == subset]
        if not sub_rows:
            continue
        print(f"=== {subset} ===")
        sub_rows, _ = cap_per_group(sub_rows, args.max_per_group, rng)
        sub_rows = balance(sub_rows, args.max_ratio, rng)

        # 按类别分别分组切分，保证每一类内部都是按作品划分
        per_split = defaultdict(list)
        n_groups = 0
        for cls in CLASS_ORDER:
            cls_rows = [r for r in sub_rows if r["class"] == cls]
            if not cls_rows:
                continue
            assigned, ng = split_by_group(cls_rows, ratios, rng, args.min_groups)
            n_groups += ng
            for split, items in assigned.items():
                per_split[split] += items
            counts = {k: len(v) for k, v in assigned.items()}
            print(f"  {cls:<10} groups={ng:<4} train/val/test = "
                  f"{counts['train']}/{counts['val']}/{counts['test']}")

        for split, items in per_split.items():
            for r in items:
                src = CLEAN / r["file"]
                dst = DS / subset / split / r["class"] / Path(r["file"]).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
            print(f"  -> {subset}/{split}: {len(items)} 张")

        print(f"  共 {n_groups} 个作品分组\n")
        summary.append((subset, per_split))

    # 统计
    print("=== 最终数据集 ===")
    for subset, per_split in summary:
        print(f"[{subset}]")
        for cls in CLASS_ORDER:
            line = []
            tot = 0
            for split in ("train", "val", "test"):
                n = sum(1 for r in per_split[split] if r["class"] == cls)
                tot += n
                line.append(f"{split}={n}")
            if tot:
                print(f"  {cls:<10} {'  '.join(line)}   合计={tot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
