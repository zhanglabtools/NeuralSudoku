# NeuralSudoku

《融合图超图协同循环网络与状态反思机制的高效数独求解算法》配套代码、实验记录与补充材料。

## 从这里开始

| 你的目标 | 阅读或运行入口 | 所需材料 |
|---|---|---|
| 查找论文某个图表的数据 | [图表与附件索引](EVIDENCE_MAP.md) | 汇总可直接浏览；逐题大文件见完整附件 |
| 了解各目录和实验编号 | [目录指南](DIRECTORY_GUIDE.md) | 仓库即可 |
| 无需训练，复算图 4 与表 C.2 | [192 题说明](ablation_192/README.txt)、[配对统计脚本](ablation_192/recompute_stratified_bootstrap.py) | 仓库即可；需要 NumPy |
| 无需训练，核对三次完整模型与九次结构训练 | [逐次统计](training_repeats/training_repeats.csv)、[复核脚本](scripts/verify_reported_results.py) | 完整附件；需要 NumPy |
| 用已有权重运行模型，或重新训练 | [环境与运行说明](code/README_CODE.txt)、[评估入口](code/portable_eval.py)、[训练入口](code/portable_train.py) | 完整附件及相应运行环境 |

## 下载代码或完整附件

- **代码仓库**：克隆本仓库，或使用 GitHub 的 “Source code” 下载，获取源码、说明、配置、小型 CSV/JSON 记录及统计脚本。仓库不包含 `.pt`、`.npz` 和其他大于 5 MiB 的文件。
- **完整附件**：[v1.0.0 Release](https://github.com/zhanglabtools/NeuralSudoku/releases/tag/v1.0.0) · [下载 ZIP](https://github.com/zhanglabtools/NeuralSudoku/releases/download/v1.0.0/NeuralSudoku-supplement-v1.0.0.zip)。包含完整数据缓存、主要模型权重与保存的逐题结果；解压后进入 `NeuralSudoku/` 使用。无需 Git LFS。
- **下载校验**：[ZIP 的 SHA256](https://github.com/zhanglabtools/NeuralSudoku/releases/download/v1.0.0/NeuralSudoku-supplement-v1.0.0.sha256) · [完整附件逐文件清单](https://github.com/zhanglabtools/NeuralSudoku/releases/download/v1.0.0/NeuralSudoku-supplement-v1.0.0.manifest.sha256)。GitHub 自动生成的源码包不包含完整附件的数据和权重。

## 目录结构

以下为完整附件的结构；标注“完整附件”的内容不随仓库克隆提供。目录的进一步说明见[目录指南](DIRECTORY_GUIDE.md)。

```text
NeuralSudoku/
├── README.md                 阅读与运行入口
├── DIRECTORY_GUIDE.md        实验编号、目录用途和数据范围
├── EVIDENCE_MAP.md           论文图表 → 对应记录
├── code/                    模型源码、portable 入口、依赖与第三方实现
├── ablation_192/            固定样本、逐题 CSV、bootstrap 脚本
├── training_repeats/        A07/A08 的整数计数、均值与样本标准差
├── results/                 审计、预算、CPU、结构比较与完整训练结果
├── historical_results/      固定参考模型的机制、双解与轨迹记录
├── data/                    300 万题缓存与失败池（完整附件）
├── checkpoints/             固定参考模型与基线权重（完整附件）
├── logs/                    保存的历史训练日志
├── scripts/                 无需模型训练的结果复核脚本
├── SOURCE_FILES.json        实验文件来源与提交时哈希
├── MANIFEST.sha256          当前分发内容的逐文件校验
└── verify_files.py          文件校验入口
```

直接浏览：[代码](code/) · [192 题消融](ablation_192/) · [独立训练统计](training_repeats/) · [实验结果](results/) · [历史机制记录](historical_results/) · [训练日志](logs/)。权重和大型结果保留在完整附件的相同目录中。

## 无需训练的复核

在克隆目录或完整附件解压后的 `NeuralSudoku/` 中，先校验所分发的文件，再复算 192 题结果：

```text
python verify_files.py
python ablation_192/recompute_stratified_bootstrap.py --data-dir ablation_192 --output-dir recomputed/ablation_192
```

核对 A07 的九次结构训练和 A08 的三次完整训练，需要完整附件中的 NPZ 逐题结果：

```text
python scripts/verify_reported_results.py --root . --output-dir recomputed/training_repeats
```

统计脚本使用 NumPy，另存复算结果，不覆盖已有实验记录。它们核查保存的输出，不执行模型推理或训练。文件校验仅需要 Python 标准库；[当前清单](MANIFEST.sha256)不包含自身及之后生成的输出。

## 运行模型

下载完整附件后，按[环境与运行说明](code/README_CODE.txt)配置 Python 与相应的 PyTorch 构建。该说明先提供 seed 0 的两题 CPU 检查，再给出完整清洁测试和训练命令。训练、原生 SATNet 扩展、全量审计各有依赖与计算要求；部分审计和计时入口使用 Linux 接口。

评估时显式配对同一种子的 A07 联合主干与 A08 反思器，并使用 `results/clean_evaluation/test_global_ids.npy`。新的运行输出保存至 `reproduction/`，与保存的论文结果分开。

## 数据与结果范围

数据来自 Radcliffe 的 “3 million Sudoku puzzles with ratings” 第 2 版。完整附件保存 300 万题缓存；训练、验证、原始测试分别为 2399047、300250、300703 题。清洁测试排除一条与训练集对称等价的记录后为 300702 题，其中困难题 6543 题。SATNet、ABL-Refl、HRM 的历史单次成绩采用原始测试分母，不能与清洁测试混作同一运行。

独立训练、固定权重的重复计时和噪声推理是不同统计单位。192 题机制样本以固定清单复核，历史抽样缺完整基础种子日志；96 题双解扰动未保存每条轨迹的完整候选盘面。具体模型身份和适用范围见[图表索引](EVIDENCE_MAP.md)及[目录指南](DIRECTORY_GUIDE.md)。论文正文、LaTeX 工程与审稿回复不包含在本仓库和完整附件中。

## 第三方实现

随附的 [SATNet](code/source_snapshot/external/SATNet-master/) 使用 [MIT 许可](code/source_snapshot/external/SATNet-master/LICENSE)，[HRM](code/source_snapshot/external/HRM/) 使用 [Apache-2.0 许可](code/source_snapshot/external/HRM/LICENSE)。适配边界、依赖与原生扩展构建说明见[代码说明](code/README_CODE.txt)。
