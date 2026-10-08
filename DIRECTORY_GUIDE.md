# 目录指南

本指南解释保存记录的用途。按论文图表查找时，使用[图表与附件索引](EVIDENCE_MAP.md)；运行命令见[代码说明](code/README_CODE.txt)。

## 仓库与完整附件

下方可点击链接均指向仓库中实际保留的目录或小型记录。模型权重、NPZ 和大于 5 MiB 的文件只在[完整附件](https://github.com/zhanglabtools/NeuralSudoku/releases/download/v1.0.0/NeuralSudoku-supplement-v1.0.0.zip)中，解压后保持相同目录结构。仓库另附 `FULL_PACKAGE_MANIFEST.sha256`，其内容与 ZIP 内的 `MANIFEST.sha256` 一致。

## 代码、数据和复核

| 位置 | 用途与入口 |
|---|---|
| [code/](code/) | [运行说明](code/README_CODE.txt)、[核心依赖](code/requirements.txt)、[可选依赖](code/requirements_optional.txt)；`source_snapshot/` 保存模型和各实验实现，`portable_*.py` 提供可移植入口 |
| [ablation_192/](ablation_192/) | [192 题固定清单](ablation_192/sample_ids.csv)、[全局行号映射](ablation_192/sample_global_ids.csv)和逐题 CSV；`sample_puzzles.npz` 题面及参考解在完整附件中 |
| [training_repeats/](training_repeats/) | [整数计数与逐次成绩](training_repeats/training_repeats.csv)、[统计复核结果](training_repeats/validation_summary.json)；A07 九次结构训练与 A08 三次完整训练 |
| [scripts/verify_reported_results.py](scripts/verify_reported_results.py) | 从完整附件的逐题 NPZ 复算计数、均值和样本标准差；不训练模型 |
| `data/`（完整附件） | `cache_full_3m.npz` 为完整数据缓存，另保存固定参考系统的失败池 |
| `checkpoints/`（完整附件） | 固定参考模型及 RRN、SATNet、ABL-Refl、HRM 对应权重；A07/A08 独立训练权重位于各自结果目录 |
| [logs/](logs/) | 保存的 SATNet、ABL-Refl、HRM 历史训练日志 |
| [SOURCE_FILES.json](SOURCE_FILES.json)、[源码来源清单](code/source_manifest.json) | 文件来源、实验时与提交时哈希；当前分发文件的校验值见 [MANIFEST.sha256](MANIFEST.sha256) |

完整缓存包含 `puzzles`、`solutions`、`clues`、`ratings`、`splits` 五个数组。题面与参考解形状为 `(3000000, 81)`，题面 0 表示空格；`splits` 的 0、1、2 表示训练、验证、原始测试。全局行号从 0 开始，清洁测试排除第 240822 行。192 题由 64 道验证题和 128 道测试题构成，四个结局层各 48 题；统计种子为 20260801，层内配对重采样 20000 次。

## results 中的实验编号

| 目录与可读入口 | 保存内容 | 对应用途 |
|---|---|---|
| [dataset_audit](results/dataset_audit/summary.json) | 划分与唯一解审计摘要；`test_uniqueness.csv` 全量审计在完整附件中 | 表 1、附录 C.2 |
| [full_symmetry](results/full_symmetry/summary.json) | 全数独对称群的审计摘要、类别与重复题见证 | 数据泄漏核查 |
| [clean_evaluation](results/clean_evaluation/) | 清洁测试全局索引及其说明 | 300702 题测试与 6543 题困难子集 |
| [A02_d4_repeat3](results/A02_d4_repeat3/shared_repeat0_pair_recomputed.json) | 动态／冻结线索状态的同题配对记录 | 表 6，固定参考系统 |
| [A04](results/A04/clean_all_test/condition_summary.csv) | 五组反思预算与随机扰动的条件统计 | 表 D.3；重复为噪声推理，不是独立训练 |
| [A04_all](results/A04_all/rrn_budget/) | RRN 长传播步数下的清洁测试统计 | 表 3 |
| [A05_summary](results/A05_summary/summary.csv)、[逐次计时](results/A05_summary/repeat_values.csv) | 三种方法、两种样本集和两种批量，共 12 组配置、36 次计时 | 表 4、表 D.1—D.2 |
| [A05_timing_samples](results/A05_timing_samples/) 与 `A05_CPU_*` | CPU 计时的固定样本身份及各条件原始运行记录 | 上述计时汇总的来源 |
| [A06/rrn128_reference](results/A06/rrn128_reference/) | 128 维 RRN 参考配置的预算与测试记录 | 固定参考基线记录 |
| [A07](results/A07/)、[逐次汇总](results/A07/aggregate/per_seed.csv) | 仅格图、仅超图、联合结构 × 3 个种子 | 表 5、表 E.1—E.2；使用 50000 次更新的末次权重 |
| [A08](results/A08/)、[三种子汇总](results/A08/three_seed_summary.json) | 完整 NeuralSudoku 的三次独立训练 | 表 2、表 C.1；与各自 A07 联合主干配对 |
| [historical_baselines](results/historical_baselines/README.md) | 历史单次基线的结果及统计口径 | 表 2；不含全部基线逐题预测盘面 |
| [reflection_cycles](results/reflection_cycles/README.md) | 固定参考模型各反思周期的累计求解数 | 图 3 |

A07 使用 `matched_parameters_fixed_exposure_final` 协议，包含仅超图 seed 1 的零结果。A08 所用三个联合主干的验证最佳点均为第 50000 次更新，其 `best.pt` 与 `final.pt` 分别同哈希；反思器验证选点分别为 10000、10000、9500 次更新。A07/A08 的权重与逐题 NPZ 均在完整附件中。

## historical_results 中的机制记录

这些目录保留特定历史参考系统的实验身份，不应替代 A08 的三次独立训练结果。

| 目录 | 用途 |
|---|---|
| [hybrid_c5_explainability_pilot_20260731_full_v2](historical_results/hybrid_c5_explainability_pilot_20260731_full_v2/) | 192 题机制分析的验证部分来源；大型周期轨迹 CSV 在完整附件中 |
| [hybrid_c5_explainability_confirm_test_20260731_v1](historical_results/hybrid_c5_explainability_confirm_test_20260731_v1/) | 192 题机制分析的测试部分来源；正式配对区间以 `ablation_192/` 为入口 |
| [hybrid_c5_two_solution_20260801](historical_results/hybrid_c5_two_solution_20260801/) | 4096 题双解数据摘要；对应 NPZ 在完整附件中 |
| [hybrid_c5_dual_explainability_20260806](historical_results/hybrid_c5_dual_explainability_20260806/) | 双解数据的逐实例审计与证书；大型 JSONL 在完整附件中 |
| [hybrid_multiseed_two_solution_20260802](historical_results/hybrid_multiseed_two_solution_20260802/eval/) | 三个历史传播模型的双解分支统计，对应表 7 |
| [hybrid_interpretability_reframed_20260804](historical_results/hybrid_interpretability_reframed_20260804/early_branch/) | 128 题概率轨迹，对应图 5(a) |
| [hybrid_hidden_attractor_20260802](historical_results/hybrid_hidden_attractor_20260802/pilot_joint/) | 96 题状态扰动的样本、条件和统计，对应图 5(b) |

固定参考系统的困难题正确数为 6351；A08 seed 0 的 6356 属于另一训练。双解扰动的 95/96 是每题 1280 次尝试后至少转入另一合法解的题目覆盖率；完整附件也未保存全部轨迹的逐候选盘面。更多边界见[图表索引](EVIDENCE_MAP.md)。
