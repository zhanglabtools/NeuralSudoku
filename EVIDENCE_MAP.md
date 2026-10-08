# 图表与附件索引

本索引连接论文图表与对应记录，路径均相对于附件根目录。训练、推理及统计复算方法见 [README](README.md) 和[代码说明](code/README_CODE.txt)，目录用途见[目录指南](DIRECTORY_GUIDE.md)。链接指向仓库中可浏览的记录；NPZ、权重和大型记录需下载[完整附件](https://github.com/zhanglabtools/NeuralSudoku/releases/download/v1.0.0/NeuralSudoku-supplement-v1.0.0.zip)后在同路径读取。

| 论文位置 | 附件材料 |
|---|---|
| 表1、附录C.2：划分与泄漏审计 | [划分审计](results/dataset_audit/summary.json)、300703题唯一性记录 `results/dataset_audit/test_uniqueness.csv`（完整附件）、[对称审计](results/full_symmetry/summary.json) |
| 表2：历史单次基线 | [历史汇总及口径](results/historical_baselines/README.md) |
| 表2、表C.1：完整模型三次训练 | [A08汇总](results/A08/three_seed_summary.json)、[各次记录](results/A08/) |
| 表3：RRN长步数 | [全量测试记录](results/A04_all/rrn_budget/) |
| 表4、表D.1—D.2：CPU成本 | [计时汇总](results/A05_summary/summary.csv)、[逐次测量](results/A05_summary/repeat_values.csv) |
| 表5、表E.1—E.2：三结构消融 | [A07逐次统计](results/A07/aggregate/per_seed.csv)、[配置与测试记录](results/A07/) |
| 图3：反思周期累计收益 | [周期计数及说明](results/reflection_cycles/README.md) |
| 图4、表C.2：192题配对消融 | [样本与复算说明](ablation_192/README.txt)、[固定样本清单](ablation_192/sample_ids.csv) |
| 表6：动态/冻结线索状态 | [同题配对复算](results/A02_d4_repeat3/shared_repeat0_pair_recomputed.json) |
| 表7：三个历史传播模型的双解分支 | [逐模型结果](historical_results/hybrid_multiseed_two_solution_20260802/eval/) |
| 图5(a)：128题概率轨迹 | [轨迹记录](historical_results/hybrid_interpretability_reframed_20260804/early_branch/) |
| 图5(b)：96题状态扰动 | [样本与条件记录](historical_results/hybrid_hidden_attractor_20260802/pilot_joint/) |
| 表D.3：反思预算与随机扰动 | [条件统计](results/A04/clean_all_test/condition_summary.csv) |

## 统计范围

表2的 SATNet、ABL-Refl、HRM 是原测试集300703题上的单次训练结果；RRN与完整模型采用清洁测试集300702题，困难子集均为6543题。A08表示完整 NeuralSudoku 的三次独立训练，A07表示三结构各三次训练。A07聚合文件仅保留 `matched_parameters_fixed_exposure_final` 协议的9次训练，包含仅超图种子1的零结果。

表3、图3、表6及反思预算使用指定的固定参考模型，其困难题正确数6351；A08种子0的正确数6356属于另一次训练。CPU聚合仅保留RRN、联合主干及完整NeuralSudoku配置，共12组配置、36次计时；各配置的三次计时是固定模型重复测量。表D.3的三次重复是固定模型噪声推理，不是三次独立训练。

A07结构比较采用50000次更新后的 `final.pt`。A08实际引用各自联合主干 `best.pt`；三个主干的最佳点均为50000次更新，且与对应final文件的SHA256相同。A08反思器仍分别使用验证选定的10000、10000、9500次更新权重。同一主干在A07/A08中复用，不重复计算训练次数。

## 核查边界

192题按完整模型结局分层，是机制样本。附件以固定清单和保留统计支持区间复算；历史抽样缺完整基础种子日志，不承诺仅凭种子重新生成相同样本。

双解扰动95/96是每题1280次尝试后至少转入另一合法解的覆盖率。现有材料不含全部122880条轨迹的逐候选盘面，因此不声称已逐条重验所有候选或完成全部轨迹的精确重跑。

公开附件保留科学参数、数值、模型权重和数据数组；论文正文、图形排版文件与审稿回复不包含在内。
