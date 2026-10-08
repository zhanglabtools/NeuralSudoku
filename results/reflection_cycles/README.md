# 图3反思周期计数

[original_reflection_cycle_curve_20260813.csv](original_reflection_cycle_curve_20260813.csv)对应固定参考模型在6543道困难题上的累计计数：初始传播4405题，随后五个周期为4747、5579、6076、6274、6351题。

`solved`为截至该阶段已接受的合法解题数，`new_solved`为该阶段新增题数；第0阶段的新增数即初始传播正确数。`exact_percent`等于100×solved/total。该曲线使用一个固定模型，不是独立训练均值，也不是每个周期重新开始推理的成功率。

CSV未含机器路径，计数与百分比原样保留。固定参考模型与完整方法三次独立训练的区别见附件根目录的[EVIDENCE_MAP.md](../../EVIDENCE_MAP.md)。
