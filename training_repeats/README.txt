独立训练逐次统计

training_repeats.csv由对应逐题NPZ复算，包含三结构各3次及完整方法3次训练的整数正确数、分母和求解率。validation_summary.json保留均值、样本标准差及检查结果。

从附件根目录运行：
python scripts/verify_reported_results.py --root . --output-dir recomputed/training_repeats

三结构用50000次更新末次权重；完整方法采用各自验证选择的权重。超图seed1零结果完整计入。统计种子、推理噪声种子与独立训练种子不混合。
