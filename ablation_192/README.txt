192题机制消融资料

本目录对应论文图4及附录C.3表C.2，共64道验证题和128道测试题，均满足rating>4。

一、文件与样本标识
sample_ids.csv列出192道固定样本，字段为split、split_position和stratum。split为val或test；split_position表示该数据划分内的位置；stratum表示结局分层。每道题由(split, split_position)共同标识。

val/summary.json和val/c5_ablation_summary.csv对应64道验证题；test/下的同名文件对应128道测试题。CSV每行记录一道题在一个消融条件下的结果，summary.json保留样本分层、推理设置和各条件的原始统计。样本清单保留原CSV中full条件的顺序，先列val，再列test。

recompute_stratified_bootstrap.py读取上述CSV并计算分层配对统计。stratified_bootstrap_192.json和stratified_bootstrap_192.csv给出本图的复算结果；原始CSV和JSON保持不变。

二、样本分层
anchor_success：初始64步传播成功。
c5_early：第1—2个反思周期成功。
c5_late：第3—5个反思周期成功。
c5_failure：5个反思周期后仍失败。
验证划分每层16题，测试划分每层32题；合并后共四个结局层，每层48题。这组样本按完整模型的求解结局分层选取，用于机制分析。

三、图4对应条件和计数
full：完整方法，144/192。
no_recovery：无关系再传播，49/192。
recovery_only：无全部学习型写回，61/192。
no_cell：无空格状态直接写回，66/192。

统计时合并val与test的CSV，以first_valid_cycle大于或等于0计为预算内至少生成一次合法候选，再按ablation字段分别求和。final_any_valid和final_any_exact记录末周期状态。

四、分层配对差值与区间
按(split, split_position)对齐完整方法与消融条件。令每道题的完整方法和消融结果分别为f_i和a_i，均取first_valid_cycle大于或等于0的指示值。差值为100×sum(a_i-f_i)/192，单位为百分点。

将验证题与测试题按上述四种结局合并为四层，每层48题。每次在各层内等概率、有放回抽取48个题目索引，同题在各条件下的结果同时进入该次抽样。汇总四层后计算生成率差，共重采样20000次；以差值分布的2.5%与97.5%分位数形成95%区间，统计随机种子为20260801。

三项消融差值依次为−49.48、−43.23和−40.62个百分点，对应区间为[−50.00,−48.44]、[−46.35,−40.10]和[−44.27,−36.98]。逐题原始计数分别为失去95题、新增0题；失去84题、新增1题；失去79题、新增1题。最后一项未舍入差值为−40.625，脚本采用Python两位小数格式输出−40.62。

五、复算方法
安装Python 3和NumPy，在附件根目录运行：
python ablation_192/recompute_stratified_bootstrap.py --data-dir ablation_192 --output-dir ablation_192/recomputed

脚本默认使用20000次重采样和种子20260801，输出JSON与CSV。代码明确记录样本排序、四层顺序及分位数计算方式，不修改原始数据。

六、全局题号与题面
sample_global_ids.csv 将原始划分内位置映射到 data/cache_full_3m.npz 的从0起算全局行号；sample_puzzles.npz 包含同序题面、参考解、划分和难度。原始抽样基础种子日志不完整，复核使用固定清单。原始周期记录位于 historical_results/hybrid_c5_explainability_pilot_20260731_full_v2/ 与 historical_results/hybrid_c5_explainability_confirm_test_20260731_v1/。
