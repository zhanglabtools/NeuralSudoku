代码与复现说明

本目录为论文配套代码。以下命令均在解压后的附件根目录运行，python 指向已配置依赖的 Python 3.10 或以上环境。数据位于 data/，权重位于 checkpoints/、results/A07/、results/A08/ 和 historical_results/。新的运行结果请保存至 reproduction/，避免覆盖提交时的实验记录。

1. 环境与代码改动

核心依赖：numpy、pandas、torch，列于 code/requirements.txt。PyTorch 应选择与实际 CPU/CUDA 环境兼容的构建。本附件未锁定一个未经完整测试的跨平台环境；整理时完成语法、清单完整性和 NumPy 边界检查，未重新执行 GPU 训练或 SATNet 原生扩展编译。

可选依赖列于 code/requirements_optional.txt：pynauty 用于全对称审计，python-sat 用于 SAT/ABL 基线，psutil 用于计时，adam-atan2、einops、pydantic、tqdm 用于 HRM 适配。全对称审计使用 multiprocessing 的 fork，CPU 计时脚本使用 resource 等接口，建议在 Linux 环境运行这些入口。全量唯一解/对称审计和神经训练可能耗时较长，小规模检查不能代替论文全量结果。

source_snapshot/ 的训练、模型、候选选择和统计逻辑沿用所提供代码；仅将历史机器上的路径字符串改为附件相对路径。source_manifest.json 记录输入与整理后文件哈希。对这些修改逐文件比较 AST，确认字符串常量之外的结构不变。active_batch_reference.py 仅更新文件说明，函数逻辑不变。portable_*.py 为新增路径适配入口。

portable_run.py 在附件根目录执行指定脚本，因此传给被调用脚本的相对路径均以附件根目录为起点。portable_eval.py、portable_train.py 的显式输入路径按调用时的工作目录解析。推荐按本说明从附件根目录运行。

2. 快速核查与 192 题统计

python code/verify_code.py
python code/portable_eval.py --help
python code/portable_train.py --help
python code/portable_run.py --help

192 题配对消融数据与统计脚本在 ../ablation_192/（相对于本 code/ 目录），训练重复记录在 ../training_repeats/。从附件根目录运行：
python ablation_192/recompute_stratified_bootstrap.py --help

应沿用 ablation_192 内脚本的固定抽样与 bootstrap 协议。source_snapshot 中其他全量审计或随机预算入口属于各自实验，不可代替 192 题正式统计。

3. 使用已提供的独立系统权重

先运行 seed0 的两题检查：
python code/portable_eval.py --backbone results/A07/joint_seed0/best.pt --reflection results/A08/seed0/best.pt --cache data/cache_full_3m.npz --split test --global-ids results/clean_evaluation/test_global_ids.npy --output reproduction/c5_seed0_smoke --device cpu --batch-size 1 --limit 2

全量清洁测试删除 --limit 2，并按硬件设置 --device cuda --batch-size 32；seed1、seed2 分别选择对应编号的 A07 backbone 与 A08 reflection。只评估 D4 时加 --rating-min-exclusive 4。每次使用新的输出目录。

portable_eval.py 显式加载 backbone，不使用权重内部保留的旧路径；若反思权重包含 backbone 参数，还会检查二者完全一致。该入口调用原始 a08_seed_pipeline.evaluate_indices 和 full 条件：按 parent、cycle、最低合法 slot 的顺序选第一个合法候选，真值仅用于选定后的 Exact 评分。输出 summary.json 和 details.npz；未解出的 selected_grid 全为 0，须结合 first_valid 使用。指定 --limit 后 summary 会明确标识是否覆盖完整选择。测试集必须传入清洁 global IDs，以免误用未经排除的完整原始 split。

旧参考系统可同样将 --backbone、--reflection 显式指向 checkpoints/ 与 historical_results/ 中相应权重。不要混用不同系统的 backbone 与 reflection。

4. 训练 A07 与 A08

先仅查看将执行的命令：
python code/portable_train.py e2 --architecture joint --seed 0 --cache data/cache_full_3m.npz --validation-ids results/A07/validation_selection/global_ids.npy --output reproduction/A07/joint_seed0 --device cuda --dry-run

确认环境就绪后删除 --dry-run 执行。architecture 支持 graph、hypergraph、joint，固定对应 (D,msg_hidden)=(208,384)、(128,432)、(128,256)，训练 50000 次更新，batch=64，train_T=32，eval_T=64，种子为 0、1、2。--max-new-updates 2 可按原 50000 更新协议运行两次后安全暂停，退出码 75 表示尚未完成；继续时加 --resume。短运行不是完整训练结果。

在匹配 seed 的 A07 完成后训练独立 C5：
python code/portable_train.py a08 --seed 0 --stage all --backbone results/A07/joint_seed0/best.pt --cache data/cache_full_3m.npz --clean-test-ids results/clean_evaluation/test_global_ids.npy --output reproduction/A08/seed0 --device cuda --dry-run

删除 --dry-run 执行；--stage 可选 mine、train、evaluate、all。需要 backbone 同目录的 finished.json 通过原始完备性检查。默认反思训练预算 10000，验证选择 best；实际所选 best 步数以随附记录为准。A07 三个 joint best 均对应 50000 步，与相应 final 权重一致。已提供 A08 结果目录中的 protocol.json 是运行时溯源，不应当作移动后可直接续训的工作目录。复现请从新目录开始；现成权重推理使用上一节 portable_eval.py。

5. 其他实验入口

可通过以下方式查看任一入口参数（脚本名不含 .py）：
python code/portable_run.py audit_dataset -- --help

与实验的对应关系：
- 唯一解及分割审计：audit_dataset；内部依赖 sudoku_dlx_solver。
- 全对称等价审计：audit_full_symmetry；依赖 pynauty，支持 --self-test。
- CPU DLX/SAT：benchmark_exact_solvers；统一计时：benchmark_unified。
- RRN 步数/候选预算：eval_rrn_candidate_ceiling；该入口包含 oracle Exact 天花板，不能把其 oracle 指标当作部署选择成绩。
- 随机预算匹配：eval_c5_random_budget；完整清洁测试：a04_all_test；同目录 a04_budget_plan.json 为预算配置。
- 双解：eval_hybrid_c5_two_solution、eval_hybrid_c5_dual_equivariance、audit_dual_solution_dataset，及同目录依赖。
- SATNet、ABL、HRM 历史主表：train_official_satnet_kaggle、train_abl_refl_paper_kaggle、train_hrm_official_kaggle；适配模型类在 reproduce_satnet_abl_hrm。
- 独立基线训练：a08_train_baseline；HRM 执行适配：hrm_train_execution_v2、hrm_execution_v2。训练配置采用附件对应实验记录，不将命令行默认值视为全部正式配置。

对会调用旧 C5 加载器的入口，在脚本名之前添加 --backbone 可覆盖序列化的旧路径，例如：
python code/portable_run.py --backbone results/A07/joint_seed0/best.pt a04_all_test -- --help

旧入口中部分参数名使用下划线，部分使用连字符，请以各入口 --help 为准。portable_run.py 不会自动替换实验配置或统计口径。可在脚本名之前加 --dry-run 仅检查入口和参数，不导入模型。

6. 第三方实现与许可

source_snapshot/external/SATNet-master/ 保留 SATNet Python 包、C++/CUDA 扩展源码、setup.py、README 和 MIT LICENSE。原生 SATNet 需在兼容的编译环境中构建，例如从附件根执行：
python -m pip install --no-build-isolation ./code/source_snapshot/external/SATNet-master
本次整理未执行该编译，也未以 satnet_lite 替代原生 SATNet；reproduce_satnet_abl_hrm 中的 satnet_lite 是不同适配模型。

source_snapshot/external/HRM/ 保留模型源码、配置、依赖说明和 Apache-2.0 LICENSE。其中 flash_attn.py 是源材料已包含的 PyTorch scaled_dot_product_attention 兼容实现，不代表安装了原生 FlashAttention。若复现完整上游 HRM 工程，可另参考该目录 requirements.txt；本文适配入口所需依赖较少。

无需复制特定机器上的 Python 环境或二进制扩展。python-sat、pynauty 等包应在目标环境安装；附件保留代码和实验记录，原生库是否成功构建需由实际运行环境验证。
