# 更新日志

## 2026-06-10 16:25 - 支持同问题并行运行 MCTS-AHD 与 MCGS 策略

- 更新时间：2026-06-10 16:25 CST
- 更新记录：修复同一问题多进程运行时共享 `gpt.py` 导致候选评估互相覆盖的问题，并补充远程部署与四路并行运行说明。
- 更新内容：
  - `problem_adapter.py`: 每个候选评估改为在当前 Hydra run 的 `evaluations/` 子目录生成独立 `gpt.py`，通过 bootstrap 优先加载隔离模块。
  - `main.py`: 最终验证改为使用 run 目录内的 `final_validation/gpt.py`，并按问题类型选择 `eval.py` 或 `eval_black_box.py`。
  - `README.md`: 新增 GitHub 远程部署步骤、OpenAI-compatible 配置示例，以及 MCTS-AHD 加 MCGS 三种 UCT 策略的并行运行命令。
- 验证情况：已运行 `python -m compileall main.py problem_adapter.py source`、`conda run -n mcts-ahd python main.py --help`、`conda run -n mcts-ahd python main.py --cfg job algorithm=mcgs dual_lineage_backup=true uct_value_mode=dual_max max_fe=1`、`git diff --check`，并用隔离 `gpt.py` 完成 `tsp_constructive/eval.py` 训练集 bootstrap 烟测。
- Git 提交：fix(parallel): 隔离候选评估文件以支持并行运行

## 2026-06-10 14:46 - 迁移 CO-Bench MCGS 到原 MCTS-AHD 仓库

- 更新时间：2026-06-10 14:46 CST
- 更新记录：在原 MCTS-AHD 仓库中新增 MCGS 分支，实现与 CO-Bench MCGS 一致的双谱系 S1、父系 credit 和 UCT value mode。
- 更新内容：
  - `source/mcgs.py`: 新增 MCGS 节点、谱系字段、S1 上下文、双谱系回传和 UCT 计算。
  - `source/mcgs_algorithm.py`: 新增原仓库数据/评估路径下的 MCGS 主流程，并输出 `mcgs_graph_latest.json`。
  - `ahd_adapter.py` 与配置文件: 增加 `algorithm=mcgs` 分支和 MCGS/Hydra 参数。
  - `source/evolution.py` 与 `source/evolution_interface.py`: 支持结构化 S1 prompt 和父代节点 ID 追踪。
  - `utils/utils.py` 与 `cfg/llm_client/openai.yaml`: 支持 OpenAI-compatible/Qwen `base_url/api_key/model` 覆盖。
- 验证情况：已运行 `conda run -n mcts-ahd python -m compileall source utils problem_adapter.py ahd_adapter.py main.py`、双谱系构造断言测试、`git diff --check`、`python main.py --help`、`python main.py --cfg job algorithm=mcgs ...`。
- Git 提交：feat(mcgs): 迁移双谱系 MCGS 到原 MCTS-AHD 仓库
