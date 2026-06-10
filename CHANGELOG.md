# 更新日志

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
