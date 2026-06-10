from source.mcts_ahd import MCTS_AHD
from source.mcgs_algorithm import MCGSAlgorithm
from source.getParas import Paras
from source import prob_rank, pop_greedy
from problem_adapter import Problem

from utils.utils import init_client

class AHD:
    def __init__(self, cfg, root_dir, workdir, client) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.problem = Problem(cfg, root_dir)
        self.algorithm = self.cfg.get("algorithm", "mcts_ahd")

        self.paras = Paras()
        self.paras.set_paras(method=self.algorithm,
                             init_size=self.cfg.init_pop_size,
                             pop_size=self.cfg.pop_size,
                             s1_reference_top_k=self.cfg.get("s1_reference_top_k", 3),
                             dual_lineage_backup=self.cfg.get("dual_lineage_backup", False),
                             uct_value_mode=self.cfg.get("uct_value_mode", "maternal_only"),
                             paternal_reward_discount=self.cfg.get("paternal_reward_discount", 0.5),
                             paternal_credit_depth=self.cfg.get("paternal_credit_depth", 1),
                             llm_model=client,
                             ec_fe_max=self.cfg.max_fe,
                             exp_output_path=f"{workdir}/",
                             exp_debug_mode=False,
                             eva_timeout=cfg.timeout)
        init_client(self.cfg)

    def evolve(self):
        print("- Evolution Start -")

        if self.algorithm == "mcts_ahd":
            method = MCTS_AHD(self.paras, self.problem, prob_rank, pop_greedy)
            display_name = "MCTS-AHD"
        elif self.algorithm == "mcgs":
            method = MCGSAlgorithm(self.paras, self.problem, prob_rank, pop_greedy)
            display_name = "MCGS"
        else:
            raise ValueError(
                f"Unsupported algorithm {self.algorithm!r}. "
                "Use algorithm=mcts_ahd or algorithm=mcgs."
            )

        results = method.run()

        print("> End of Evolution! ")
        print("-----------------------------------------")
        print(f"---  {display_name} successfully finished!  ---")
        print("-----------------------------------------")

        return results
