import copy
import json
import math
import random

from .mcgs import MCGS, MCGSNode, s1_context_size
from .evolution_interface import InterfaceEC


class MCGSAlgorithm:
    def __init__(self, paras, problem, select, manage, **kwargs):
        self.prob = problem
        self.select = select
        self.manage = manage

        self.use_local_llm = paras.llm_use_local
        self.url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        self.use_local_llm = kwargs.get('use_local_llm', False)
        assert isinstance(self.use_local_llm, bool)
        if self.use_local_llm:
            assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
            assert isinstance(kwargs.get('url'), str)
            self.url = kwargs.get('url')

        self.init_size = paras.init_size
        self.pop_size = paras.pop_size
        self.fe_max = paras.ec_fe_max
        self.eval_times = 0

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        self.s1_reference_top_k = paras.s1_reference_top_k
        if self.s1_reference_top_k < 0:
            raise ValueError("s1_reference_top_k must be >= 0")
        self.dual_lineage_backup = paras.dual_lineage_backup
        self.uct_value_mode = paras.uct_value_mode
        self.paternal_reward_discount = paras.paternal_reward_discount
        self.paternal_credit_depth = paras.paternal_credit_depth
        paras.ec_m = 5
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode
        self.ndelay = 1

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path
        self.exp_n_proc = paras.exp_n_proc
        self.timeout = paras.eva_timeout
        self.use_numba = paras.eva_numba_decorator
        self.mcgs = None

        print("- MCGS parameters loaded -")
        random.seed(2024)

    def add2pop(self, population, offspring):
        for ind in population:
            if ind['algorithm'] == offspring['algorithm']:
                if self.debug_mode:
                    print("duplicated result, retrying ... ")
        population.append(offspring)

    def expand(self, mcgs, cur_node, nodes_set, option):
        if option == 's1':
            s1_context = mcgs.build_s1_context(cur_node, self.s1_reference_top_k)
            if s1_context_size(s1_context) <= 1:
                return nodes_set
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(
                self.eval_times,
                s1_context,
                cur_node.raw_info,
                cur_node.children_info,
                option,
            )
        elif option == 'e1':
            e1_set = [
                copy.deepcopy(children.subtree[random.choices(range(len(children.subtree)), k=1)[0]].raw_info)
                for children in mcgs.root.children
            ]
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(
                self.eval_times,
                e1_set,
                cur_node.raw_info,
                cur_node.children_info,
                option,
            )
        else:
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(
                self.eval_times,
                nodes_set,
                cur_node.raw_info,
                cur_node.children_info,
                option,
            )
        if offsprings is None:
            print(f"Timeout emerge, no expanding with action {option}.")
            return nodes_set

        offsprings['operator'] = option
        if option != 'e1':
            print(
                f"Action: {option}, Father Obj: {cur_node.raw_info['objective']}, "
                f"Now Obj: {offsprings['objective']}, Depth: {cur_node.depth + 1}"
            )
        else:
            if self.interface_ec.check_duplicate_obj(mcgs.root.children_info, offsprings['objective']):
                print(f"Duplicated e1, no action, Father is Root, Abandon Obj: {offsprings['objective']}")
                return nodes_set
            print(f"Action: {option}, Father is Root, Now Obj: {offsprings['objective']}")

        if offsprings['objective'] != float('inf'):
            self.add2pop(nodes_set, offsprings)
            size_act = min(len(nodes_set), self.pop_size)
            nodes_set = self.manage.population_management(nodes_set, size_act)
            nownode = MCGSNode(
                offsprings['algorithm'],
                offsprings['code'],
                offsprings['objective'],
                parent=cur_node,
                depth=cur_node.depth + 1,
                visit=1,
                Q=-1 * offsprings['objective'],
                raw_info=offsprings,
            )
            mcgs.register_node(nownode)
            mcgs.apply_generation_context(
                nownode,
                option,
                prompt_parent_ids=offsprings.get("prompt_parent_ids", []),
                s1_context={
                    "maternal_path_ids": offsprings.get("s1_maternal_context_ids", []),
                    "paternal_top_k_ids": offsprings.get("s1_paternal_context_ids", []),
                },
            )
            if option == 'e1':
                nownode.subtree.append(nownode)
            cur_node.add_child(nownode)
            cur_node.children_info.append(offsprings)
            mcgs.backpropagate(nownode)
        return nodes_set

    @staticmethod
    def _json_number(value):
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value
        if math.isfinite(number):
            return number
        return None

    def export_mcgs_graph(self, mcgs=None):
        mcgs = mcgs or self.mcgs
        if mcgs is None:
            return {"schema_version": 1, "nodes": [], "edges": []}

        nodes = []
        edges = []
        stack = [mcgs.root]
        while stack:
            node = stack.pop()
            mcgs.register_node(node)
            raw_info = node.raw_info if isinstance(node.raw_info, dict) else {}
            operator = raw_info.get("operator", "root" if node is mcgs.root else None)
            nodes.append({
                "id": node.node_id,
                "node_id": node.node_id,
                "parent_id": node.parent.node_id if node.parent is not None else None,
                "operator": operator,
                "depth": node.depth,
                "visits": node.visits,
                "reward": self._json_number(node.reward),
                "Q": self._json_number(node.Q),
                "maternal_q": self._json_number(node.maternal_q),
                "paternal_q": self._json_number(node.paternal_q),
                "selection_q": self._json_number(node.selection_q),
                "objective": self._json_number(raw_info.get("objective")),
                "maternal_best_descendant_id": node.maternal_best_descendant_id,
                "paternal_best_descendant_id": node.paternal_best_descendant_id,
                "paternal_credit_count": node.paternal_credit_count,
                "paternal_reference_count": node.paternal_reference_count,
                "paternal_parent_ids": list(getattr(node, "paternal_parent_ids", [])),
                "s1_maternal_context_ids": list(getattr(node, "s1_maternal_context_ids", [])),
                "s1_paternal_context_ids": list(getattr(node, "s1_paternal_context_ids", [])),
                "algorithm": node.algorithm,
                "code": node.code,
            })
            if node.parent is not None:
                edges.append({
                    "source": node.parent.node_id,
                    "target": node.node_id,
                    "type": "maternal",
                    "edge_type": "maternal",
                    "operator": operator,
                })
            for paternal_id in getattr(node, "paternal_parent_ids", []):
                edges.append({
                    "source": paternal_id,
                    "target": node.node_id,
                    "type": "paternal_reference",
                    "edge_type": "paternal_reference",
                    "operator": operator,
                })
            if operator == "s1":
                for maternal_id in getattr(node, "s1_maternal_context_ids", []):
                    edges.append({
                        "source": maternal_id,
                        "target": node.node_id,
                        "type": "s1_maternal_context",
                        "edge_type": "s1_maternal_context",
                        "operator": operator,
                    })
                for paternal_id in getattr(node, "s1_paternal_context_ids", []):
                    edges.append({
                        "source": paternal_id,
                        "target": node.node_id,
                        "type": "s1_paternal_context",
                        "edge_type": "s1_paternal_context",
                        "operator": operator,
                    })
            stack.extend(reversed(node.children))

        return {
            "schema_version": 1,
            "algorithm": "mcgs",
            "dual_lineage_backup": mcgs.dual_lineage_backup,
            "uct_value_mode": mcgs.uct_value_mode,
            "paternal_reward_discount": mcgs.paternal_reward_discount,
            "paternal_credit_depth": mcgs.paternal_credit_depth,
            "s1_reference_top_k": self.s1_reference_top_k,
            "q_min": self._json_number(mcgs.q_min),
            "q_max": self._json_number(mcgs.q_max),
            "nodes": nodes,
            "edges": edges,
        }

    def write_mcgs_graph(self, mcgs):
        filename = self.output_path + "mcgs_graph_latest.json"
        with open(filename, 'w') as f:
            json.dump(self.export_mcgs_graph(mcgs), f, indent=2)

    def run(self):
        print("- Initialization Start -")

        interface_prob = self.prob
        self.interface_ec = InterfaceEC(
            self.m,
            self.api_endpoint,
            self.api_key,
            self.llm_model,
            self.debug_mode,
            interface_prob,
            use_local_llm=self.use_local_llm,
            url=self.url,
            select=self.select,
            n_p=self.exp_n_proc,
            timeout=self.timeout,
            use_numba=self.use_numba,
        )

        brothers = []
        mcgs = MCGS(
            'Root',
            dual_lineage_backup=self.dual_lineage_backup,
            uct_value_mode=self.uct_value_mode,
            paternal_reward_discount=self.paternal_reward_discount,
            paternal_credit_depth=self.paternal_credit_depth,
        )
        self.mcgs = mcgs
        n_op = len(self.operators)

        self.eval_times, brothers, offsprings = self.interface_ec.get_algorithm(self.eval_times, brothers, "i1")
        offsprings['operator'] = "i1"
        brothers.append(offsprings)
        nownode = MCGSNode(
            offsprings['algorithm'],
            offsprings['code'],
            offsprings['objective'],
            parent=mcgs.root,
            depth=1,
            visit=1,
            Q=-1 * offsprings['objective'],
            raw_info=offsprings,
        )
        mcgs.register_node(nownode)
        mcgs.apply_generation_context(nownode, "i1", prompt_parent_ids=offsprings.get("prompt_parent_ids", []))
        mcgs.root.add_child(nownode)
        mcgs.root.children_info.append(offsprings)
        mcgs.backpropagate(nownode)
        nownode.subtree.append(nownode)

        for i in range(1, self.init_size):
            self.eval_times, brothers, offsprings = self.interface_ec.get_algorithm(self.eval_times, brothers, "e1")
            offsprings['operator'] = "e1"
            brothers.append(offsprings)
            nownode = MCGSNode(
                offsprings['algorithm'],
                offsprings['code'],
                offsprings['objective'],
                parent=mcgs.root,
                depth=1,
                visit=1,
                Q=-1 * offsprings['objective'],
                raw_info=offsprings,
            )
            mcgs.register_node(nownode)
            mcgs.apply_generation_context(nownode, "e1", prompt_parent_ids=offsprings.get("prompt_parent_ids", []))
            mcgs.root.add_child(nownode)
            mcgs.root.children_info.append(offsprings)
            mcgs.backpropagate(nownode)
            nownode.subtree.append(nownode)

        nodes_set = brothers
        size_act = min(len(nodes_set), self.pop_size)
        nodes_set = self.manage.population_management(nodes_set, size_act)
        self.write_mcgs_graph(mcgs)
        print("- Initialization Finished - Evolution Start -")

        filename = self.output_path + "best_population_generation_" + str(self.eval_times) + ".json"
        while self.eval_times < self.fe_max:
            print(f"Current performances of MCGS nodes: {mcgs.rank_list}")
            cur_node = mcgs.root
            while len(cur_node.children) > 0 and cur_node.depth < mcgs.max_depth:
                uct_scores = [mcgs.uct(node, max(1 - self.eval_times / self.fe_max, 0)) for node in cur_node.children]
                selected_pair_idx = uct_scores.index(max(uct_scores))
                if int((cur_node.visits) ** mcgs.alpha) > len(cur_node.children):
                    if cur_node == mcgs.root:
                        op = 'e1'
                        nodes_set = self.expand(mcgs, cur_node, nodes_set, op)
                    else:
                        i = 1
                        op = self.operators[i]
                        nodes_set = self.expand(mcgs, cur_node, nodes_set, op)
                cur_node = cur_node.children[selected_pair_idx]

            for i in range(n_op):
                op = self.operators[i]
                print(f"Iter: {self.eval_times}/{self.fe_max} OP: {op}", end="|")
                op_w = self.operator_weights[i]
                for j in range(op_w):
                    nodes_set = self.expand(mcgs, cur_node, nodes_set, op)
                assert len(cur_node.children) == len(cur_node.children_info)

            filename = self.output_path + "population_generation_" + str(self.eval_times) + ".json"
            with open(filename, 'w') as f:
                json.dump(nodes_set, f, indent=5)

            filename = self.output_path + "best_population_generation_" + str(self.eval_times) + ".json"
            with open(filename, 'w') as f:
                json.dump(nodes_set[0], f, indent=5)
            self.write_mcgs_graph(mcgs)

        return nodes_set[0]["code"], filename
