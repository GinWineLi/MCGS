from __future__ import annotations

import copy
import math


UCT_VALUE_MODES = ("maternal_only", "dual_max", "paternal_only")


def individual_ids_from_context(context):
    if isinstance(context, dict):
        individuals = list(context.get("maternal_path", [])) + list(context.get("paternal_top_k", []))
    else:
        individuals = list(context or [])

    node_ids = []
    seen = set()
    for individual in individuals:
        if not isinstance(individual, dict):
            continue
        node_id = individual.get("node_id")
        if node_id is not None and node_id not in seen:
            seen.add(node_id)
            node_ids.append(node_id)
    return node_ids


def s1_context_size(context):
    return len(individual_ids_from_context(context))


class MCGSNode:
    def __init__(self, algorithm, code, obj, depth=0, is_root=False, parent=None, visit=0, raw_info=None, Q=0):
        self.algorithm = algorithm
        self.code = code
        self.parent = parent
        self.depth = depth
        self.children = []
        self.children_info = []
        self.visits = visit
        self.subtree = []
        self.raw_info = raw_info
        self.Q = Q
        self.reward = -1 * obj

        self.maternal_q = Q
        self.paternal_q = -math.inf
        self.selection_q = Q
        self.maternal_best_reward = self.reward
        self.paternal_best_reward = -math.inf
        self.maternal_best_descendant_id = None
        self.paternal_best_descendant_id = None
        self.paternal_credit_count = 0
        self.paternal_reference_count = 0

        self.node_id = None
        self.paternal_parent_ids = []
        self.s1_maternal_context_ids = []
        self.s1_paternal_context_ids = []

    def add_child(self, child_node: MCGSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCGSNode(node_id={self.node_id}, Q={self.Q:.2f}, visits={self.visits})"


class MCGS:
    def __init__(
        self,
        root_answer,
        dual_lineage_backup=False,
        uct_value_mode="maternal_only",
        paternal_reward_discount=0.5,
        paternal_credit_depth=1,
    ):
        self.exploration_constant_0 = 0.1
        self.alpha = 0.5
        self.max_depth = 10
        self.epsilon = 1e-10
        self.discount_factor = 1
        self.q_min = 0
        self.q_max = -10000
        self.rank_list = []
        self._next_node_id = 0
        self.nodes_by_id = {}
        self.configure_dual_lineage(
            dual_lineage_backup=dual_lineage_backup,
            uct_value_mode=uct_value_mode,
            paternal_reward_discount=paternal_reward_discount,
            paternal_credit_depth=paternal_credit_depth,
        )

        self.root = MCGSNode(algorithm=root_answer, code=root_answer, depth=0, obj=0, is_root=True)
        self.register_node(self.root)

        self.critiques = []
        self.refinements = []
        self.rewards = []
        self.selected_nodes = []

    def configure_dual_lineage(
        self,
        dual_lineage_backup=False,
        uct_value_mode="maternal_only",
        paternal_reward_discount=0.5,
        paternal_credit_depth=1,
    ):
        if uct_value_mode not in UCT_VALUE_MODES:
            raise ValueError(f"uct_value_mode must be one of {UCT_VALUE_MODES}, got {uct_value_mode!r}")
        if not 0 <= paternal_reward_discount <= 1:
            raise ValueError("paternal_reward_discount must be between 0 and 1")
        if paternal_credit_depth != 1:
            raise ValueError("paternal_credit_depth currently only supports direct references (1)")
        self.dual_lineage_backup = bool(dual_lineage_backup)
        self.uct_value_mode = uct_value_mode
        self.paternal_reward_discount = paternal_reward_discount
        self.paternal_credit_depth = paternal_credit_depth

    def register_node(self, node: MCGSNode):
        if node.node_id is None:
            while f"n{self._next_node_id:04d}" in self.nodes_by_id:
                self._next_node_id += 1
            node.node_id = f"n{self._next_node_id:04d}"
            self._next_node_id += 1
        self.nodes_by_id[node.node_id] = node
        self._ensure_node_value_fields(node)
        if node.maternal_best_descendant_id is None:
            node.maternal_best_descendant_id = node.node_id
        if isinstance(node.raw_info, dict):
            node.raw_info["node_id"] = node.node_id
        return node.node_id

    def _ensure_node_value_fields(self, node: MCGSNode):
        if not hasattr(node, "maternal_q"):
            node.maternal_q = node.Q
        if not hasattr(node, "paternal_q"):
            node.paternal_q = -math.inf
        if not hasattr(node, "selection_q"):
            node.selection_q = node.maternal_q
        if not hasattr(node, "maternal_best_reward"):
            node.maternal_best_reward = node.maternal_q
        if not hasattr(node, "paternal_best_reward"):
            node.paternal_best_reward = -math.inf
        if not hasattr(node, "maternal_best_descendant_id"):
            node.maternal_best_descendant_id = None
        if not hasattr(node, "paternal_best_descendant_id"):
            node.paternal_best_descendant_id = None
        if not hasattr(node, "paternal_credit_count"):
            node.paternal_credit_count = 0
        if not hasattr(node, "paternal_reference_count"):
            node.paternal_reference_count = 0
        if not hasattr(node, "paternal_parent_ids"):
            node.paternal_parent_ids = []
        if not hasattr(node, "s1_maternal_context_ids"):
            node.s1_maternal_context_ids = []
        if not hasattr(node, "s1_paternal_context_ids"):
            node.s1_paternal_context_ids = []
        node.Q = node.maternal_q
        self._refresh_selection_q(node)

    def _refresh_selection_q(self, node: MCGSNode):
        maternal_q = node.maternal_q
        paternal_q = node.paternal_q
        if not self.dual_lineage_backup or self.uct_value_mode == "maternal_only":
            node.selection_q = maternal_q
        elif self.uct_value_mode == "dual_max":
            node.selection_q = max(maternal_q, paternal_q)
        elif self.uct_value_mode == "paternal_only":
            node.selection_q = paternal_q if math.isfinite(paternal_q) else maternal_q
        else:
            raise ValueError(f"Unsupported uct_value_mode: {self.uct_value_mode}")
        return node.selection_q

    def _get_uct_value(self, node: MCGSNode):
        self._ensure_node_value_fields(node)
        return self._refresh_selection_q(node)

    def get_uct_value(self, node: MCGSNode):
        return self._get_uct_value(node)

    def get_node(self, node_id):
        return self.nodes_by_id.get(node_id)

    def maternal_path_nodes(self, node: MCGSNode):
        path = []
        current = node
        while current is not None and current is not self.root:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

    def build_s1_context(self, node: MCGSNode, paternal_top_k: int):
        if paternal_top_k < 0:
            raise ValueError("paternal_top_k must be >= 0")
        maternal_nodes = self.maternal_path_nodes(node)
        maternal_ids = [self.register_node(item) for item in maternal_nodes]
        maternal_id_set = set(maternal_ids)

        paternal_nodes = []
        seen = set()
        for maternal_node in maternal_nodes:
            for paternal_id in getattr(maternal_node, "paternal_parent_ids", []):
                if paternal_id in maternal_id_set or paternal_id in seen:
                    continue
                paternal_node = self.get_node(paternal_id)
                if paternal_node is None:
                    continue
                seen.add(paternal_id)
                paternal_nodes.append(paternal_node)

        paternal_nodes.sort(key=lambda item: (-item.reward, self.register_node(item)))
        paternal_nodes = paternal_nodes[:paternal_top_k]
        paternal_ids = [self.register_node(item) for item in paternal_nodes]
        return {
            "maternal_path": [copy.deepcopy(item.raw_info) for item in maternal_nodes],
            "paternal_top_k": [copy.deepcopy(item.raw_info) for item in paternal_nodes],
            "maternal_path_ids": maternal_ids,
            "paternal_top_k_ids": paternal_ids,
        }

    def apply_generation_context(self, node: MCGSNode, operator, prompt_parent_ids=None, s1_context=None):
        self.register_node(node)
        maternal_parent_id = self.register_node(node.parent) if node.parent is not None else None
        prompt_parent_ids = list(dict.fromkeys(prompt_parent_ids or []))
        if operator in ("e1", "e2"):
            node.paternal_parent_ids = [node_id for node_id in prompt_parent_ids if node_id != maternal_parent_id]
        if operator == "s1" and isinstance(s1_context, dict):
            node.s1_maternal_context_ids = list(s1_context.get("maternal_path_ids", []))
            node.s1_paternal_context_ids = list(s1_context.get("paternal_top_k_ids", []))
            node.paternal_parent_ids = list(dict.fromkeys(node.s1_paternal_context_ids))

        for paternal_id in node.paternal_parent_ids:
            reference = self.get_node(paternal_id)
            if reference is not None:
                reference.paternal_reference_count += 1

        if isinstance(node.raw_info, dict):
            node.raw_info["paternal_parent_ids"] = list(node.paternal_parent_ids)
            node.raw_info["s1_maternal_context_ids"] = list(node.s1_maternal_context_ids)
            node.raw_info["s1_paternal_context_ids"] = list(node.s1_paternal_context_ids)

    def backpropagate(self, node: MCGSNode):
        self._ensure_node_value_fields(node)
        if node.maternal_q not in self.rank_list:
            self.rank_list.append(node.maternal_q)
            self.rank_list.sort()
        self.q_min = min(self.q_min, node.maternal_q)
        self.q_max = max(self.q_max, node.maternal_q)
        self._refresh_selection_q(node)

        parent = node.parent
        while parent:
            self._ensure_node_value_fields(parent)
            best_child = max(parent.children, key=lambda child: child.maternal_q)
            best_child_q = best_child.maternal_q
            parent.maternal_q = parent.maternal_q * (1 - self.discount_factor) + best_child_q * self.discount_factor
            parent.Q = parent.maternal_q
            parent.maternal_best_reward = best_child.maternal_best_reward
            parent.maternal_best_descendant_id = best_child.maternal_best_descendant_id
            parent.visits += 1
            self._refresh_selection_q(parent)
            if parent.code != 'Root' and parent.parent.code == 'Root':
                parent.subtree.append(node)
            parent = parent.parent

        if self.dual_lineage_backup:
            self._backpropagate_paternal(node)

    def _backpropagate_paternal(self, node: MCGSNode):
        credited_reward = self.paternal_reward_discount * node.reward
        for paternal_id in node.paternal_parent_ids:
            reference = self.get_node(paternal_id)
            if reference is None:
                continue
            self._ensure_node_value_fields(reference)
            reference.paternal_credit_count += 1
            if credited_reward > reference.paternal_q:
                reference.paternal_q = credited_reward
                reference.paternal_best_reward = credited_reward
                reference.paternal_best_descendant_id = node.node_id
            selection_q = self._refresh_selection_q(reference)
            if self.uct_value_mode != "maternal_only":
                self.q_min = min(self.q_min, selection_q)
                self.q_max = max(self.q_max, selection_q)

    def uct(self, node: MCGSNode, eval_remain):
        self.exploration_constant = self.exploration_constant_0 * eval_remain
        value = self._get_uct_value(node)
        if self.q_max == self.q_min:
            normalized_value = 0.0
        else:
            normalized_value = (value - self.q_min) / (self.q_max - self.q_min)
        return normalized_value + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / node.visits
        )

    def is_fully_expanded(self, node: MCGSNode):
        return len(node.children) >= self.max_children or any(
            child.Q > node.Q for child in node.children
        ) or node.code == 'Root'
