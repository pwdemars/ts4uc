#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def anytime_uniform_cost_search(node, 
                        terminal_timestep, 
                        net_demand_scenarios,
                        **policy_kwargs):
    """Uniform cost search with backup"""
    if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
        return node_mod.get_solution(node)
    frontier = queue.PriorityQueue()
    frontier.put((0, id(node), node)) # include the object id in the priority queue. prevents type error when path_costs are identical.
    while True:
        assert frontier, "Failed to find a goal state"
        node = frontier.get()[2]
        if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
            return node_mod.get_solution(node)
        actions = expansion.get_actions(node, **policy_kwargs)
        for action in actions:
            net_demand_scenarios_t = np.take(net_demand_scenarios, node.state.episode_timestep+1, axis=1)
            child = expansion.get_child_node(node, action, net_demand_scenarios_t)
            node.children[action.tobytes()] = child
            frontier.put((child.path_cost, id(child), child))

            # Early stopping if root has one child
            if node.parent is None and len(actions) == 1:
                return [actions[0]], 0
        backup(node)