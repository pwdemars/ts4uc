#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time 
import queue
import gc

from ts4uc.tree_search import node as node_mod, expansion, informed_search

def solve_day_ahead_anytime(env, 
                            time_budget, 
                            net_demand_scenarios,
                            tree_search_func, 
                            **params):
    """
    """
    env.reset()
    final_schedule = np.zeros((env.episode_length, env.num_gen))

    root = node_mod.Node(env=env,
            parent=None,
            action=None,
            step_cost=0,
            path_cost=0)

    for t in range(env.episode_length):
        s = time.time()
        path = tree_search_func(root, 
                                      time_budget,
                                      net_demand_scenarios,
                                      **params)
        a_best = path[0]

        final_schedule[t, :] = a_best
        env.step(a_best, deterministic=True)
        print(f"Period {env.episode_timestep+1}", np.array(a_best, dtype=int), round(time.time()-s, 2))

        root = root.children[a_best.tobytes()]
        root.parent, root.path_cost = None, 0

        gc.collect()
        
    return final_schedule

def ida_star(node,
             time_budget,
             net_demand_scenarios,
             heuristic_method,
             **policy_kwargs):
    start_time = time.time() 
    horizon = 1
    terminal_timestep = min(node.state.episode_timestep + horizon, node.state.episode_length-1)
    while (time.time() - start_time) < time_budget:
        print("Horizon: {}".format(horizon))
        if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
            best_path, _ = node_mod.get_solution(node)
            break
        frontier = queue.PriorityQueue()
        frontier.put((0, id(node), node)) # include the object id in the priority queue. prevents type error when path_costs are identical.
        while (time.time() - start_time) < time_budget:
            assert frontier, "Failed to find a goal state"
            node = frontier.get()[2]
            if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
                best_path, _ = node_mod.get_solution(node)
                break
            actions = expansion.get_actions(node, **policy_kwargs)
            for action in actions:
                net_demand_scenarios_t = np.take(net_demand_scenarios, node.state.episode_timestep+1, axis=1)
                child = expansion.get_child_node(node, action, net_demand_scenarios_t)
                child.heuristic_cost = informed_search.heuristic(child, terminal_timestep - child.state.episode_timestep, heuristic_method)
                node.children[action.tobytes()] = child
                frontier.put((child.path_cost + child.heuristic_cost, id(child), child))

                # Early stopping if root has one child
                if node.parent is None and len(actions) == 1:
                    best_path, _ [actions[0]], 0
                    break
        horizon += 1
        terminal_timestep = min(node.state.episode_timestep + horizon, node.state.episode_length-1)

    return best_path

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
