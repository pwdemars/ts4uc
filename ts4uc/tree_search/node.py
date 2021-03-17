#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Node(object):
    """Node class for search trees"""
    def __init__(self, env, parent, action, step_cost, path_cost):
        self.state = env
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.is_expanded = False
        self.step_cost = step_cost
        self.children = {}
        self.heuristic_cost = None
        self.num_visits = 0

def get_solution(node):
    """Return the solution path (list of actions) leading to node."""
    s = []
    path_cost = node.path_cost
    while node.parent is not None:
        s.insert(0, node.action)
        node = node.parent
    return s, path_cost