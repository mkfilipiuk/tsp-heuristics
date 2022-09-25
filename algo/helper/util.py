import numpy as np
from numba import njit

@njit
def check_unvisited_node(unvisited):
    return np.any(unvisited)

# @njit
# def get_unvisited_node(unvisited):
#     r = np.argmax(unvisited > 0)
#     if unvisited[r]:
#         return r
#     return -1

@njit
def find_best_route(node_no, travel_route, min_distance):
    shortest_travel_route = np.argmin(min_distance)
    shortest_min_distance = min_distance[shortest_travel_route]
    return shortest_min_distance, travel_route[shortest_travel_route]

# @njit
# def in_travel_route(node, travel_route):
#     for t in travel_route:
#         if t == node:
#             return True
#     return False
