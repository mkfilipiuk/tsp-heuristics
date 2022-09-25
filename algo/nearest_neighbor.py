from tsp_heuristics.algo.helper import util
import numpy as np
import math
import numba
from numba import njit

@njit
def nearest_neighbour(distance_matrix: np.array):
    node_no = distance_matrix.shape[0]

    min_distance = np.zeros((node_no, 1), dtype=numba.float32)  # distances with starting node as min_distance[i]
    travel_route = np.zeros((node_no, node_no), dtype=numba.int32) 
    
    # Step 1
    for start_node in range(node_no):
        # Step 3
        unvisited = np.ones((node_no,), dtype=numba.int32)  # all nodes are unvisited
        unvisited[start_node] = 0
        travel_route[start_node][0] = start_node  # travel route starts with start_node

        node = start_node
        iteration = 1
        while util.check_unvisited_node(unvisited) and iteration < node_no:
            # Step 2
            # closest_arc = np.Inf
            # closest_node = node_no

            closest_node = np.argmin(np.where(unvisited == 1, distance_matrix[node], np.inf))
            closest_arc = distance_matrix[node][closest_node]
            
            # for node2 in range(node_no):
            #     if unvisited[node2] and 0 < distance_matrix[node][node2] < closest_arc:
            #         closest_arc = distance_matrix[node][node2]
            #         closest_node = node2

            if closest_node >= node_no:
                min_distance[start_node] = np.Inf
                break

            node = closest_node
            unvisited[node] = 0
            min_distance[start_node] = min_distance[start_node] + closest_arc
            travel_route[start_node][iteration] = node
            iteration = iteration + 1

        if not np.isinf(min_distance[start_node]):
            last_visited = travel_route[start_node][node_no-1]
            if distance_matrix[last_visited][start_node] > 0:
                min_distance[start_node] = min_distance[start_node] + distance_matrix[last_visited][start_node]
            else:
                min_distance[start_node] = np.Inf

    [shortest_min_distance, shortest_travel_route] = util.find_best_route(node_no, travel_route, min_distance)

    return shortest_min_distance, shortest_travel_route


if __name__ == '__main__':
    print(nearest_neighbour())