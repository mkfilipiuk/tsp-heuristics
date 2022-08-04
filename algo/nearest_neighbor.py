from tsp_heuristics.algo.helper import util
import numpy as np
import math


def nearest_neighbour(distance_matrix: np.array):
    node_no = distance_matrix.shape[0]

    min_distance = np.zeros((node_no,), dtype=float)  # distances with starting node as min_distance[i]
    travel_route = [[0 for x in range(0, node_no)] for y in range(0, node_no)]

    # Step 1
    for start_node in range(0, node_no):
        # Step 3
        unvisited = np.ones((node_no,), dtype=int)  # all nodes are unvisited
        unvisited[start_node] = 0
        travel_route[start_node][0] = start_node  # travel route starts with start_node

        node = start_node
        iteration = 1
        while util.check_unvisited_node(unvisited) and iteration < node_no:
            # Step 2
            closest_arc = float('inf')
            closest_node = node_no

            for node2 in range(0, node_no):
                if unvisited[node2] == 1 and 0 < distance_matrix[node][node2] < closest_arc:
                    closest_arc = distance_matrix[node][node2]
                    closest_node = node2

            if closest_node >= node_no:
                min_distance[start_node] = float('inf')
                break

            node = closest_node
            unvisited[node] = 0
            min_distance[start_node] = min_distance[start_node] + closest_arc
            # print(min_distance[start_node])
            travel_route[start_node][iteration] = node
            iteration = iteration + 1

        if not math.isinf(min_distance[start_node]):
            last_visited = travel_route[start_node][node_no-1]
            if distance_matrix[last_visited][start_node] > 0:
                min_distance[start_node] = min_distance[start_node] + distance_matrix[last_visited][start_node]
            else:
                min_distance[start_node] = float('inf')


    [shortest_min_distance, shortest_travel_route] = util.find_best_route(node_no, travel_route, min_distance)

    return shortest_min_distance, shortest_travel_route


if __name__ == '__main__':
    print(nearest_neighbour())