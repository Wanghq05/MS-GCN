import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 50#50


class Graph:
    def __init__(self, CoM=21, labeling_mode='spatial'):
        self.num_node = num_node
        self.CoM = CoM
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A1 = tools.get_hierarchical_graph(num_node, tools.get_edgeset(dataset='NTU', CoM=2))  # L, 3, 25, 25
            A2 = tools.get_hierarchical_graph(num_node, tools.get_edgeset(dataset='NTU', CoM=21))
            A = A1+A2
            A[3][1][3][28] = 1
            A[3][2][28][3] = 1
        else:
            raise ValueError()
        return A


















# # 三个邻接矩阵 [3,3,50,50]
# <<<<<<< HEAD
# # self_link0 = [(i, i) for i in range(num_node)]
# # inward_ori_index0 = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
# #                     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
# #                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
# #                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12),
# #
# #                     #(4,29),
# #
# #                     (26, 27), (27, 46), (28, 46), (29, 28), (30, 46), (31, 30), (32, 31),
# #                     (33, 32), (34, 46), (35, 34), (36, 35), (37, 36), (38, 26),
# #                     (39, 38), (40, 39), (41, 40), (42, 26), (43, 42), (44, 43),
# #                     (45, 44), (47, 48), (48, 33), (49, 50), (50, 37)]
# #
# #
# # inward0 = [(i - 1, j - 1) for (i, j) in inward_ori_index0]
# # outward0 = [(j, i) for (i, j) in inward0]
# # neighbor0 = inward0 + outward0
# =======
# self_link = [(i, i) for i in range(num_node)]
# inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
#                     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
#                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
#                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12),
#
#                     (4,29),
#
#                     (26, 27), (27, 46), (28, 46), (29, 28), (30, 46), (31, 30), (32, 31),
#                     (33, 32), (34, 46), (35, 34), (36, 35), (37, 36), (38, 26),
#                     (39, 38), (40, 39), (41, 40), (42, 26), (43, 42), (44, 43),
#                     (45, 44), (47, 48), (48, 33), (49, 50), (50, 37)]
#
#
# inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
# outward = [(j, i) for (i, j) in inward]
# neighbor = inward + outward
# >>>>>>> parent of e8f3350 (3个邻接矩阵 9个CTR-GC +AHA)
# # self_link1 = [(i, i) for i in range(25)]
# # inward_ori_index1 = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
# #                     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
# #                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
# #                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
# #
# # inward1 = [(i - 1, j - 1) for (i, j) in inward_ori_index1]
# # outward1 = [(j, i) for (i, j) in inward1]
# # neighbor1 = inward1 + outward1
# # self_link2 = [(i+25, i+25) for i in range(25)]
# # inward_ori_index2 = [(26, 27), (27, 46), (28, 46), (29, 28), (30, 46), (31, 30), (32, 31),
# #                     (33, 32), (34, 46), (35, 34), (36, 35), (37, 36), (38, 26),
# #                     (39, 38), (40, 39), (41, 40), (42, 26), (43, 42), (44, 43),
# #                     (45, 44), (47, 48), (48, 33), (49, 50), (50, 37)]
# # inward2 = [(i - 1, j - 1) for (i, j) in inward_ori_index2]
# # outward2 = [(j, i) for (i, j) in inward2]
# # neighbor2 = inward2 + outward2
# <<<<<<< HEAD
# # class Graph:
# #     def __init__(self, labeling_mode='spatial'):
# #         self.num_node = num_node
# #         self.self_link = self_link1
# #         self.inward0 = inward0
# #         self.outward0 = outward0
# #         self.neighbor0 = neighbor0
# #         self.inward1 = inward1
# #         self.outward1 = outward1
# #         self.neighbor1 = neighbor1
# #         self.inward2 = inward2
# #         self.outward2 = outward2
# #         self.neighbor2 = neighbor2
# #         self.A = self.get_adjacency_matrix(labeling_mode)
# #
# #     def get_adjacency_matrix(self, labeling_mode=None):
# #         if labeling_mode is None:
# #             return self.A
# #         if labeling_mode == 'spatial':
# #             A1 = tools.get_spatial_graph(num_node, self_link0, inward0, outward0)
# #             # B1 = A1.sum(0)
# #             A2 = tools.get_spatial_graph(num_node, self_link1, inward1, outward1)
# #             # B2 = A2.sum(0)
# #             A3 = tools.get_spatial_graph(num_node, self_link2, inward2, outward2)
# #             # B3 = A3.sum(0)
# #             # A = np.stack((B2,B3,B1))
# #             A = np.stack((A2,A3,A1))
# #         else:
# #             raise ValueError()
# #         return A
# =======
# class Graph:
#     def __init__(self, labeling_mode='spatial'):
#         self.num_node = num_node
#         self.self_link = self_link
#         # self.inward0 = inward0
#         # self.outward0 = outward0
#         self.neighbor = neighbor
#         # self.inward1 = inward1
#         # self.outward1 = outward1
#         # self.neighbor1 = neighbor1
#         # self.inward2 = inward2
#         # self.outward2 = outward2
#         # self.neighbor2 = neighbor2
#         self.A = self.get_adjacency_matrix(labeling_mode)
#
#     def get_adjacency_matrix(self, labeling_mode=None):
#         if labeling_mode is None:
#             return self.A
#         if labeling_mode == 'spatial':
#             A = tools.get_spatial_graph(num_node, self_link, inward, outward)
#             # B1 = A1.sum(0)
#             # A2 = tools.get_spatial_graph(num_node, self_link1, inward1, outward1)
#             # # B2 = A2.sum(0)
#             # A3 = tools.get_spatial_graph(num_node, self_link2, inward2, outward2)
#             # # B3 = A3.sum(0)
#             # A = np.stack((A2,A3,A1))
#             # A = np.stack((A2,A3,A1))
#         else:
#             raise ValueError()
#         return A
# >>>>>>> parent of e8f3350 (3个邻接矩阵 9个CTR-GC +AHA)
