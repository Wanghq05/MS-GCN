import numpy as np


def get_groups(dataset='NTU', CoM=21):
    groups = []

    if dataset == 'NTU':
        if CoM == 2:
            groups.append([2])
            groups.append([1, 21])
            groups.append([13, 17, 3, 5, 9])
            groups.append([14, 18, 4, 6, 10])
            groups.append([15, 19, 7, 11])
            groups.append([16, 20, 8, 12])
            groups.append([22, 23, 24, 25])

        ## Center of mass : 21
        elif CoM == 21:
            groups.append([27])
            groups.append([26, 46])
            groups.append([38, 42, 28, 30, 34])
            groups.append([39, 43, 29, 31, 35])
            groups.append([40, 44, 32, 36])
            groups.append([41, 45, 33, 37])
            groups.append([47, 48, 49, 50])
        else:
            raise ValueError()

    return groups


def get_edgeset(dataset='NTU', CoM=21):
    groups = get_groups(dataset=dataset, CoM=CoM)

    for i, group in enumerate(groups):
        group = [i - 1 for i in group]
        groups[i] = group  # 减一  关键点0-24
    H = [groups[i] + groups[i + 1] for i in range(len(groups) - 1)]
    print(H)

    identity = []
    forward_hierarchy = []  # 前向层次结构
    reverse_hierarchy = []  # 反向层次结构

    for i in range(len(groups) - 1):  # 0-6
        self_link = groups[i] + groups[i + 1]  # list[0]+list[1]
        self_link = [(i, i) for i in self_link]  # 还是自身节点连接
        identity.append(self_link)  # 25个关节点坐标的自身连接
        forward_g = []
        for j in groups[i]:
            for k in groups[i + 1]:
                forward_g.append((j, k))  # 将第一个层次集里面的节点与第二个层次集里面的节点相连
        forward_hierarchy.append(forward_g)

        reverse_g = []
        for j in groups[-1 - i]:
            for k in groups[-2 - i]:
                reverse_g.append((j, k))
        reverse_hierarchy.append(reverse_g)  # 最后一个层次集和倒数第二个层次集相连

    edges = []
    for i in range(len(groups) - 1):
        edges.append([identity[i], forward_hierarchy[i], reverse_hierarchy[-1 - i]])

    return edges

def get_hierarchical_graph(num_node, edges):
    A = []
    for edge in edges:
        A.append(get_graph(num_node, edge))
    A = np.stack(A)
    return A
def get_graph(num_node, edges):

    I = edge2mat(edges[0], num_node)
    Forward = normalize_digraph(edge2mat(edges[1], num_node))#边连接进行归一化处理
    Reverse = normalize_digraph(edge2mat(edges[2], num_node))
    A = np.stack((I, Forward, Reverse))
    return A # 3, 25, 25
# 原始CTR-GCN创建邻接矩阵的原理
def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An

def normalize_digraph(A):
    Dl = np.sum(A, 0)#求出每列元素的和 这不是度矩阵
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD#Dn是单位矩阵，输出的还是AD=A


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))#edge2mat(inward, num_node)将inward对应A的位置置一，代表最近连接边相连
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A#3*25*25 A[1]代表seif_link A[2]inward矩阵 A[3]outward 矩阵

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A



def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A