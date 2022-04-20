import torch
import copy
import random
import scipy.sparse as sp
import numpy as np


def shear(data_numpy, r=0.5):
    s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]
    s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

    R = np.array([[1,          s1_list[0], s2_list[0]],
                  [s1_list[1], 1,          s2_list[1]],
                  [s1_list[2], s2_list[2], 1        ]])

    R = R.transpose()
    data_numpy = np.dot(data_numpy.transpose([1, 2, 3, 0]), R)
    data_numpy = data_numpy.transpose(3, 0, 1, 2)
    return data_numpy


def temperal_crop(data_numpy, temperal_padding_ratio=6):
    C, T, V, M = data_numpy.shape
    padding_len = T // temperal_padding_ratio
    frame_start = np.random.randint(0, padding_len * 2 + 1)
    data_numpy = np.concatenate((data_numpy[:, :padding_len][:, ::-1],
                                 data_numpy,
                                 data_numpy[:, -padding_len:][:, ::-1]),
                                axis=1)
    data_numpy = data_numpy[:, frame_start:frame_start + T]
    return data_numpy


def aug_random_mask(input_feature, drop_percent=0.2):
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def aug_random_edge(input_adj, drop_percent=0.2):
    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    single_index_list = []
    for i in list(index_list):
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))

    edge_num = int(len(row_idx) / 2)  # 9228 / 2
    add_drop_num = int(edge_num * percent / 2)
    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)

    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0

    '''
    above finish drop edges
    '''
    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:
        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1

    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj


def aug_drop_node(input_fea, input_adj, drop_percent=0.2):
    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)

    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)  # number of drop nodes
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def aug_subgraph(input_fea, input_adj, drop_percent=0.2):
    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)
    node_num = input_fea.shape[0]

    all_node_list = [i for i in range(node_num)]
    s_node_num = int(node_num * (1 - drop_percent))
    center_node_id = random.randint(0, node_num - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []

    for i in range(s_node_num - 1):

        all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()

        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break

    drop_node_list = sorted([i for i in all_node_list if not i in sub_node_id_list])

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def delete_row_col(input_matrix, drop_list, only_row=False):
    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out


# scale(10)
def reduce2part(X, joint_num=25):
    left_leg_up = [16, 17]
    left_leg_down = [18, 19]
    right_leg_up = [12, 13]
    right_leg_down = [14, 15]
    torso = [0, 1]
    head = [2, 3, 20]
    left_arm_up = [8, 9]
    left_arm_down = [10, 11, 23, 24]
    right_arm_up = [4, 5]
    right_arm_down = [6, 7, 21, 22]
    X = X.cpu().numpy()
    x_torso = np.mean(X[:, :, :, torso, :], axis=3)  # [N * T, V=1]
    x_leftlegup = np.mean(X[:, :, :, left_leg_up, :], axis=3)
    x_leftlegdown = np.mean(X[:, :, :, left_leg_down, :], axis=3)
    x_rightlegup = np.mean(X[:, :, :, right_leg_up, :], axis=3)
    x_rightlegdown = np.mean(X[:, :, :, right_leg_down, :], axis=3)
    x_head = np.mean(X[:, :, :, head, :], axis=3)
    x_leftarmup = np.mean(X[:, :, :, left_arm_up, :], axis=3)
    x_leftarmdown = np.mean(X[:, :, :, left_arm_down, :], axis=3)
    x_rightarmup = np.mean(X[:, :, :, right_arm_up, :], axis=3)
    x_rightarmdown = np.mean(X[:, :, :, right_arm_down, :], axis=3)
    # X_part = np.concatenate((x_leftlegup, x_leftlegdown, x_rightlegup, x_rightlegdown, x_torso, x_head, x_leftarmup,
    #                          x_leftarmdown, x_rightarmup, x_rightarmdown), axis=-1) \
    #                         .reshape([X.shape[0], X.shape[1], X.shape[2], 10, 2])
    X_part = np.concatenate((x_torso, x_head, x_rightarmup, x_rightarmdown, x_leftarmup, x_leftarmdown,
                             x_rightlegup, x_rightlegdown, x_leftlegup, x_leftlegdown), axis=-1) \
                            .reshape([X.shape[0], X.shape[1], X.shape[2], 10, 2])
    X_part=torch.tensor(X_part).cuda()
    return X_part


# scale(5)
def reduce2body(X, joint_num=25):
    left_leg = [16, 17, 18, 19]
    right_leg = [12, 13, 14, 15]
    torso = [0, 1, 2, 3, 20]
    left_arm = [8, 9, 10, 11, 23, 24]
    right_arm = [4, 5, 6, 7, 21, 22]
    x_torso = np.mean(X[:, :, :, torso, :], axis=3)  # [N * T, V=1]
    x_leftleg = np.mean(X[:, :, :, left_leg, :], axis=3)
    x_rightleg = np.mean(X[:, :, :, right_leg, :], axis=3)
    x_leftarm = np.mean(X[:, :, :, left_arm, :], axis=3)
    x_rightarm = np.mean(X[:, :, :, right_arm, :], axis=3)
    X_body = np.concatenate((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm), axis=-1)\
                            .reshape([X.shape[0], X.shape[1],  X.shape[2], 5, 2])
    return X_body


# scale(29)
def interpolation(X, joint_num=25):
    left_leg_up = [16, 17]
    left_leg_down = [18, 19]
    right_leg_up = [12, 13]
    right_leg_down = [14, 15]
    torso = [0, 1]
    head_1 = [2, 3]
    head_2 = [2, 20]
    left_arm_up = [8, 9]
    left_arm_down_1 = [10, 11]
    left_arm_down_2 = [11, 24]
    left_arm_down_3 = [24, 23]
    right_arm_up = [4, 5]
    right_arm_down_1 = [6, 7]
    right_arm_down_2 = [7, 22]
    right_arm_down_3 = [22, 21]
    shoulder_1 = [8, 20]
    shoulder_2 = [4, 20]
    elbow_1 = [9, 10]
    elbow_2 = [5, 6]
    spine_mm = [20, 1]
    hip_1 = [0, 16]
    hip_2 = [0, 12]
    knee_1 = [17, 18]
    knee_2 = [13, 14]
    x_torso = np.mean(X[:, :, :, torso, :], axis=3)  # [N * T, V=1]
    x_leftlegup = np.mean(X[:,:, :, left_leg_up, :], axis=3)
    x_leftlegdown = np.mean(X[:,:, :, left_leg_down, :], axis=3)
    x_rightlegup = np.mean(X[:,:, :, right_leg_up, :], axis=3)
    x_rightlegdown = np.mean(X[:,:, :, right_leg_down, :], axis=3)
    x_head_1 = np.mean(X[:,:, :, head_1, :], axis=3)
    x_head_2 = np.mean(X[:,:, :, head_2, :], axis=3)
    x_leftarmup = np.mean(X[:,:, :, left_arm_up, :], axis=3)
    x_leftarmdown_1 = np.mean(X[:,:, :, left_arm_down_1, :], axis=3)
    x_leftarmdown_2 = np.mean(X[:,:, :, left_arm_down_2, :], axis=3)
    x_leftarmdown_3 = np.mean(X[:,:, :, left_arm_down_3, :], axis=3)
    x_rightarmup = np.mean(X[:,:, :, right_arm_up, :], axis=3)
    x_rightarmdown_1 = np.mean(X[:,:, :, right_arm_down_1, :], axis=3)
    x_rightarmdown_2 = np.mean(X[:,:, :, right_arm_down_2, :], axis=3)
    x_rightarmdown_3 = np.mean(X[:,:, :, right_arm_down_3, :], axis=3)
    shoulder_1 = np.mean(X[:,:, :, shoulder_1, :], axis=3)
    shoulder_2 = np.mean(X[:,:, :, shoulder_2, :], axis=3)
    elbow_1 = np.mean(X[:,:, :, elbow_1, :], axis=3)
    elbow_2 = np.mean(X[:,:, :, elbow_2, :], axis=3)
    spine_mm = np.mean(X[:,:, :, spine_mm, :], axis=3)
    hip_1 = np.mean(X[:,:, :, hip_1, :], axis=3)
    hip_2 = np.mean(X[:,:, :, hip_2, :], axis=3)
    knee_1 = np.mean(X[:,:, :, knee_1, :], axis=3)
    knee_2 = np.mean(X[:,:, :, knee_2, :], axis=3)
    X_part = np.concatenate((x_leftlegup, x_leftlegdown, x_rightlegup,
                             x_rightlegdown, x_torso, x_head_1, x_head_2, x_leftarmup,
                             x_leftarmdown_1, x_leftarmdown_2, x_leftarmdown_3,
                             x_rightarmup, x_rightarmdown_1, x_rightarmdown_2, x_rightarmdown_3,
                             shoulder_1, shoulder_2, elbow_1, elbow_2, spine_mm,
                             hip_1, hip_2, knee_1, knee_2), axis=-1) \
        .reshape([X.shape[0], X.shape[1],X.shape[2], 24, 2])
    # 25+24
    X_interp = np.concatenate((X, X_part), axis=-2)
    return X_interp

#测试
# data_seq = np.ones((8,3, 50, 25, 2))
# new=np.zeros((8,3,50,13,2))
# new1=np.zeros((8,3,50,25,2))
# new2=reduce2part(data_seq,25)
# new2=reduce2body(data_seq,25)
# new2=interpolation(data_seq,25)
