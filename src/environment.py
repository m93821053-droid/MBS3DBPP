import numpy as np
from CuttingBox import CuttingBoxCreator
import copy

class Env():
    def __init__(self, bin_size_x, bin_size_y, bin_size_z, bin_size_ds_x, bin_size_ds_y, bin_type_list, max_bin_num, box_num, min_factor,
                 max_factor, feature_num=6, trunc_step=50, distance_threshold=0, gap_filling=False,
                 online=True, orientation=True, support_constraint=False, box_set=None):

        self.max_step = 8192

        # Environment Parameters
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.bin_size_z = bin_size_z
        self.bin_size_ds_x = bin_size_ds_x
        self.bin_size_ds_y = bin_size_ds_y
        self.bin_type_list = bin_type_list
        self.max_bin_num = max_bin_num
        self.box_num = box_num
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.feature_num = feature_num + 1
        self.distance_threshold = distance_threshold
        self.gap_filling = gap_filling
        self.online = online
        self.orientation = orientation
        self.support_constraint = support_constraint
        self.box_set = box_set
        self.trunc_step = trunc_step
        self.trunc_factor = 0.5
        self.corner_constraint = True
        self.ur_last = 0.5

        # Environment Variables
        self.gap = 0
        self.total_volume = 0
        self.max_index = [None] * len(self.bin_type_list)
        self.packing_result = []
        self.residual_box_num = box_num
        self.total_trunc_height = [0]*len(self.bin_type_list)


        # Environment Constants
        self.rotation_matrix = np.array([[[1, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, 1]],
                                         [[1, 0, 0],
                                          [0, 0, 1],
                                          [0, 1, 0]],
                                         [[0, 1, 0],
                                          [1, 0, 0],
                                          [0, 0, 1]],
                                         [[0, 0, 1],
                                          [1, 0, 0],
                                          [0, 1, 0]],
                                         [[0, 1, 0],
                                          [0, 0, 1],
                                          [1, 0, 0]],
                                         [[0, 0, 1],
                                          [0, 1, 0],
                                          [1, 0, 0]]])

        self.rotation_matrix_all = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                                             [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                                             [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0]])

        self.original_bin_height = [np.zeros((bin_size[0], bin_size[1]), dtype=np.float32) for bin_size in bin_type_list]
        self.original_state_bin = [self.get_bin_feature(i, self.bin_type_list[k]) for k,i in enumerate(self.original_bin_height)]
        self.block_size_x = [bin_size[0] // self.bin_size_ds_x for bin_size in bin_type_list]
        self.block_size_y = [bin_size[1] // self.bin_size_ds_y for bin_size in bin_type_list]

        self.original_state_bin_ds = [self.down_sampling(i)[0] for i in self.original_state_bin]
        self.original_max_index = [self.down_sampling(i)[1] for i in self.original_state_bin]


        mean_factor = (self.min_factor + self.max_factor) / 2
        mean_bin_size_z_list = [(min(bin_size[0], bin_size[1]) * self.min_factor + max(bin_size[0], bin_size[1]) * self.max_factor) / 2
                                for bin_size in bin_type_list]
        self.max_trunc_height = [int(mean_factor ** 2 * mean_bin_size_z * self.trunc_step * (1 / self.trunc_factor))
                                 for mean_bin_size_z in mean_bin_size_z_list]



    def get_distance(self, height_feature, bin_size, x_ori, is_equal=True, neg=False, threshold=0):
        plane_feature = np.ones_like(height_feature)
        if x_ori:
            count = np.ones((1, bin_size[1]))
            if neg:
                for x_index in range(1, bin_size[0]):
                    comparison = abs(height_feature[x_index, :] - height_feature[x_index - 1, :]) <= threshold
                    count = np.where(comparison, count + 1, 0)
                    plane_feature[x_index, :] = count * 1
            else:
                for x_index in range(bin_size[0] - 2, -1, -1):
                    if is_equal:
                        comparison = abs(height_feature[x_index, :] - height_feature[x_index + 1, :]) <= threshold
                    else:
                        comparison = height_feature[x_index, :] >= height_feature[x_index + 1, :]
                    count = np.where(comparison, count + 1, 1)
                    plane_feature[x_index, :] = count * 1
        else:
            count = np.ones((1, bin_size[0]))
            if neg:
                for y_index in range(1, bin_size[1]):
                    comparison = abs(height_feature[:, y_index] - height_feature[:, y_index - 1]) <= threshold
                    count = np.where(comparison, count + 1, 0)
                    plane_feature[:, y_index] = count * 1
            else:
                for y_index in range(bin_size[1] - 2, -1, -1):
                    if is_equal:
                        comparison = abs(height_feature[:, y_index] - height_feature[:, y_index + 1]) <= threshold
                    else:
                        comparison = height_feature[:, y_index] >= height_feature[:, y_index + 1]
                    count = np.where(comparison, count + 1, 1)
                    plane_feature[:, y_index] = count * 1

        return plane_feature

    def get_bin_feature(self, height_feature, bin_size):
        h_feature = height_feature
        p_feature_x = self.get_distance(h_feature, bin_size, x_ori=True)
        p_feature_y = self.get_distance(h_feature, bin_size, x_ori=False)
        p_feature_x_neg = self.get_distance(h_feature, bin_size, x_ori=True, neg=True)
        p_feature_y_neg = self.get_distance(h_feature, bin_size, x_ori=False, neg=True)
        p_feature_x_gap = self.get_distance(h_feature, bin_size, x_ori=True, is_equal=False)
        p_feature_y_gap = self.get_distance(h_feature, bin_size, x_ori=False, is_equal=False)

        bin_feature_state = np.stack(
            [h_feature, p_feature_x, p_feature_y, p_feature_x_neg, p_feature_y_neg, p_feature_x_gap, p_feature_y_gap],
            -1)

        if self.feature_num == 5:
            bin_feature_state = np.stack(
                [h_feature, p_feature_x, p_feature_y, p_feature_x_neg, p_feature_y_neg],
                -1)
        else:
            bin_feature_state = np.stack(
                [h_feature, p_feature_x, p_feature_y, p_feature_x_neg, p_feature_y_neg, p_feature_x_gap,
                 p_feature_y_gap],
                -1)

        return bin_feature_state

    @staticmethod
    def generate_box_array(bin_size_x, bin_size_y, bin_size_z, box_num, min_factor, max_factor, box_set=None):
        if box_set == "cut_1":
            boxCreator = CuttingBoxCreator([bin_size_x, bin_size_y, bin_size_z], [1, 1, 1, 5, 5, 5])
            for i in range(box_num):
                boxCreator.generate_box_size()
                # if boxCreator.box_list[-1][0] == bin_size_x:
                #     boxCreator.box_list.pop(-1)
                #     break
            box_array = np.array(boxCreator.box_list)
        elif box_set is not None:
            box_array = box_set[np.random.choice(np.arange(0, box_set.shape[0]), box_num, replace=True)]
        else:
            box_array_x = np.random.randint(int(bin_size_x * min_factor), int(bin_size_x * max_factor + 1), [box_num,])
            box_array_y = np.random.randint(int(bin_size_y * min_factor), int(bin_size_y * max_factor + 1), [box_num,])
            box_array_z = np.random.randint(int(min(bin_size_x,bin_size_y) * min_factor),
                                            int(max(bin_size_x,bin_size_y) * max_factor + 1),
                                            [box_num,])
            box_array = np.stack([box_array_x,box_array_y,box_array_z],-1)

        return box_array

    def down_sampling(self, bin_state):
        bin_size_x, bin_size_y, feature_num = bin_state.shape
        bin_state_split = np.stack(np.split(bin_state, self.bin_size_ds_x, 0), 0)
        bin_state_split = np.stack(np.split(bin_state_split, self.bin_size_ds_y, 2), 1).reshape(
            self.bin_size_ds_x * self.bin_size_ds_y, -1, feature_num)
        max_target = bin_state_split[:, :, 1] * bin_state_split[:, :, 2]
        max_idx = max_target.argmax(-1).reshape(-1, 1, 1)

        bin_state_ds = np.take_along_axis(bin_state_split, max_idx, 1).reshape(self.bin_size_ds_x, self.bin_size_ds_y,-1)

        return bin_state_ds, max_idx


    def down_sampling_mask(self, mask, max_index):
        box_num, ori_num, bin_size_x, bin_size_y = mask.shape
        mask_split = np.stack(np.split(mask, self.bin_size_ds_x, 2), 2)
        mask_split = np.stack(np.split(mask_split, self.bin_size_ds_y, 4), 3).reshape(box_num, ori_num,
                                                                                   self.bin_size_ds_x * self.bin_size_ds_y,
                                                                                   -1)
        mask_ds = np.take_along_axis(mask_split, max_index.reshape(1, 1, -1, 1), -1).reshape(box_num, ori_num,
                                                                                             self.bin_size_ds_x,
                                                                                             self.bin_size_ds_y)

        return mask_ds

    def reset(self, box_array=None):
        # 生成一个装箱问题，并返回状态
        # 重置参数
        # print("",flush=True)
        # print("Reset",flush=True)
        # print("",flush=True)
        self.not_first_bin = False
        self.gap = 0
        self.total_volume = 0
        self.total_trunc_height = [0] * len(self.bin_type_list)
        self.packing_result = [[] for _ in range(len(self.bin_type_list))]
        self.residual_box_num = self.box_num
        self.used_bin = [0] * len(self.bin_type_list)
        self.used_bin_type = [i for i in range(len(self.bin_type_list))]
        # 构造state_box
        if box_array is None:
            box_array = self.generate_box_array(self.bin_size_x, self.bin_size_y, self.bin_size_z, self.box_num, self.min_factor,
                                                self.max_factor, self.box_set)
        else:
            box_array = box_array * 1
        self.box_array = box_array * 1

        if self.online:
            state_box = self.box_array[:1, :]
            state_box = np.pad(state_box, ((0, self.box_num - 1), (0, 0)), constant_values=-1e9)
        else:
            state_box = self.box_array

        # 构造state_bin
        self.bin_height = copy.deepcopy(self.original_bin_height) * 1
        self.state_bin_list = self.original_state_bin * 1
        self.packing_mask_list = [[]]*len(self.state_bin_list)
        for id, state_bin in enumerate(self.state_bin_list):
            packing_mask = self.get_mask(state_bin, state_box, self.bin_type_list[self.used_bin_type[id]])
            state_bin = self.original_state_bin_ds[id] * 1
            self.max_index[id] = self.original_max_index[id] * 1
            packing_mask = self.down_sampling_mask(packing_mask, self.max_index[id])

            if self.corner_constraint:
                packing_mask = self.add_corner_constraint(state_bin, packing_mask)
            self.state_bin_list[id] = state_bin
            self.packing_mask_list[id] = packing_mask

        # 构造state_multi_bin

        self.state = (np.array(self.state_bin_list), state_box, np.array(self.packing_mask_list), np.array(self.used_bin_type), np.array(self.used_bin))

        return self.state

    def add_corner_constraint(self, state_bin, packing_mask):
        bin_height_temp = state_bin[:,:,0]
        tag_x = np.ones_like(bin_height_temp,dtype=np.bool_)
        tag_y = tag_x * 1
        for x_index in range(1,tag_x.shape[0]):
            tag = np.where(bin_height_temp[x_index,:] == bin_height_temp[x_index-1,:],0,1)
            tag_x[x_index,:] = tag * 1
        for y_index in range(1,tag_y.shape[1]):
            tag = np.where(bin_height_temp[:,y_index] == bin_height_temp[:,y_index-1],0,1)
            tag_y[:,y_index] = tag * 1
        tag_xy = (tag_x * tag_y).astype(np.bool_)
        tag_xy = tag_xy[np.newaxis,np.newaxis,:,:]
        packing_mask = packing_mask + ~tag_xy

        return packing_mask


    def get_mask_no_constraint(self, bin_type, state_bin, state_box, max_index=None):
        x_residual_size = np.zeros((self.bin_size_ds_x, self.bin_size_ds_y)) + np.arange(self.bin_size_ds_x, 0, -1).reshape(-1,1) * self.block_size_x[bin_type]
        y_residual_size = np.zeros((self.bin_size_ds_x, self.bin_size_ds_y)) + np.arange(self.bin_size_ds_y, 0, -1) * self.block_size_y[bin_type]
        position_residual_size = np.stack([x_residual_size, y_residual_size], 2)

        if self.online:
            available_box_num = 1
        else:
            available_box_num = self.residual_box_num

        box_array = state_box[:available_box_num] * 1  # residual_box_num x 3

        box_rotation_array = np.matmul(box_array, self.rotation_matrix_all).reshape(-1, 6,
                                                                                    3)  # residual_box_num x 6 x 3
        box_rotation_array = box_rotation_array[:, :, :2]

        if max_index is not None:
            max_index_size = np.stack([max_index / (self.block_size_y), max_index % (self.block_size_y)], -1)
            position_residual_size = position_residual_size - max_index_size.reshape(self.bin_size_ds_x,
                                                                                     self.bin_size_ds_y,
                                                                                     2)  # 10 x 10 x 2

        packing_available = box_rotation_array.reshape(-1, 6, 1, 1, 2) <= position_residual_size.reshape(1, 1,
                                                                                                         self.bin_size_ds_x,
                                                                                                         self.bin_size_ds_y,
                                                                                                         2)
        packing_available = packing_available.all(-1)

        packing_available = np.pad(packing_available, ((0, self.box_num - available_box_num), (0, 0), (0, 0), (0, 0)),
                                   constant_values=False)

        return ~packing_available

    def get_mask(self, state_bin, state_box, bin_size):
        bin_height = state_bin[:, :, 0] * 1  # bin_size x bin_size
        x_feature = state_bin[:, :, 1] * 1  # bin_size x bin_size
        y_feature = state_bin[:, :, 2] * 1  # bin_size x bin_size

        if self.online:
            available_box_num = 1
        else:
            available_box_num = self.residual_box_num

        box_array = state_box[:available_box_num] * 1  # residual_box_num x 3
        box_rotation_array = np.matmul(box_array, self.rotation_matrix_all).reshape(-1, 6,
                                                                                    3)  # residual_box_num x 6 x 3

        box_x_array, box_y_array, box_z_array = (box_rotation_array[:, :, i] for i in range(3))  # residual_box_num x 6

        x_array = y_feature.reshape(1, 1, bin_size[0], bin_size[1]) >= box_y_array.reshape(-1, 6, 1, 1)
        x_array = x_array.astype(int)
        for x_index in range(bin_size[0] - 2, -1, -1):
            x_array[:, :, x_index, :] = (x_array[:, :, x_index + 1, :] + 1) * x_array[:, :, x_index, :]
        x_limit = np.minimum(x_array, x_feature.reshape(1, 1, bin_size[0],
                                                        bin_size[1]))  # residual_box_num x 6 x bin_size x bin_size
        plane_available = (box_x_array.reshape(-1, 6, 1, 1) <= x_limit)

        height_overtop = (box_z_array.reshape(-1, 6, 1, 1) + bin_height.reshape(1, 1, bin_size[0],
                                                                                bin_size[1])) <= bin_size[2]
        x_feature2 = height_overtop.astype(int)
        y_feature2 = x_feature2 * 1
        for x_index in range(bin_size[0] - 2, -1, -1):
            x_feature2[:, :, x_index, :] = (x_feature2[:, :, x_index + 1, :] + 1) * x_feature2[:, :, x_index, :]
        for y_index in range(bin_size[1] - 2, -1, -1):
            y_feature2[:, :, :, y_index] = (y_feature2[:, :, :, y_index + 1] + 1) * y_feature2[:, :, :, y_index]

        x_array2 = y_feature2 >= box_y_array.reshape(-1, 6, 1, 1)
        x_array2 = x_array2.astype(int)
        for x_index in range(bin_size[0] - 2, -1, -1):
            x_array2[:, :, x_index, :] = (x_array2[:, :, x_index + 1, :] + 1) * x_array2[:, :, x_index, :]
        x_limit2 = np.minimum(x_array2, x_feature2)  # residual_box_num x 6 x bin_size x bin_size
        height_available = (box_x_array.reshape(-1, 6, 1, 1) <= x_limit2)  # residual_box_num x 6 x bin_size x bin_size

        if self.support_constraint:
            height_available = height_available * plane_available

        height_available = np.pad(height_available, ((0, self.box_num - available_box_num), (0, 0), (0, 0), (0, 0)),
                                  constant_values=False)
        #
        # if not self.orientation:
        #     height_available[:, 1:, :, :] = False

        if self.orientation is False:
            height_available[:, 1:, :, :] = False
        elif self.orientation == 2:
            height_available[:, 1, :, :] = False
            height_available[:, 3:, :, :] = False


        return ~height_available

    def step(self, action: tuple, debug=True):
        a_c, a_i, a_xy, a_r = action
        bin_type = self.used_bin_type[a_c]

        if self.max_index[a_c] is not None:
            sub_index = int(self.max_index[a_c][a_xy])
            a_x = a_xy // self.bin_size_ds_y * self.block_size_x[bin_type] + sub_index // self.block_size_y[bin_type]
            a_y = a_xy % self.bin_size_ds_y * self.block_size_y[bin_type] + sub_index % self.block_size_y[bin_type]
        else:
            a_x = a_xy // self.bin_size_y
            a_y = a_xy % self.bin_size_y


        # box_shape = self.state[1][a_i]
        box_shape = self.box_array[a_i]
        # box_shape = np.matmul(box_shape, self.rotation_matrix[a_r])
        box_rotation_shape = np.matmul(box_shape, self.rotation_matrix_all).reshape(6, 3)
        box_shape_rot = box_rotation_shape[a_r]
        box_length, box_width, box_height = map(int, box_shape_rot)
        # if a_x+box_length > self.bin_size or a_y+box_width > self.bin_size:
        #     a_x = min(a_x, self.bin_size-box_length)
        #     a_y = min(a_y, self.bin_size-box_width)

        if debug:
            try:
                assert a_x + box_length <= self.bin_type_list[bin_type][0] and a_y + box_width <= self.bin_type_list[bin_type][1]
            except AssertionError as e:
                #print(self.bin_height[a_c], flush=True)
                print("=======================================================", flush=True)
                print(action,flush=True)
                print(self.original_bin_height, flush=True)
                print(self.bin_height,flush=True)
                print(a_x , box_length, a_y , box_width,flush=True)
                print("=======================================================", flush=True)
                raise e

        # 优化后 - 使用视图避免大块内存复制
        a_z = np.max(self.bin_height[a_c][a_x:a_x + box_length, a_y:a_y + box_width])
        # 直接更新原数组区域
        self.bin_height[a_c][a_x:a_x + box_length, a_y:a_y + box_width] = a_z + box_height
        if self.bin_size_z < 10000 and debug:
            try:
                assert a_z <= self.bin_type_list[bin_type][2]
            except AssertionError as e:
                #print(self.bin_height[a_c], flush=True)
                print("==========================2===========================", flush=True)
                print(action,flush=True)
                print(self.original_bin_height,flush=True)
                print(self.bin_height,flush=True)
                #print(place_pack, flush=True)
                print("=======================================================", flush=True)
                raise e
        self.packing_result[a_c].append([box_length, box_width, box_height, a_x, a_y, a_z])

        # generate state_box
        self.box_array = np.delete(self.box_array, a_i, 0)
        self.box_array = np.pad(
            self.box_array, ((0, self.box_num - self.box_array.shape[0]), (0, 0)), constant_values=-1e9)
        if self.online:
            state_box = self.box_array[:1, :]
            state_box = np.pad(state_box, ((0, self.box_num - 1), (0, 0)), constant_values=-1e9)
        else:
            state_box = self.box_array

        self.residual_box_num -= 1

        # 初始化new bin
        used_new_bin = False
        if self.used_bin[a_c] == 0:
            used_new_bin = True
            self.used_bin[a_c] = 1
            if len(self.used_bin) < self.max_bin_num:
                self.packing_result.append([])
                self.total_trunc_height.append(0)
                #self.gap.append(0)
                #self.total_volume.append(0)
                self.used_bin.append(0)
                self.used_bin_type.append(self.used_bin_type[a_c])
                n_bin_height = copy.deepcopy(self.original_bin_height[bin_type])
                self.bin_height.append(n_bin_height)
                n_state_bin = self.original_state_bin[bin_type] * 1

                n_packing_mask = self.get_mask(n_state_bin, state_box, self.bin_type_list[self.used_bin_type[a_c]])
                n_state_bin = self.original_state_bin_ds[bin_type] * 1
                self.max_index.append(self.original_max_index[bin_type] * 1)
                n_packing_mask = self.down_sampling_mask(n_packing_mask, self.max_index[len(self.max_index) - 1])

                if self.corner_constraint:
                    n_packing_mask = self.add_corner_constraint(n_state_bin, n_packing_mask)
                self.state_bin_list.append(n_state_bin)
                self.packing_mask_list.append(n_packing_mask)


        # generate state_bin
        for bin_id in range(len(self.used_bin)):
            if np.all(self.packing_mask_list[bin_id]):
                continue
            elif self.used_bin[bin_id] == 0:
                self.packing_mask_list[bin_id][self.residual_box_num:] = True
                continue
            if np.max(self.bin_height[bin_id]) - self.total_trunc_height[bin_id] >= self.max_trunc_height[self.used_bin_type[bin_id]]:
                self.total_trunc_height[bin_id] += (self.trunc_factor * self.max_trunc_height[self.used_bin_type[bin_id]])
                bin_height_trunc = self.bin_height[bin_id] - self.total_trunc_height[bin_id]
                np.clip(bin_height_trunc, 0, None, out=bin_height_trunc)
            else:
                bin_height_trunc = self.bin_height[bin_id] - self.total_trunc_height[bin_id]
                np.clip(bin_height_trunc, 0, None, out=bin_height_trunc)
            #
            # if self.residual_box_num % self.trunc_step == 0 \
            #     and self.box_num-self.residual_box_num >= self.trunc_step \
            #     and self.residual_box_num != 0:
            #     mean_factor = (self.min_factor + self.max_factor) / 2
            #     mean_bin_size_z = (min(self.bin_size_x,self.bin_size_y) * self.min_factor \
            #                         +max(self.bin_size_x,self.bin_size_y) * self.max_factor) / 2
            #     self.total_trunc_height += int(mean_factor ** 2  * mean_bin_size_z * self.trunc_step)
            #     bin_height_trunc = self.bin_height - self.total_trunc_height
            #     np.clip(bin_height_trunc,0,None,out=bin_height_trunc)
            #
            # else:
            #     bin_height_trunc = self.bin_height - self.total_trunc_height
            #     np.clip(bin_height_trunc, 0, None, out=bin_height_trunc)
            #     print("max height:",np.max(bin_height_trunc))

            state_bin = self.get_bin_feature(bin_height_trunc, self.bin_type_list[self.used_bin_type[bin_id]])
            state_bin_ds, self.max_index[bin_id] = self.down_sampling(state_bin)

            # need downsampling

            if self.residual_box_num != 0:
                packing_mask = self.get_mask(state_bin, state_box, self.bin_type_list[self.used_bin_type[bin_id]])
                state_bin, self.max_index[bin_id] = self.down_sampling(state_bin)
                packing_mask = self.down_sampling_mask(packing_mask, self.max_index[bin_id])
            else:
                state_bin, self.max_index[bin_id] = self.down_sampling(state_bin)
                packing_mask = np.ones((self.box_num, 6, self.bin_size_ds_x, self.bin_size_ds_y))

            if self.corner_constraint:
                packing_mask = self.add_corner_constraint(state_bin,packing_mask)

            self.state_bin_list[bin_id] = state_bin
            self.packing_mask_list[bin_id] = packing_mask


        self.state = (np.array(self.state_bin_list), state_box, np.array(self.packing_mask_list), np.array(self.used_bin_type), np.array(self.used_bin))

        # done
        if self.residual_box_num == 0 or np.array(self.packing_mask_list).all():
            done = True
        else:
            done = False

        # reward

        self.total_volume += box_length * box_width * box_height

        total_bin_volume = 0
        for k in range(len(self.state_bin_list)):
            total_bin_volume += (
                        self.bin_type_list[self.used_bin_type[k]][0] * self.bin_type_list[self.used_bin_type[k]][1]
                        * self.bin_type_list[self.used_bin_type[k]][2] * self.used_bin[k])
        new_gap = total_bin_volume - self.total_volume
        reward = self.gap - new_gap
        # reward = 0
        self.gap = new_gap

        # print(bin_id, total_bin_volume)
        self.use_ratio = self.total_volume / total_bin_volume * 100

        return self.state, reward, done



if __name__ == '__main__':
    env = Env(10, 50, 0.1, 0.6,
              online=False, orientation=True, support_constraint=False, large_bin_size=True, limit_height=100000,
              box_set=None)
    for i in range(50):
        env.reset()

    # state_bin = np.random.randint(0,100,(100,100))
    # state_box = np.random.randint(0,50,(50,3))
    # for i in range(5):
    #     Env.get_mask(Env,state_bin,state_box)
