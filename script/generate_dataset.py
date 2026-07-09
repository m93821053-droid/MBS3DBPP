import numpy as np
import h5py
from CuttingBox import CuttingBoxCreator

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
        box_array_x = np.random.randint(int(bin_size_x * min_factor), int(bin_size_x * max_factor + 1), [box_num, ])
        box_array_y = np.random.randint(int(bin_size_y * min_factor), int(bin_size_y * max_factor + 1), [box_num, ])
        box_array_z = np.random.randint(int(min(bin_size_x, bin_size_y) * min_factor),
                                        int(max(bin_size_x, bin_size_y) * max_factor + 1),
                                        [box_num, ])
        box_array = np.stack([box_array_x, box_array_y, box_array_z], -1)

    return box_array

file_name = 'test_dataset.h5'

# 生成 1024 个 50x3 的 NumPy 数组
def generate_dates():
    num_arrays = 1024
    data = np.zeros((num_arrays, *(50,3)))
    for i in range(num_arrays):
        data[i] = generate_box_array(100, 100, 100, 50, 0.1, 0.5, box_set=None)

    # 将数据存储到 HDF5 文件
    with h5py.File(file_name, 'w') as f:
        # 创建一个数据集来存储所有数组
        f.create_dataset('arrays', data=data)

    print(f"Data has been written to {file_name}")

def print_datas():
    with h5py.File(file_name, 'r') as f:
        # 获取数据集
        data = f['arrays'][:]
        print(f"Data shape: {data.shape}")
        for i in range(data.shape[0]):
            print(data[i])

if __name__ == '__main__':
    #generate_dates()
    print_datas()