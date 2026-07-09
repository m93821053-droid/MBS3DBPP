import time
import torch
from environment import Env


def Agent_explore_env(action_queue,
                      result_queue,
                      bin_size_x,
                      bin_size_y,
                      bin_size_z,
                      bin_size_ds_x,
                      bin_size_ds_y,
                      bin_size_ds_x_mb,
                      bin_size_ds_y_mb,
                      bin_type_list,
                      max_bin_num,
                      box_num,
                      min_factor,
                      max_factor,
                      plane_feature_num,
                      trunc_step,
                      online,
                      orientation,
                      support_constraint,
                      distance_threshold=0, gap_filling=False, box_set=None, iter_num=10000000000):

    env = Env(bin_size_x = bin_size_x,
              bin_size_y = bin_size_y,
              bin_size_z = bin_size_z,
              bin_size_ds_x = bin_size_ds_x,
              bin_size_ds_y = bin_size_ds_y,
              bin_type_list = bin_type_list,
              max_bin_num = max_bin_num,
              box_num=box_num,
              min_factor=min_factor,
              max_factor=max_factor,
              feature_num=plane_feature_num,
              trunc_step=trunc_step,
              distance_threshold=distance_threshold,
              gap_filling=gap_filling,
              online=online,
              orientation=orientation,
              support_constraint=support_constraint,
              box_set=box_set)

    iter_num = iter_num * box_num

    for _ in range(iter_num):
        action = action_queue.get()
        if action is False:
            state = env.reset()
            result_queue.put((state, 0, 0))
            action = action_queue.get()
        next_state, reward, done = env.step(action)
        if done:
            use_ratio = env.use_ratio
            next_state = env.reset()
        else:
            use_ratio = 0
        result_queue.put((next_state, reward, done, use_ratio))
        #print(result_queue.qsize(),flush=True)


def solve_problem(action_queue, result_queue, box_array_list, env):
    for box_array in box_array_list:
        done = False
        while not done:
            action = action_queue.get()
            if action is False:
                state = env.reset(box_array)
                result_queue.put((state, 0, 0))
                action = action_queue.get()
            next_state, reward, done = env.step(action)
            if done:
                use_ratio = env.use_ratio
                packing_result = env.packing_result
                used_bin_type = env.used_bin_type
                next_state = env.reset(box_array)
            else:
                use_ratio = 0
                packing_result = 0
            result_queue.put((next_state, reward, done, use_ratio, packing_result))