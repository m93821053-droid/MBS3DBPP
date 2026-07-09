import numpy as np
import torch
import os


class Arguments:
    def __init__(self, ):
        self.cwd = None  # current work directory. None means set automatically
        self.break_step = 10**7  # break training after 'total_step > break_step'

        # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.visible_gpu = '0, 1'
        # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.worker_num = 2

        self.load_model = True
        self.load_step = 0

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.lr_actor = 1e-5
        self.lr_critic = 1e-4
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        self.net_dim = 2 ** 9  # the network width
        # num of transitions sampled from replay buffer.
        self.batch_size = 1024
        self.repeat_times = 4  # collect target_step, then update network
        self.target_step = 1024 * 4  # repeatedly update network to keep critic's loss small
        self.max_memo = self.target_step  # capacity of replay buffer
        # GAE for on-policy sparse reward: Generalized Advantage Estimation.
        self.if_per_or_gae = True
        self.num_threads = 2
        self.trunc_step = 50

        '''Arguments for evaluate'''
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        '''Arguments for Network'''
        self.d_model = 128
        self.n_head = 4
        self.d_inner = 1024
        self.nlayers = 3

        '''Arguments for Environment'''
        self.bin_size_x = 100
        self.bin_size_y = 100
        self.bin_size_z = 100 # if no height limit, set the bin_size_z to 100000
        self.bin_size_ds_x = 10
        self.bin_size_ds_y = 10
        self.bin_size_dds_x = 2
        self.bin_size_dds_y = 2
        self.bin_type_list = [(50,50,50),(80,80,80),(100,100,100),(120,120,120),(50,80,100)]
        self.max_bin_num = 1000
        #self.bin_type_list = [(10,10,10),(20,20,20)]
        self.box_num = 50
        self.min_factor = 0.1
        self.max_factor = 0.5
        self.plane_feature_num = 6
        self.distance_threshold = 0
        self.gap_filling = False
        self.online = False
        self.support_constraint = False
        self.box_set_num = None
        self.orientation = 2


        # self.save_dir = "{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{}_{:d}_{:d}".format(
        #                         self.bin_size_x,self.bin_size_y,self.bin_size_z,self.bin_size_ds_x,
        #                             self.bin_size_ds_y,self.box_num,
        #                             int(self.min_factor*10),int(self.max_factor*10),int(self.online),
        #                             int(self.support_constraint),self.box_set_num,
        #                             self.orientation,self.trunc_step)
        self.save_dir = "{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_{}_{:d}_{:d}".format(
                                    self.bin_size_x,self.bin_size_y,self.bin_size_z,self.bin_size_ds_x,
                                        self.bin_size_ds_y,self.box_num,
                                        int(self.min_factor*10),int(self.max_factor*10),int(self.online),
                                        int(self.support_constraint),len(self.bin_type_list),self.box_set_num,
                                        self.orientation,self.trunc_step)
        print(self.save_dir)
        '''Other Arguments'''
        # self.explore_num = 1
        # self.process_num = self.target_step // self.box_num // self.explore_num
        self.process_num = 16

    def init_before_training(self):

        if not os.path.exists("save"):
            os.makedirs("save")

        save_index = 0
        while True:
            save_dir = self.save_dir + "_{}".format(save_index)

            if not os.path.exists("save/"+save_dir):
                if save_index == 0:
                    self.load_model = False
                    print("No save file, The load_model is set to False")
                if self.load_model:
                    save_dir = self.save_dir + "_{}".format(save_index-1)
                else:
                    os.makedirs("save/"+save_dir)
                self.save_dir = save_dir
                self.cwd = "./" + "save/" + save_dir + "/"
                if self.load_model:
                    with open(self.cwd+"last_step.txt","r") as f:
                        self.load_step = int(f.read())
                        print("load step:",self.load_step)
                break
            else:
                save_index += 1

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        #os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)
