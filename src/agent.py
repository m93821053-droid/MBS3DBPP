from network import ActorNetwork, CriticNetwork

import torch

import numpy as np
from copy import deepcopy
import logging
from network import ActorNetwork, CriticNetwork
import time


class AgentPPO:
    def __init__(self, bin_size_x,bin_size_y, box_num, bin_type_list, online,plane_feature_num, lr_actor, lr_critic, if_use_gae=True, load_model=None,
                 cwd=None, gpu_id=0, env_num=1):
        super().__init__()

        self.criterion = torch.nn.MSELoss()

        self.ratio_clip = 0.12  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.01~0.05
        # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.lambda_gae_adv = 0.99
        # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw


        self.device = torch.device(f"cuda:{gpu_id}" if (
                torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")  # 定义GPU
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trajectory_list = [list() for _ in range(env_num)]  # 定义轨迹列表
        # 1.设置获得reward的方式
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

        binary_dim = 10
        self.cri = CriticNetwork(bin_size_x, bin_size_y, box_num, online,plane_feature_num=plane_feature_num,binary_dim=binary_dim).to(self.device)
        self.act = ActorNetwork(bin_size_x,bin_size_y, box_num, bin_type_list, online,plane_feature_num=plane_feature_num,binary_dim=binary_dim).to(self.device)
        # self.cri = torch.nn.DataParallel(self.cri)
        # self.act = torch.nn.DataParallel(self.act)
        # self.cri = self.cri.to(self.device)
        # self.act = self.act.to(self.device)

        if load_model:
            self.act.load_state_dict(torch.load(cwd + "actor.pth"))
            self.cri.load_state_dict(torch.load(cwd + "critic.pth"))
            logging.info("================Load model================")

        # if load_model:
        #     state_dict1 = torch.load(cwd + "actor.pth")
        #     new_state_dict1 = {"module." + k: v for k, v in state_dict1.items()}
        #     self.act.load_state_dict(new_state_dict1)
        #     state_dict2 = torch.load(cwd + "critic.pth")
        #     new_state_dict2 = {"module." + k: v for k, v in state_dict2.items()}
        #     self.cri.load_state_dict(new_state_dict2)
        #     logging.info("================Load model================")

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), lr_critic)
        self.act_optim = torch.optim.Adam(self.act.parameters(), lr_actor)

    def select_action(self, state):
        state = [torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0) for s in state]
        actions, probs = self.act.get_action(state)
        actions = [act[0].detach().cpu().numpy() for act in actions]
        probs = [prob[0, :].detach().cpu().numpy() for prob in probs]

        return actions, probs

    def explore_env_mp(self, action_queue_list, result_queue_list, target_step, reward_scale, gamma):
        process_num = len(action_queue_list)
        [action_queue_list[pi].put(False) for pi in range(process_num)]
        self.goal_avg = 0
        episode_num = 0
        srdan_temp = [[] for _ in range(process_num)]
        last_done = [0] * process_num

        result_list = [result_queue.get() for result_queue in result_queue_list]

        state_list = [result[0] for result in result_list]


        # print(len(state_list), flush=True)
        # print(len(state_list[0]), flush=True)
        # print(state_list[0], flush=True)
        # print(state_list[0][0], flush=True)
        use_ratio_list = []
        for i in range(target_step // process_num):

            state = list(map(list, zip(*state_list)))
            for k in range(len(state)):
                state[k] = [torch.as_tensor(i,dtype=torch.float32) for i in state[k]]

            state[0] = torch.nn.utils.rnn.pad_sequence(state[0], batch_first=True, padding_value=0)
            state[1] = torch.nn.utils.rnn.pad_sequence(state[1], batch_first=True, padding_value=0)
            state[2] = torch.nn.utils.rnn.pad_sequence(state[2], batch_first=True, padding_value=1)
            state[3] = torch.nn.utils.rnn.pad_sequence(state[3], batch_first=True, padding_value=0)
            state[4] = torch.nn.utils.rnn.pad_sequence(state[4], batch_first=True, padding_value=0)

            state = [torch.as_tensor(np.array(s), dtype=torch.float32, device=self.device) for s in state]
            # for i in range(len(state)):
            #     print(state[1].size(),flush=True)
            actions, probs = self.act.get_action(state)
            # action_list = [[act[p].detach().cpu().numpy() for act in actions] for p in range(process_num)]
            action_list = np.array([act.detach().cpu().numpy() for act in actions]).transpose()
            # action_list2 = list(zip(*[act.detach().cpu().numpy() for act in actions]))
            # probs_list = [[prob[p, :].detach().cpu().numpy() for prob in probs] for p in range(process_num)]
            probs_list = list(zip(*[prob.detach().cpu().numpy() for prob in probs]))
            action_int_list = action_list.tolist()
            [action_queue_list[pi].put(action_int_list[pi]) for pi in range(process_num)]
            result_list = [result_queue.get() for result_queue in result_queue_list]
            use_ratio_list += [result[3] for result in result_list if result[2] != 0]



            [srdan_temp[pi].append(
                (state_list[pi], result_list[pi][1], result_list[pi][2], action_list[pi], probs_list[pi])) for pi in
                range(process_num)]

            result_list = list(map(list, zip(*result_list)))
            state_list = result_list[0]

            for process_index in range(process_num):
                if result_list[2][process_index]:
                    self.goal_avg = (self.goal_avg * episode_num + result_list[3][process_index]) / (episode_num + 1)
                    episode_num += 1
                    last_done[process_index] = i

        srdan_list = list()
        for process_index in range(process_num):
            srdan_list.extend(srdan_temp[process_index][:last_done[process_index] + 1])
        srdan_list = list(map(list, zip(*srdan_list)))
        state_list = list(map(list, zip(*(srdan_list[0]))))
        for k in range(len(state_list)):
            state_list[k] = [torch.as_tensor(i, dtype=torch.float32) for i in state_list[k]]
        state_list[0] = torch.nn.utils.rnn.pad_sequence(state_list[0], batch_first=True, padding_value=0)
        state_list[1] = torch.nn.utils.rnn.pad_sequence(state_list[1], batch_first=True, padding_value=0)
        state_list[2] = torch.nn.utils.rnn.pad_sequence(state_list[2], batch_first=True, padding_value=1)
        state_list[3] = torch.nn.utils.rnn.pad_sequence(state_list[3], batch_first=True, padding_value=0)
        state_list[4] = torch.nn.utils.rnn.pad_sequence(state_list[4], batch_first=True, padding_value=0)

        ary_state = [np.array(state, dtype=np.float32) for state in state_list]
        ary_reward = np.array(srdan_list[1], dtype=np.float32) * reward_scale
        ary_mask = (1.0 - np.array(srdan_list[2], dtype=np.float32)) * gamma
        # different from `np.float32`
        action_list = list(map(list, zip(*(srdan_list[3]))))
        ary_action = [np.array(action, dtype=np.float32) for action in action_list]
        noise_list = list(map(list, zip(*(srdan_list[4]))))
        for k in range(len(noise_list)):
            noise_list[k] = [torch.as_tensor(i, dtype=torch.float32) for i in noise_list[k]]
            noise_list[k] = torch.nn.utils.rnn.pad_sequence(noise_list[k], batch_first=True, padding_value=0)
        ary_noise = [np.array(noise, dtype=np.float32) for noise in noise_list]
        if len(use_ratio_list) != 0:
            use_ratio = sum(use_ratio_list) / len(use_ratio_list)
        else:
            use_ratio = 0
        return ary_state, ary_action, ary_noise, ary_reward, ary_mask, use_ratio

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau,trunc_step):
        with torch.no_grad():
            buf_len = buffer[3].shape[0]
            buf_state = [torch.as_tensor(ary, device=self.device) for ary in buffer[0]]
            buf_action = [torch.as_tensor(ary, device=self.device) for ary in buffer[1]]
            buf_noise = [torch.as_tensor(ary, device=self.device) for ary in buffer[2]]

            '''get buf_r_sum, buf_logprob'''
            # bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            bs = batch_size * 2
            buf_value = [self.cri([s[i:i + bs] for s in buf_state]) for i in range(0, buf_len, bs)]

            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            ary_r_sum, ary_advantage = self.get_reward_sum(buf_len,
                                                           ary_reward=buffer[3],
                                                           ary_mask=buffer[4],
                                                           ary_value=buf_value.cpu().numpy(),
                                                           trunc_step=trunc_step)  # detach()
            buf_r_sum, buf_advantage = [torch.as_tensor(ary, device=self.device)
                                        for ary in (ary_r_sum, ary_advantage)]
            buf_advantage = (buf_advantage - buf_advantage.mean()
                             ) / (buf_advantage.std() + 1e-5)
            del buf_noise, buffer[:], ary_r_sum, ary_advantage

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = logprob = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(
                batch_size,), requires_grad=False, device=self.device)

            state = [s[indices] for s in buf_state]
            action = [act[indices] for act in buf_action]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob, obj_entropy = self.act.get_logprob_entropy(
                state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * \
                         ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor)

            # critic network predicts the reward_sum (Q value) of state
            value = self.cri(state).squeeze(-1)
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optim, obj_critic)

        return obj_critic.item(), obj_actor.item(), logprob.mean().item()  # logging_tuple

    @staticmethod
    def get_reward_sum_raw(buf_len, ary_reward, ary_mask, ary_value):
        ary_r_sum = np.empty(buf_len, dtype=np.float32)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            ary_r_sum[i] = ary_reward[i] + ary_mask[i] * pre_r_sum
            pre_r_sum = ary_r_sum[i]

        ary_advantage = ary_r_sum - (ary_mask * ary_value[:, 0])
        return ary_r_sum, ary_advantage

    def get_reward_sum_gae_old(self, buf_len, ary_reward, ary_mask, ary_value,trunc_step=50):
        ary_r_sum = np.empty(buf_len, dtype=np.float32)  # old policy value
        ary_advantage = np.empty(buf_len, dtype=np.float32)  # advantage value

        pre_r_sum = 0
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            ary_r_sum[i] = ary_reward[i] + ary_mask[i] * pre_r_sum
            pre_r_sum = ary_r_sum[i]

            ary_advantage[i] = ary_reward[i] + ary_mask[i] * \
                               (pre_advantage - ary_value[i])  # fix a bug here
            pre_advantage = ary_value[i] + \
                            ary_advantage[i] * self.lambda_gae_adv
        return ary_r_sum, ary_advantage

    def get_reward_sum_gae(self,buf_len,ary_reward,ary_mask,ary_value,trunc_step=20):
        ary_r_sum = np.empty(buf_len, dtype=np.float32)  # old policy value
        ary_advantage = np.empty(buf_len, dtype=np.float32)  # advantage value

        pre_r_sum = 0
        pre_advantage = 0
        step = 0
        factor = (ary_mask[-2] * self.lambda_gae_adv) ** (trunc_step-1)
        factor_gamma = ary_mask[-2] ** (trunc_step-1)
        for i in range(buf_len-1,-1,-1):
            step += 1

            if ary_mask[i] == 0:
                step = 0

            ary_r_sum[i] = ary_reward[i] + ary_mask[i] * pre_r_sum
            pre_r_sum = ary_r_sum[i]

            # if step < trunc_step:
            #     ary_r_sum[i] = ary_reward[i] + ary_mask[i] * pre_r_sum
            #     if step == trunc_step-1:
            #         pre_r_sum = ary_r_sum[i] - factor_gamma * ary_reward[i+trunc_step-1]
            #     else:
            #         pre_r_sum = ary_r_sum[i]
            #
            # else:
            #     ary_r_sum[i] = ary_reward[i] + ary_mask[i] * pre_r_sum
            #     pre_r_sum = ary_r_sum[i] - factor_gamma * ary_reward[i+trunc_step-1]

            if i + trunc_step >= buf_len:
                value = 0
            else:
                value = ary_value[i + trunc_step]
            if step < trunc_step:
                ary_advantage[i] = ary_reward[i] + ary_mask[i] * pre_advantage - ary_value[i]  # fix a bug here
                if step == trunc_step-1:

                    pre_advantage = ary_advantage[i] - factor \
                                * (ary_reward[i+trunc_step-1] + ary_mask[i+trunc_step-1] * value-ary_value[i+trunc_step-1])
                else:
                    pre_advantage = ary_value[i] + ary_advantage[i] * self.lambda_gae_adv
            else:
                ary_advantage[i] = ary_reward[i] + ary_mask[i] * ary_value[i+1] - ary_value[i] + ary_mask[i] * self.lambda_gae_adv * pre_advantage
                pre_advantage = ary_advantage[i] - factor \
                                * (ary_reward[i+trunc_step-1] + ary_mask[i+trunc_step-1] * value-ary_value[i+trunc_step-1])
        return ary_r_sum, ary_advantage


    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
