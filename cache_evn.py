import numpy as np
import random


class CacheEnv(object):
    veh_num = 3
    action_bound = [0, 1]
    action_dim = veh_num * 2
    state_dim = veh_num * 4

    veh_num = 3
    # 每个用户请求的文件大小
    file_size_veh1 = random.randint(20, 30)
    file_size_veh2 = random.randint(20, 30)
    file_size_veh3 = random.randint(20, 30)
    # file_size_veh4 = random.randint(20, 30)
    # file_size_veh5 = random.randint(20, 30)
    file_size = np.array((file_size_veh1, file_size_veh2, file_size_veh3))

    # 每个用户的v2v传输速度，由与相邻车辆距离决定
    rate_v2v_veh1 = random.randint(2, 6)
    rate_v2v_veh2 = random.randint(2, 6)
    rate_v2v_veh3 = random.randint(2, 6)
    # rate_v2v_veh4 = random.randint(2, 6)
    # rate_v2v_veh5 = random.randint(2, 6)
    rate_v2v = np.array((rate_v2v_veh1, rate_v2v_veh2, rate_v2v_veh3))

    # 每个用户的v2r传输速度
    rate_v2r = np.array((4, 4, 4))

    # 时间槽：每步花费时间
    dt = 1.0

    def __init__(self, mode='easy'):

        self.system_info = np.zeros((self.veh_num, 5)) # 文件大小,v2v传输速度,v2r预缓存数据量,v2r预缓存数据量
        for i in range(self.veh_num):
            self.system_info[i, 0] = self.file_size[i]



    def step(self, action):
        done = False
        # action = (node1 angular v, node2 angular v)
        action = np.clip(action, *self.action_bound)
        # 划分动作维度
        action_File = np.zeros(self.veh_num)
        action_V2R = np.zeros(self.veh_num)

        for i in range(self.veh_num):
            action_File[i] = action[i]

        for i in range(self.veh_num, self.action_dim):
            action_V2R[i - self.veh_num] = action[i]
        print('各用户预缓存文件比例', action_File)
        print('各用户任务分配给V2V比例', action_File)

        # v2r, v2v的预缓存数据量大小
        cache_data_pre_file = action_File * self.system_info[:, 0] #v2r，v2v总缓存文件大小
        cache_data_pre_v2r = cache_data_pre_file * action_V2R
        cache_data_pre_v2v = cache_data_pre_file - cache_data_pre_v2r
        self.system_info[:, 1] = cache_data_pre_v2r
        self.system_info[:, 2] = cache_data_pre_v2v
        print('v2r预缓存数据量:', cache_data_pre_v2r)
        print('v2v预缓存数据量:', cache_data_pre_v2v)


        # 累计缓存数据量大小，累加
        self.system_info[:, 3] += self.system_info[:, 1]
        self.system_info[:, 4] += self.system_info[:, 2]

        pre_cache_v2r = self.system_info[:, 1]
        datalost_v2r = pre_cache_v2r - (self.rate_v2r * self.dt)
        print('v2r数据丢失：',datalost_v2r)

        pre_cache_v2v = self.system_info[:, 2]
        datalost_v2v = pre_cache_v2v - (self.rate_v2v * self.dt)
        print('v2r数据丢失：',datalost_v2v)

        r = 0.0
        # recive
        for i in range(self.veh_num):
            r += pre_cache_v2r[i]

        for i in range(self.veh_num):
            r += pre_cache_v2v[i]

        # lost
        for i in range(self.veh_num):
            r -= 2 * datalost_v2r[i]

        for i in range(self.veh_num):
            r -= 2 * datalost_v2v[i]

        # miss
        for i in range(self.veh_num):
            r -= 2 * (self.system_info[:, 0] - self.system_info[:, 1] - self.system_info[:, 2])[i]


        for i in range(self.veh_num):
            if datalost_v2r[i] < 0:
                r += 3

        for i in range(self.veh_num):
            if datalost_v2v[i] < 0:
                r += 3

        # done条件

        if sum(self.system_info[:, 3]) + sum(self.system_info[:, 4]):
            pass
            if datalost_v2r.all() < 0 and datalost_v2v.all < 0:
                done = True

        s = np.hstack([datalost_v2r, datalost_v2v, cache_data_pre_v2r, cache_data_pre_v2v])

        return s, r, done

    def reset(self):
        self.system_info[:, 1:5] = 0

        # v2r, v2v的预缓存数据量大小

        cache_data_pre_v2r = self.system_info[:, 2]
        cache_data_pre_v2v = self.system_info[:, 3]

        pre_cache_v2r = self.system_info[:, 1]
        datalost_v2r = pre_cache_v2r - (self.rate_v2r * self.dt)

        pre_cache_v2v = self.system_info[:, 2]
        datalost_v2v = pre_cache_v2v - (self.rate_v2v * self.dt)

        s = np.hstack([datalost_v2r, datalost_v2v, cache_data_pre_v2r, cache_data_pre_v2v ])

        return s


    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)






