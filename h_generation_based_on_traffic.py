import numpy as np
# import matplotlib.pyplot as plt
import math
from gym import spaces  # , logger


class Traffic:
    def __init__(self, n_uveh, n_bveh):
        # np.random.seed(123)
        self.n_uveh = n_uveh  # number of user vehicles.
        self.n_bveh = n_bveh  # number of base vehicles.
        # 路段参数
        self.road_length = 500  # length of road within the range of the AP; denoted as L.
        self.lane_width = 5  # width of each lane; there are 2 lanes in this model
        self.n_lanes = 2  # number of lanes
        self.lane1 = self.lane_width / 2  # y coordinate of the 1st lane
        self.lane2 = self.lane_width * 3 / 2  # y coordinate of the 2nd lane
        # 车辆移动模型参数
        self.alpha_veh = 0.9  # alpha parameter of the Gauss-Markov Mobility Model
        self.mu_veh = 40 / 3.6  # mu is the asymptotic mean of vn, when n approaches infinity
        self.sigma_veh = 2  # sigma is the asymptotic standard deviation of vn when n approaches infinity
        # ap参数
        self.ap_position = (self.road_length / 2, 0)  # AP position [x, y]
        self.ap_height = 30  # AP antenna height m
        self.ap_gain = 10  # AP 天线增益
        self.ap_f = 3e9  # ap 计算机性能
        # 通信参数
        self.veh_gain = 2  # 车辆天线增益
        self.fc = 915e6  # carrier frequency
        self.de = 2.8  # pass loss exponent2.8
        self.bandwidth = 2e6  # B=2MHz
        self.noise = 1e-10  # receiver noise power N=10^-10
        # 其他参数
        self.phi = 100  # number of cycles needed to execute a bit of input task file
        self.max_latency = 3  # 最大容许的任务时延
        # self.observ_dim = 2 * n_uveh + 2 * n_bveh
        self.observ_dim = 2 * n_uveh + 2 * n_bveh + n_bveh * n_uveh + n_uveh + 1
        self.n_step = 0  # 统计步数
        self.n_actions = (n_bveh + 1) ** n_uveh  # action空间的大小
        # 容器初始化
        self.uveh = []  # container of User vehicles.
        self.bveh = []  # container of Base vehicles.
        self.queue = []  # 任务队列，（te--任务刚开始上传,s--任务剩余文件大小,I--任务归属车辆编号,s0--任务初始大小)
        self.v2i_channel_gain = np.zeros(n_uveh)  # V2I channel gain of each User vehicles
        self.v2v_channel_gain = np.zeros((n_uveh, n_bveh))  # V2V channel gain between User i and Base j
        self.distance_ij = np.zeros((n_uveh, n_bveh))  # distance between user vehicles and base vehicles
        self.comrate_his = []  # 按照每一步，记录所有computation rate
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.observ_dim,))
        self.actions_all = np.zeros((self.n_actions, n_uveh), dtype='int_')  # action空间
        a = - np.ones((n_uveh,), dtype='int_')
        for i in range(self.n_actions):
            self.actions_all[i] = a
            num_jinwei = 0
            for j in range(-1, -n_uveh, -1):  # num_jinwei最大只能有n_uveh-1
                if a[j] == n_bveh - 1:
                    num_jinwei += 1
                else:
                    break
            if num_jinwei == 0:
                a[-1] += 1
            else:
                a[-1 - num_jinwei] += 1
                a[-1:-1 - num_jinwei:-1] = -1

    def reset(self, test=0):
        # initialize the environment, generate all cars to the *left side* of the road, as well as hi and hij
        self.uveh = []
        self.bveh = []
        self.queue = []
        self.n_step = 0
        self.comrate_his = []
        x_limit = 1 / 5  # x limitation of all new cars, ensuring new cars are on the left side of the road.
        if test:
            self.n_uveh, self.n_bveh = 4, 4
            self.uveh.append(Car(1, 1 / 5 * self.road_length, self.lane2, 0))
            self.uveh.append(Car(1, 2 / 5 * self.road_length, self.lane2, 0))
            self.uveh.append(Car(1, 3 / 5 * self.road_length, self.lane2, 0))
            self.uveh.append(Car(1, 4 / 5 * self.road_length, self.lane2, 0))
            self.bveh.append(Car(1, 1 / 5 * self.road_length, self.lane1, 0))
            self.bveh.append(Car(1, 2 / 5 * self.road_length, self.lane1, 0))
            self.bveh.append(Car(1, 3 / 5 * self.road_length, self.lane1, 0))
            self.bveh.append(Car(1, 4 / 5 * self.road_length, self.lane1, 0))
        else:
            for i in range(self.n_uveh):  # 车辆在两个车道上随机摆放
                if np.random.randint(self.n_lanes) == 0:  # select a lane from 2 lanes randomly
                    lane = self.lane1
                else:
                    lane = self.lane2
                if i % 2 == 0:  # assign a weight to each vehicle based on index
                    w = 1
                else:
                    w = 1  # 1.5
                x = np.random.uniform(0, x_limit * self.road_length)  # select x coordinate for a vehicle
                self.uveh.append(Car(w=w, x=x, y=lane, v=0))  # add a User vehicle
            for j in range(self.n_bveh):
                if np.random.randint(self.n_lanes) == 0:  # select a lane from 2 lanes randomly
                    lane = self.lane1
                else:
                    lane = self.lane2
                x = np.random.uniform(0, x_limit * self.road_length)  # select x coordinate for a vehicle
                self.bveh.append(Car(w=0, x=x, y=lane, v=0))  # add a Base vehicle
        self.renew_v2v_channel_gain(test)
        self.renew_v2i_channel_gain(test)
        observ = self.get_observation(test)
        return observ  # 返回初始observation

    def renew_traffic(self):  # renew the traffic state every Ts including v, x
        for i in range(self.n_uveh):  # Users
            self.uveh[i].v = np.max([0, self.alpha_veh * self.uveh[i].v + (1 - self.alpha_veh) * self.mu_veh + (
                    np.sqrt(
                        1 - self.alpha_veh ** 2) * np.random.randn() * self.sigma_veh)])  # renew v according to markov model
            self.uveh[i].x += self.uveh[i].v  # renew x according to v
        for j in range(self.n_bveh):  # Bases
            self.bveh[j].v = np.max([0, self.alpha_veh * self.bveh[j].v + (1 - self.alpha_veh) * self.mu_veh + (
                    np.sqrt(
                        1 - self.alpha_veh ** 2) * np.random.randn() * self.sigma_veh)])  # renew v according to markov model
            self.bveh[j].x += self.bveh[j].v  # renew x according to v

    def renew_v2i_channel_gain(self, test=0):  # renew the channel gain of all v2i channels
        self.v2i_channel_gain = np.zeros(self.n_uveh)  # V2I channel gain of each User vehicles
        for i in range(self.n_uveh):
            dx = abs(self.uveh[i].x - self.ap_position[0])
            dy = abs(self.uveh[i].y - self.ap_position[1])
            dz = self.ap_height
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            average = self.veh_gain * self.ap_gain * (3e8 / (4 * math.pi * self.fc * distance)) ** self.de
            if test:
                self.v2i_channel_gain[i] = average
            else:
                self.v2i_channel_gain[i] = average * np.random.exponential()

    def renew_v2v_channel_gain(self, test=0):
        # renew the channel gain of all v2v channels and distance from user vehicles to base vehicles
        self.v2v_channel_gain = np.zeros((self.n_uveh, self.n_bveh))
        # V2V channel gain between User i and Base j
        for i in range(self.n_uveh):
            for j in range(self.n_bveh):
                dx = abs(self.uveh[i].x - self.bveh[j].x)
                dy = abs(self.uveh[i].y - self.bveh[j].y)
                distance = np.max([math.hypot(dx, dy), 15])  # 返回欧几里德范数 sqrt(x*x + y*y)
                self.distance_ij[i][j] = math.hypot(dx, dy)
                average = self.veh_gain * self.veh_gain * (3e8 / (4 * math.pi * self.fc * distance)) ** self.de
                if test:
                    self.v2v_channel_gain[i][j] = average
                else:
                    self.v2v_channel_gain[i][j] = average * np.random.exponential()

    def get_observation(self, test=0):  # 得到目前的环境observation值
        if test:
            gain_x, gain_h, gain_q = 1, 1, 1
        else:
            gain_x, gain_h, gain_q = 0.05, 8e8, 3e-8  # 为使状态变量接近1
        queue_size = sum([i[1] for i in self.queue]) * gain_q
        queue_size = np.array([queue_size])
        users_location_x = np.array([i.x for i in self.uveh]) * gain_x
        users_location_y = np.array([i.y for i in self.uveh])
        bases_location_x = np.array([i.x for i in self.bveh]) * gain_x
        bases_location_y = np.array([i.y for i in self.bveh])
        hi = self.v2i_channel_gain * gain_h
        hij = self.v2v_channel_gain.ravel() * gain_h
        if test:
            observ = {'queue_size': queue_size[0], 'users_location_x': users_location_x,
                      'users_location_y': users_location_y,
                      'bases_location_x': bases_location_x, 'bases_location_y': bases_location_y, 'hi': hi, 'hij': hij}
        else:
            observ = np.concatenate(
                (queue_size, users_location_x, users_location_y, bases_location_x, bases_location_y, hi, hij))
            # observ = np.concatenate((users_location_x, users_location_y, bases_location_x, bases_location_y))
        return observ

    def ci(self, i):  # 返回用户车辆i到AP的信道容量Ci
        return self.bandwidth * np.log2(1 + self.uveh[i].p * self.v2i_channel_gain[i] / self.noise)

    def cij(self, i, j):  # 返回用户车辆i到基站车辆j的信道容量Cij
        return self.bandwidth * np.log2(1 + self.uveh[i].p * self.v2v_channel_gain[i, j] / self.noise)

    def step(self, action, test=0):  # action目前仅仅是index 如果test=1，缩小reward方便调试
        comrate_v2v = np.zeros(self.n_uveh)  # 记录v2v车辆在本帧的计算率，序号为v2i车辆的元素必为0
        a = self.actions_all[action]  # decode action
        idle_time_ap = 0  # idle time of AP in each step
        abandon = 0  # number of tasks abandoned in each step
        out_of_range = 0  # number of cases when User was out of Ap's range
        a_v2i = np.nonzero(a == -1)[0]  # all indices of vehicles in v2i mode
        a_v2v = np.nonzero(a >= 0)[0]  # all indices of vehicles in v2v mode
        for i in self.bveh:  # 计算完成，解除对基站车辆的占用
            i.choice = -2
        for i in self.uveh:
            i.choice = -2
        if np.size(a_v2v):
            for i in a_v2v:  # 得到所有v2v车辆对基站车辆的选择# 得到计算速率
                j = a[i]  # user[i]必是v2v
                if self.bveh[j].choice == -2:  # 对基站车辆来说，序号（大于等于0）代表与其配对的用户车辆,-2代表尚未被选择
                    self.bveh[j].choice = i
                    self.uveh[i].choice = j
                    comrate_v2v[i] = self.cij(i, j) * self.bveh[int(j)].f / (
                            self.cij(i, j) * self.phi + self.bveh[int(j)].f)
                else:
                    # print('被占用的base序号为：', j, '  其选择为:', self.bveh[j].choice)
                    comrate_v2v[i] = 0
        comrate_sum_v2v = sum(comrate_v2v)
        #  下面开始v2i模式(计算部件工作）
        comrate_sum_v2i = 0
        t_current = 0
        while True:
            if self.queue:  # 目前的时间为帧头时间+t_current
                task = self.queue[0]  # 队头任务,深复制
                t_task = task[1] * self.phi / self.ap_f  # 任务完成所需时间
                t_latency = t_current + t_task + self.n_step - task[0]  # 任务的延时
                if t_latency > self.max_latency:  # 任务完成将超过最大时延
                    abandon += 1
                    del self.queue[0]  # 放弃该任务，弹出队列
                elif t_current + t_task > 1:  # 任务无法在该帧完成
                    t_left = 1 - t_current  # 帧里剩余时间
                    s = self.ap_f / self.phi * t_left  # 剩余时间内可执行的任务量
                    task[1] -= s
                    break
                else:
                    del self.queue[0]  # finish该任务，弹出队列
                    t_current += t_task  # 推进现在时间
                    if self.uveh[task[2]].x <= self.road_length:  # 如果不在范围内，则不增加总计算率。
                        comrate_sum_v2i += task[3]  # 增加总计算率
                    else:
                        out_of_range += 1
            else:
                idle_time_ap += 1 - t_current
                break  # 所有队列的积压任务在本帧计算完毕，还剩一定空闲时间。
        if np.size(a_v2i):  # 通信部件的工作：上传任务文件并加入队列,此时在帧的*开头*
            for i in a_v2i:
                self.uveh[i].choice = -1
                # (te--任务刚开始上传的时间,s--任务剩余文件大小,I--任务归属车辆编号,s0--任务初始大小)
                if self.uveh[i].x <= self.road_length:  # 如果不在范围内，则不允许加入任务队列。
                    self.queue.append([self.n_step, self.ci(i), i, self.ci(i)])  # 任务信息进入队列
                else:
                    out_of_range += 1
        self.n_step += 1
        # bias = 0
        # for i in range(self.n_uveh):
        #     bias += self.cij(i, i) * self.bveh[i].f / (self.cij(i, i) * self.phi + self.bveh[i].f)
        comrate_sum = comrate_sum_v2i + comrate_sum_v2v
        bias = 2.8e7
        reward = (comrate_sum - bias) / bias
        if not test:
            self.renew_traffic()  # 刷新环境
        self.renew_v2v_channel_gain(test)
        self.renew_v2i_channel_gain(test)
        observ = self.get_observation(test)  # 生成环境值
        users_location_x = [i.x for i in self.uveh]
        done = all(map(lambda x: x > self.road_length, users_location_x))  # 检查车辆位置是否超过AP覆盖范围

        v2i_number = sum(a == -1)
        a_set = set(a)  # 计算竞争基站车辆的数目
        if -1 in a_set:
            a_set.remove(-1)
        conflict = 0
        for i in a_set:
            num = sum(a == i)
            if num > 1:
                conflict += num - 1

        info = {'comrate': comrate_sum, 'v2i_number': v2i_number,
                'conflict': conflict, 'idle_time_ap': idle_time_ap, 'abandon': abandon,
                'out_of_range': out_of_range,}
        # if done:
        #     reward = np.mean(self.comrate_his)
        # else:
        #     reward = 0
        return observ, reward, done, info


class Car:
    def __init__(self, w, x, y, v):
        self.w = w  # weight
        self.x = x  # x_coordinate m
        self.y = y  # y_coordinate m
        self.v = v  # current velocity m/s
        self.p = 10  # signal transmission power
        self.choice = -2  # 对用户车辆来说，-1代表采用v2i，大于等于0代表v2v里面，选择卸载的目标基站车辆序号
        # 对基站车辆来说，序号（大于等于0）代表与其配对的用户车辆,-2代表尚未被选择
        self.computation_rate = 0  # 仅对用户车辆有效
        self.f = 8e8  # CPU转速，cycles per second， 仅对基站车辆有效


if __name__ == "__main__":
    test = Traffic(4, 4)  # 5users, 2bases
    steps = 0
    observation = test.reset(test=0)
    observation, reward, done, _ = test.step(604, test=0)
    for i in range(20):
        observation, reward, done, _ = test.step(0, test=0)
    observation, reward, done, _ = test.step(0, test=0)
    observation, reward, done, _ = test.step(0, test=0)
    observation, reward, done, _ = test.step(156, test=0)
    observation, reward, done, _ = test.step(156, test=0)
    observation, reward, done, _ = test.step(155, test=0)
    action_his = []
    observation_his = [observation]
    reward_his = []
    reward_sum = 0
    reward_all = np.zeros(243)  # 存储每个observation里面，所有可能的reward
    x_his = []
    optimal = []
    optimal_choice = []
    optimal_reward = []
    actions_all = np.zeros((243, 5), dtype='int_')
    tau_all = np.zeros(243)
    a = np.array([-1, -1, -1, -1, -1], dtype='int_')
    # for i in range(243):
    #     actions_all[i] = a
    #     if a[-1] != 1:  # ****0
    #         a[-1] += 1
    #     elif a[-2] != 1:  # ***01
    #         a[-2], a[-1] = a[-2] + 1, -1
    #     elif a[-3] != 1:  # **011
    #         a[-3], a[-2], a[-1] = a[-3] + 1, -1, -1
    #     elif a[-4] != 1:  # *0111
    #         a[-4], a[-3], a[-2], a[-1] = a[-4] + 1, -1, -1, -1
    #     else:  # *1111
    #         a[:] = a[0] + 1, -1, -1, -1, -1
    # # print(actions_all)
    # for i in range(32):  # 生成所有可能的决策
    #     s = bin(i).split('b')[1]
    #     s = s[::-1]
    #     for j in enumerate(s):
    #         actions_all[i][-1 - j[0]] = int(j[1])
    # episode = 0
    # test.env_init(test=0)
    # while episode < 20:  # 如果每步都取最优，reward能有多少
    #     for j in range(243):  # 生成所有可能的决策
    #         a = actions_all[j]
    #         r = test.evaluate(a, test=0)
    #         reward_all[j] = r
    #     reward_max = np.argmin(reward_all)  # 求出最优选择index；  actions_all[reward_max]为最优决策
    #     a = actions_all[reward_max]
    #     # a = np.random.randint(-1, 2, size=5)  # 随机决策
    #     observation, reward, done = test.step(a, test=0)
    #     steps += 1
    #     reward_sum += reward
    #     if done:
    #         episode += 1
    #         print('steps=',steps)
    #         steps = 0
    #         # print("Reward for this episode was: {:.2e}".format(reward_sum))
    #         reward_his.append(reward_sum)
    #         test.env_init(test=0)
    #         reward_sum = 0
    # print('mean={:.2e}'.format(np.mean(reward_his)))
'''
    test.reset(test=0)
    for i in range(1000):
        reward_action = []
        for j in range(243):  # 生成所有可能的决策
            r = round(test.evaluate(j, test=1) - test.evaluate(0, test=1), 5)  # 尝试偏置的大小
            tau_all[j] = test.evaluate(j, test=2)
            reward_all[j] = r
            a = test.actions_all[j]
            reward_action.append((r, a))
        small_to_large = np.argsort(reward_all)  # 最差选择到最佳选择
        actions_s2l = test.actions_all[small_to_large]
        reward_s2l = reward_all[small_to_large]
        tau_s2l = tau_all[small_to_large]

        reward_max = np.argmax(reward_all)  # 最佳选择的action index
        users_x = np.array([i.x for i in test.uveh])
        x_average = int(np.mean(users_x))  # 车辆的平均位置
        x_his.append(x_average)
        optimal_reward.append(reward_all[reward_max])  # 最佳reward的list
        optimal_choice.append(list(actions_all[reward_max]))  # 最佳选择的list
        optimal.append((x_average, reward_all[reward_max], list(actions_all[reward_max])))
        # a = np.random.randint(-1, 2, size=5)  # 随机决策
        a = np.random.randint(0, 243)
        action_his.append(a)
        observation, reward, done, _ = test.step(a, test=0)
        observation_his.append(observation)
        reward_his.append(reward)
        if i == 20:
            _ = 0
        if done:
            print("done with steps equaling ", i)
            break

    # env.actions_all[action]
    # [(round(env.evaluate(j, 1) - env.evaluate(0, 1), 5), env.actions_all[j]) for j in range(243)]
    # [(round(q_values[j] * 100, 4), env.actions_all[j], j) for j in range(243)]
    # (np.argmax(q_values), round(env.evaluate(np.argmax(q_values), 1) - env.evaluate(0, 1), 5),
    #  env.actions_all[np.argmax(q_values)])
    # (round(q_values[np.argmax(q_values)] * 100, 4), env.actions_all[np.argmax(q_values)], np.argmax(q_values))

    # plt.plot(x_his, optimal_reward)
    # plt.show()
    # users_x = [i.x for i in test.uveh]
    # users_y = [i.y for i in test.uveh]
    # bases_x = [i.x for i in test.bveh]
    # bases_y = [i.y for i in test.bveh]
    # subplot1 = plt.subplot(2, 1, 1)
    # subplot2 = plt.subplot(2, 1, 2)
    # plt.sca(subplot1)
    # plt.scatter(users_x, users_y, color='red')
    # plt.scatter(bases_x, bases_y, color="yellow")
    # plt.xlim(0, 500)
    # position1 = []
    # position2 = []
    # for i in range(30):
    #     test.renew_traffic()
    #     test.renew_v2i_channel_gain()
    #     test.renew_v2v_channel_gain()
    #     # position1.append(test.uveh[0].x)
    #     # position2.append(test.v2i_channel_gain[0])
    #     position1.append(test.uveh[0].x - test.bveh[0].x)
    #     position2.append(test.v2v_channel_gain[0][0])
    #
    # users_x = [i.x for i in test.uveh]
    # users_y = [i.y for i in test.uveh]
    # bases_x = [i.x for i in test.bveh]
    # bases_y = [i.y for i in test.bveh]
    # plt.sca(subplot2)
    # plt.scatter(users_x, users_y, color='red')
    # plt.scatter(bases_x, bases_y, color="yellow")
    # plt.xlim(0, 500)
    #
    # plt.figure()
    # subplot1 = plt.subplot(2, 1, 1)
    # subplot2 = plt.subplot(2, 1, 2)
    # plt.sca(subplot1)
    # plt.plot(position1)
    # plt.sca(subplot2)
    # plt.plot(np.log10(position2), color='red')
    # plt.show()
'''
