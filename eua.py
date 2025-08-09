import torch

import utils


class EvidentialCMeans:
    def __init__(self, args):
        self.args = args
        self.device = utils.get_device(args.device)
        self.c = args.num_classes
        self.max_iter = 10000
        self.m = 2
        self.epsilon = 1e-6
        self.V = None  # 聚类中心
        self.U = None  # 隶属度矩阵
        self.t = None
        self.r = args.r  # 认知不确定度增加

    def get_T(self):
        if self.t is None:
            ones_array = torch.full((1, self.c), 1 / self.c)
            identity_matrix = torch.eye(self.c)
            self.t = torch.cat((ones_array, identity_matrix), dim=0)

        return self.t

    def fit(self, X):
        """
        执行证据 C 均值算法，对输入数据 X 进行聚类。

        参数:
        X (torch.Tensor): 输入数据，形状为 (n_samples, n_features)。
        """
        X = X.to(self.device)
        n_samples, n_features = X.shape
        # 随机初始化隶属度矩阵 U
        U = torch.rand(n_samples, self.c, dtype=torch.float32)
        # 确保每行的和为 1
        U = U / U.sum(dim=1, keepdim=True)

        for _ in range(self.max_iter):
            U_old = U.clone()
            # 计算新的聚类中心
            numerator = torch.matmul(U.pow(self.m).T, X)
            denominator = U.pow(self.m).sum(dim=0, keepdim=True).T
            self.V = numerator / denominator
            # 计算每个样本到每个聚类中心的距离
            dist = torch.cdist(X, self.V)
            power = 2 / (self.m - 1)
            dist_power = dist.pow(-power)
            # 更新隶属度矩阵 U
            U = dist_power / dist_power.sum(dim=1, keepdim=True)
            # 检查是否收敛
            if torch.norm(U - U_old) < self.epsilon:
                break

        self.U = U

        return self.V

    def plus_uncertainty(self, X):
        # 计算（验证对比集）每个样本到每个聚类中心的距离
        X = X.to(self.device)
        v_extend = torch.matmul(self.get_T(), self.fit(X)).to(self.device)

        dist = torch.cdist(X, v_extend)
        power = 2 / (self.m - 1)
        dist_power = dist.pow(-power)
        # 计算隶属度矩阵 U
        U = dist_power / dist_power.sum(dim=1, keepdim=True)
        u_base = U.clone()
        # 增加认知不确定性
        last_c_columns = u_base[:, 1:]
        min_values, _ = torch.min(last_c_columns, dim=1)
        adjustment = min_values * self.r
        # 更新U，第一列加上 adjustment 乘以 c
        U[:, 0] += adjustment * self.c
        U[:, 1:] -= adjustment.unsqueeze(1)
        u_plus = U.clone()

        return u_base, u_plus, v_extend


class Degranulatuin:
    def __init__(self, args):
        self.args = args
        self.device = utils.get_device(args.device)
        self.m = 2
        self.epsilon = 1e-4
        self.max_iterations = 1000
        self.c = args.num_classes

    def find_best_d(self, distances, distance_ratios):
        # 计算距离比例，以第一个距离比例为基准进行归一化
        ratios = distance_ratios / distance_ratios[0]
        break_points = []
        for i in range(len(distances)):
            bp = distances[i] / ratios[i]
            if bp > 0:
                break_points.append(bp)
        break_points = torch.sort(torch.tensor(break_points))[0]

        # 初始化最小误差为正无穷大
        min_error = float('inf')
        best_d = 0
        # 生成候选的 d 值列表，包括 0、所有分段点和正无穷
        candidate_ds = [0] + break_points.tolist() + [float('inf')]
        # 遍历所有候选的 d 值
        for d in candidate_ds:
            target_distances = d * ratios
            error = torch.sum(torch.abs(distances - target_distances))
            # 如果当前误差小于最小误差
            if error < min_error:
                min_error = error
                # 更新最优的 d 值
                best_d = d
        return best_d

    # 该函数用于更新待求点的坐标
    def update_point(self, point, all_centers, distance_ratios, u):
        # 确保所有输入在同一设备上
        point = point.to(self.device)
        all_centers = all_centers.to(self.device)
        u = u.to(self.device)
        self.distance_ratios = self.distance_ratios.to(self.device)

        # 获取点的维度
        point_dim = all_centers.shape[1]
        # 进行最大迭代次数的循环
        for iteration in range(self.max_iterations):
            # 第一步：计算待求点与各聚类中心的距离
            distances = torch.sqrt(torch.sum((all_centers - point) ** 2, dim=1))

            # 调用 find_best_d 函数，优化寻找最优的特殊聚类中心距离 d
            best_d = self.find_best_d(distances, distance_ratios)

            # 根据最优的 d 值和距离比例计算目标距离值
            target_distances = best_d * (distance_ratios / distance_ratios[0])

            # 第二步：计算各目标距离值与真实距离值之差
            distance_differences = target_distances - distances

            # 初始化坐标更新向量为全零向量
            update_vector = torch.zeros(point_dim)
            direction = self.c * u[0] * (point - all_centers[0]) / distances[0] if distances[0] > 0 else torch.zeros(
                point_dim)
            update_vector += distance_differences[0] * direction
            # 遍历所有聚类中心
            for i in range(len(all_centers) - 1):
                # 计算从当前聚类中心指向待求点的方向向量，如果距离为 0 则方向向量为全零向量
                direction = u[i + 1] * (point - all_centers[i + 1]) / distances[i + 1] if distances[
                                                                                              i + 1] > 0 else torch.zeros(
                    point_dim)
                # 根据距离差值和方向向量更新坐标更新向量
                update_vector += distance_differences[i + 1] * direction

            # 更新待求点的坐标
            point = point + update_vector

            # 判断是否收敛，通过检查坐标更新向量的范数是否小于收敛阈值
            if torch.norm(update_vector) < self.epsilon:
                break
        # 返回更新后的待求点坐标
        return point

    # 将 mass function 转换为距离之比的函数
    def mass_function_to_distance_ratios(self, mass_functions, m):
        num_centers = len(mass_functions)
        self.distance_ratios = torch.zeros(num_centers)
        for i in range(num_centers):
            self.distance_ratios[i] = (mass_functions[0] / mass_functions[i]) ** ((m - 1) / 2)
        return self.distance_ratios

    def anti_point(self, X, u_plus, v_extend):
        X = X.to(self.device)
        u_plus = u_plus.to(self.device)
        v_extend = v_extend.to(self.device)

        X_new = torch.zeros_like(X, device=self.device)

        for j in range(len(u_plus)):
            distance_ratios = self.mass_function_to_distance_ratios(u_plus[j], self.m)
            point = X[j].clone()

            point1 = self.update_point(point, v_extend, distance_ratios, u_plus[j])
            X_new[j] = point1

        return X_new
