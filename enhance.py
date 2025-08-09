import torch

from eua import EvidentialCMeans, Degranulatuin


class Enhancer:
    def __init__(self, args):
        self.args = args

    def generate(self, train_data):
        if self.args.enhance == 'white_noise':
            x, y = train_data.get_raw()
            std_dev = torch.std(x, dim=0, keepdim=True)
            noise = torch.randn_like(x) * std_dev * self.args.white_noise_level
            return x + noise, y

        if self.args.enhance == 'eua':
            x, y = train_data.get_raw()
            # 粒化并plusu
            gra = EvidentialCMeans(self.args)
            u_base, u_plus, v_extend = gra.plus_uncertainty(x.clone())
            # 解粒
            deg = Degranulatuin(self.args)
            x_ex = deg.anti_point(x.clone(), u_plus, v_extend)
            return x_ex, y

        if self.args.enhance == 'salt_noise':
            x, y = train_data.get_raw()
            noisy_data = x.clone()
            num_samples = noisy_data.numel()
            num_noise = int(self.args.salt_noise_ratio * num_samples)

            if num_noise == 0:
                return noisy_data, y

            # 生成随机索引
            indices = torch.randperm(num_samples, device=self.args.device)[:num_noise]

            # 分割盐噪声和椒噪声的索引
            num_salt = num_noise // 2
            salt_indices = indices[:num_salt]
            pepper_indices = indices[num_salt:]

            # 计算每个特征的最大值和最小值
            max_vals, _ = torch.max(noisy_data, dim=0)  # 按特征维度计算最大值
            min_vals, _ = torch.min(noisy_data, dim=0)  # 按特征维度计算最小值

            # 处理盐噪声
            if num_salt > 0:
                # 计算每个索引对应的特征索引
                feature_indices = salt_indices % noisy_data.shape[1]
                # 获取对应特征的最大值并乘以1.1
                salt_values = max_vals[feature_indices] * 1.1
                # 应用盐噪声
                noisy_data.view(-1)[salt_indices] = salt_values

            # 处理椒噪声
            num_pepper = num_noise - num_salt
            if num_pepper > 0:
                # 计算每个索引对应的特征索引
                feature_indices = pepper_indices % noisy_data.shape[1]
                # 获取对应特征的最小值并乘以0.9，与0取较小值
                pepper_values = torch.min(min_vals[feature_indices] * 0.9)
                # 应用椒噪声
                noisy_data.view(-1)[pepper_indices] = pepper_values
            return noisy_data, y

        if self.args.enhance == 'specaugment':
            x, y = train_data.get_raw()
            # 创建数据副本，避免修改原始数据
            augmented_data = x.clone()
            num_samples, num_features = augmented_data.shape
            device = self.args.device  # 获取数据所在设备

            # 计算时间掩码大小
            time_mask_size = int(self.args.mask_ratio * num_samples)

            # 应用时间掩码 (掩盖整个样本)
            if time_mask_size > 0 and num_samples > time_mask_size:
                # 随机生成掩码起始位置
                start = torch.randint(0, num_samples - time_mask_size + 1, (1,), device=device).item()
                end = start + time_mask_size

                # 计算每个特征列的均值
                feature_means = torch.mean(augmented_data, dim=0)  # 形状: [特征数]

                # 应用时间掩码（使用向量化操作替代循环）
                augmented_data[start:end, :] = feature_means.unsqueeze(0)

            # 计算频率掩码大小
            freq_mask_size = int(self.args.mask_ratio * num_features)

            # 应用频率掩码 (掩盖整个特征)
            if freq_mask_size > 0 and num_features > freq_mask_size:
                # 随机生成掩码起始位置
                start = torch.randint(0, num_features - freq_mask_size + 1, (1,), device=device).item()
                end = start + freq_mask_size

                # 计算每个样本行的均值
                sample_means = torch.mean(augmented_data, dim=1)  # 形状: [样本数]

                # 应用频率掩码（使用向量化操作替代循环）
                augmented_data[:, start:end] = sample_means.unsqueeze(1)

            return augmented_data, y

        if self.args.enhance == 'cutout':
            x, y = train_data.get_raw()

            augmented_data = x.clone()
            num_samples, num_features = augmented_data.shape
            device = augmented_data.device  # 获取数据所在设备

            # 计算掩码大小
            sample_mask_size = max(1, int(self.args.mask_ratio * num_samples))
            feature_mask_size = max(1, int(self.args.mask_ratio * num_features))

            # 确保掩码大小不超过数据维度
            sample_mask_size = min(sample_mask_size, num_samples)
            feature_mask_size = min(feature_mask_size, num_features)

            sample_start = torch.randint(
                0, num_samples - sample_mask_size + 1, (1,), device=device
            ).item()
            feature_start = torch.randint(
                0, num_features - feature_mask_size + 1, (1,), device=device
            ).item()

            fill = torch.mean(augmented_data).to(device)

            # 应用掩码
            augmented_data[
            sample_start:sample_start + sample_mask_size,
            feature_start:feature_start + feature_mask_size
            ] = fill

            return augmented_data, y

        raise Exception('Unknown enhancement')
