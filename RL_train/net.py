"""[ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)"""
# resnet 版本
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 残差模块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)  # 对应2，3
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class QConv(nn.Module):  # 卷积网络 多输入的版本
    def __init__(self, action_dim, state1, init_dim, hide_dim):  # 通道数默认3，网络层数16,输出数
        super().__init__()

        self.net_state = ResNet(BasicBlock, [2, 2, 2, 2])  # 输出应该是batch，512

        # state的
        self.fc = nn.Sequential(nn.Linear(state1, 64), nn.Hardswish(),
                                nn.Linear(64, 128), )
        # 代替mix
        self.mix = nn.Sequential(nn.Linear(640, 128), nn.Hardswish(),
                                 nn.Linear(128, action_dim), )

    def forward(self, x):
        x1 = x["birdeye"]
        x1 = self.net_state(x1)

        x2 = x["state"]
        x2 = self.fc(x2)  # batch*128
        if x2.ndim < 2:
            x2 = torch.unsqueeze(x2, 0)

        t_tmp = torch.cat((x1, x2), dim=1)  # 拼接  # batch*640
        return self.mix(t_tmp)


class QConvDuel(nn.Module):  # Dueling DQN
    def __init__(self, h, w, action_dim, state1, init_dim, hide_dim):  # 图片高度，宽度，通道数默认3，网络层数16,输出数
        super().__init__()
        self.net_state = ResNet(BasicBlock, [2, 2, 2, 2])  # 输出应该是batch，512

        # state的
        self.fc = nn.Sequential(nn.Linear(state1, 64), nn.Hardswish(),
                                nn.Linear(64, 128), )
        # 代替mix
        self.net_val = nn.Sequential(nn.Linear(640, 128), nn.Hardswish(),
                                     nn.Linear(128, 1), )  # Q value
        self.net_adv = nn.Sequential(nn.Linear(640, 128), nn.Hardswish(),
                                     nn.Linear(128, action_dim), )  # advantage function value 1

    def forward(self, x):
        x1 = x["birdeye"]
        x1 = self.net_state(x1)

        x2 = x["state"]
        x2 = self.fc(x2)  # batch*128
        if x2.ndim < 2:
            x2 = torch.unsqueeze(x2, 0)

        t_tmp = torch.cat((x1, x2), dim=1)  # 拼接  # batch*640

        q_val = self.net_val(t_tmp)
        q_adv = self.net_adv(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # dueling Q value


class QConvTwin(nn.Module):  # Double DQN
    def __init__(self, action_dim, state1, init_dim, hide_dim):
        super().__init__()
        self.net_state = ResNet(BasicBlock, [2, 2, 2, 2])
        # state的
        self.fc = nn.Sequential(nn.Linear(state1, 64), nn.ReLU(),
                                nn.Linear(64, 128), )  # q1 value
        # 代替mix
        self.net_q1 = nn.Sequential(nn.Linear(640, 128), nn.Hardswish(),
                                    nn.Linear(128, action_dim), )  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(640, 128), nn.Hardswish(),
                                    nn.Linear(128, action_dim), )  # q2 value

    def forward(self, state):
        x1 = state["birdeye"]
        x1 = self.net_state(x1)  # batch*512

        x2 = state["state"]
        x2 = self.fc(x2)  # batch*128

        if x2.ndim < 2:
            x2 = torch.unsqueeze(x2, 0)

        t_tmp = torch.cat((x1, x2), dim=1)  # 拼接  # batch*640
        return self.net_q1(t_tmp)  # one Q value

    def get_q1_q2(self, state):
        x1 = state["birdeye"]
        x1 = self.net_state(x1)

        x2 = state["state"]
        x2 = self.fc(x2)  # batch*128

        if x2.ndim < 2:
            x2 = torch.unsqueeze(x2, 0)

        t_tmp = torch.cat((x1, x2), dim=1)  # 拼接  # batch*640
        q1 = self.net_q1(t_tmp)
        q2 = self.net_q2(t_tmp)
        return q1, q2  # two Q values


class QConvTwinDuel(nn.Module):  # D3QN: Dueling Double DQN 加了优势和q网络
    def __init__(self, h, w, action_dim, init_dim, hide_dim):  # 图片高度，宽度，通道数默认3，网络层数16,输出数
        super().__init__()
        self.net_state = ResNet(BasicBlock, [2, 2, 2, 2])
        # state的
        self.fc = nn.Sequential(nn.Linear(state1, 64), nn.ReLU(),
                                nn.Linear(64, 128), )  # q1 value

        self.net_state = nn.Sequential(nn.Linear(640, 128), nn.ReLU(),
                                       nn.Linear(128, action_dim), nn.ReLU())
        self.net_adv1 = nn.Sequential(nn.Linear(640, 128), nn.Hardswish(),
                                      nn.Linear(128, 1))  # q1 value
        self.net_adv2 = nn.Sequential(nn.Linear(640, 128), nn.Hardswish(),
                                      nn.Linear(128, 1))  # q2 value
        self.net_val1 = nn.Sequential(nn.Linear(640, 128), nn.Hardswish(),
                                      nn.Linear(128, action_dim))  # advantage function value 1
        self.net_val2 = nn.Sequential(nn.Linear(640, 128), nn.Hardswish(),
                                      nn.Linear(128, action_dim))  # advantage function value 1

    def forward(self, state):
        x1 = state["birdeye"]
        x1 = self.net_state(x1)  # batch*512

        x2 = state["state"]
        x2 = self.fc(x2)  # batch*128

        if x2.ndim < 2:
            x2 = torch.unsqueeze(x2, 0)

        t_tmp = torch.cat((x1, x2), dim=1)  # 拼接  # batch*640
        q_val = self.net_val1(t_tmp)
        q_adv = self.net_adv1(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # one dueling Q value

    def get_q1_q2(self, state):
        x1 = state["birdeye"]
        x1 = self.net_state(x1)  # batch*512

        x2 = state["state"]
        x2 = self.fc(x2)  # batch*128

        if x2.ndim < 2:
            x2 = torch.unsqueeze(x2, 0)

        tmp = torch.cat((x1, x2), dim=1)  # 拼接  # batch*640

        val1 = self.net_val1(tmp)
        adv1 = self.net_adv1(tmp)
        q1 = val1 + adv1 - adv1.mean(dim=1, keepdim=True)

        val2 = self.net_val2(tmp)
        adv2 = self.net_adv2(tmp)
        q2 = val2 + adv2 - adv2.mean(dim=1, keepdim=True)
        return q1, q2  # two dueling Q values


'''Policy Network (Actor)'''


class Actor(nn.Module):  # DPG: Deterministic Policy Gradient
    def __init__(self, h, w, action_dim, init_dim, hide_dim):  # 图片高度，宽度，通道数默认3，网络层数16,输出数
        super().__init__()
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = conv_w * conv_h * hide_dim * 2  # 算出展开大小

        self.net = nn.Sequential(nn.Conv2d(init_dim, hide_dim, kernel_size=5, stride=2), nn.ReLU(),
                                 nn.Conv2d(hide_dim, hide_dim * 2, kernel_size=5, stride=2), nn.ReLU(),
                                 nn.Conv2d(hide_dim * 2, hide_dim * 2, kernel_size=5, stride=2), nn.ReLU(),
                                 nn.Linear(linear_input_size, action_dim))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        if isinstance(state_dim, int):
            if if_use_dn:
                nn_dense = DenseNet(mid_dim // 2)
                inp_dim = nn_dense.inp_dim
                out_dim = nn_dense.out_dim

                self.net = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
                                         nn_dense,
                                         nn.Linear(out_dim, action_dim), )
            else:
                self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                         nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                         nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                         nn.Linear(mid_dim, action_dim), )
        else:
            def set_dim(i):
                return int(12 * 1.5 ** i)

            self.net = nn.Sequential(NnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
                                     nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
                                     nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.ReLU(),
                                     nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.ReLU(),
                                     nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.ReLU(),
                                     nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.ReLU(),
                                     nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.ReLU(),
                                     NnReshape(-1),
                                     nn.Linear(set_dim(5), mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, action_dim), )

        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

        layer_norm(self.net[-1], std=0.1)  # output layer for action

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get_action_noise(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def compute_logprob(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()
        delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
        logprob = -(self.a_std_log + self.sqrt_2pi_log + delta)
        return logprob.sum(1)


class ActorSAC(nn.Module):
    def __init__(self, action_dim, state1, init_dim, hide_dim):  # 通道数默认3，网络层数16,输出数
        super().__init__()

        self.net_state = ResNet(BasicBlock, [2, 2, 2, 2], )

        # state的
        self.fc = nn.Sequential(nn.Linear(state1, 64), nn.Hardswish(),
                                nn.Linear(64, 128), )

        # 代替mix
        self.net_a_avg = nn.Sequential(nn.Linear(640, 128), nn.Hardswish(),
                                       nn.Linear(128, action_dim))  # the average of action
        self.net_a_std = nn.Sequential(nn.Linear(640, 128), nn.Hardswish(),
                                       nn.Linear(128, action_dim))  # the log_std of action

        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.soft_plus = nn.Softplus()

    def forward(self, x):
        x1 = x["birdeye"]
        x1 = self.net_state(x1)  # batch*512

        x2 = x["state"]
        x2 = self.fc(x2)  # batch*128
        if x2.ndim < 2:
            x2 = torch.unsqueeze(x2, 0)

        tmp = torch.cat((x1, x2), dim=1)  # batch*n_action

        return self.net_a_avg(tmp).tanh()  # action

    def get_action(self, x):
        x1 = x["birdeye"]
        x1 = self.net_state(x1)  # batch*512

        x2 = x["state"]
        x2 = self.fc(x2)  # batch*128
        if x2.ndim < 2:
            x2 = torch.unsqueeze(x2, 0)

        t_tmp = torch.cat((x1, x2), dim=1)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
        a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get_action_logprob(self, x):
        x1 = x["birdeye"]
        x1 = self.net_state(x1)  # batch*512

        x2 = x["state"]
        x2 = self.fc(x2)  # batch*128
        if x2.ndim < 2:
            x2 = torch.unsqueeze(x2, 0)

        t_tmp = torch.cat((x1, x2), dim=1)

        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
        a_std = a_std_log.exp()

        """add noise to action in stochastic policy"""
        noise = torch.randn_like(a_avg, requires_grad=True)
        action = a_avg + a_std * noise
        # Can only use above code instead of below, because the tensor need gradients here.
        # a_noise = torch.normal(a_avg, a_std, requires_grad=True)

        '''compute logprob according to mean and std of action (stochastic policy)'''
        # # self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        # logprob = a_std_log + self.sqrt_2pi_log + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        # different from above (gradient)
        noise = torch.randn_like(a_avg, requires_grad=True)
        a_noise = a_avg + a_std * noise
        log_prob = a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        # same as below:
        # from torch.distributions.normal import Normal
        # logprob_noise = Normal(a_avg, a_std).logprob(a_noise)
        # logprob = logprob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()
        # same as below:
        # a_delta = (a_avg - a_noise).pow(2) /(2*a_std.pow(2))
        # logprob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))
        # logprob = logprob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()

        log_prob += (np.log(2.) - a_noise - self.soft_plus(-2. * a_noise)) * 2.  # better than below
        # same as below:
        # epsilon = 1e-6
        # logprob = logprob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log()
        return a_noise.tanh(), log_prob.sum(1, keepdim=True)


'''Value Network (Critic)'''


class Critic(nn.Module):  # 可以看出dqn函数，求q值
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # Q value


class CriticAdv(nn.Module):
    def __init__(self, state_dim, mid_dim, if_use_dn=False):
        super().__init__()
        if isinstance(state_dim, int):
            if if_use_dn:
                nn_dense = DenseNet(mid_dim // 2)
                inp_dim = nn_dense.inp_dim
                out_dim = nn_dense.out_dim

                self.net = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
                                         nn_dense,
                                         nn.Linear(out_dim, 1), )
            else:
                self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                         nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                         nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                         nn.Linear(mid_dim, 1), )

            self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim), nn.ReLU(),  # nn.Hardswish(),
                                     nn.Linear(mid_dim, 1))
        else:
            def set_dim(i):
                return int(12 * 1.5 ** i)

            self.net = nn.Sequential(NnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
                                     nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
                                     nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.ReLU(),
                                     nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.ReLU(),
                                     nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.ReLU(),
                                     nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.ReLU(),
                                     nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.ReLU(),
                                     NnReshape(-1),
                                     nn.Linear(set_dim(5), mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, 1))

        layer_norm(self.net[-1], std=0.5)  # output layer for Q value

    def forward(self, state):
        return self.net(state)  # Q value


class CriticTwin(nn.Module):
    def __init__(self, action_dim, state1, init_dim, hide_dim, if_use_dn=False):  # 图片高度，宽度，通道数默认3，网络层数16,输出数
        super().__init__()

        self.net_state = ResNet(BasicBlock, [2, 2, 2, 2])
        # state的 这里加了动作
        self.fc = nn.Sequential(nn.Linear(state1 + action_dim, 64), nn.ReLU(),
                                nn.Linear(64, 128), )

        # 代替mix
        self.net_q1 = nn.Sequential(nn.Linear(640, 128), nn.Hardswish(),
                                    nn.Linear(128, 1))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(640, 128), nn.Hardswish(),
                                    nn.Linear(128, 1))  # q2 value

    def forward(self, x, action):
        x1 = x["birdeye"]
        x1 = self.net_state(x1)  # batch*512

        x2 = x["state"]
        x2 = torch.cat((x2, action), dim=1)
        x2 = self.fc(x2)  # batch*128
        if x2.ndim < 2:
            x2 = torch.unsqueeze(x2, 0)

        tmp = torch.cat((x1, x2), dim=1)  # batch*n_action

        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, x, action):
        x1 = x["birdeye"]
        x1 = self.net_state(x1)  # batch*512

        x2 = x["state"]
        x2 = torch.cat((x2, action), dim=1)
        x2 = self.fc(x2)  # batch*128
        if x2.ndim < 2:
            x2 = torch.unsqueeze(x2, 0)

        tmp = torch.cat((x1, x2), dim=1)  # batch*n_action

        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


"""utils"""


class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):  # x1.shape==(-1, lay_dim*1)
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3  # x2.shape==(-1, lay_dim*4)


class ConcatNet(nn.Module):  # concatenate
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim, lay_dim), nn.ReLU(),
                                    nn.Linear(lay_dim, lay_dim), nn.Hardswish(), )
        self.dense2 = nn.Sequential(nn.Linear(lay_dim, lay_dim), nn.ReLU(),
                                    nn.Linear(lay_dim, lay_dim), nn.Hardswish(), )
        self.dense3 = nn.Sequential(nn.Linear(lay_dim, lay_dim), nn.ReLU(),
                                    nn.Linear(lay_dim, lay_dim), nn.Hardswish(), )
        self.dense4 = nn.Sequential(nn.Linear(lay_dim, lay_dim), nn.ReLU(),
                                    nn.Linear(lay_dim, lay_dim), nn.Hardswish(), )
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x0):
        x1 = self.dense1(x0)
        x2 = self.dense2(x0)
        x3 = self.dense3(x0)
        x4 = self.dense4(x0)

        return torch.cat((x1, x2, x3, x4), dim=1)


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
