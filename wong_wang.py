import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# The following DiffDecision function is not much different from the one in RNN_decision.py
# However, for the convenience of those who will use RTified WW directly afterwards, this function is put here anyway
class DiffDecision(Function):
    @staticmethod
    def forward(ctx, trajectory, dsdt_trajectory, dt):

        mask_1 = trajectory[:, :, 0] > 0
        mask_2 = trajectory[:, :, 1] > 0

        decision_time_1 = mask_1.float().argmax(dim=0).float()
        decision_time_2 = mask_2.float().argmax(dim=0).float()

        decision_time_1[mask_1.sum(dim=0) == 0] = torch.inf
        decision_time_2[mask_2.sum(dim=0) == 0] = torch.inf

        decision_times = torch.where(decision_time_1 < decision_time_2, decision_time_1, -decision_time_2)
        decision_times_choice = decision_time_1 < decision_time_2
        ctx.save_for_backward(dsdt_trajectory, decision_times_choice, decision_times)
        return decision_times * dt

    @staticmethod
    def backward(ctx, grad_output):
        dsdt_trajectory, decision_times_choice, decision_times = ctx.saved_tensors
        grads = torch.zeros_like(dsdt_trajectory)

        valid_mask = (decision_times != torch.inf) & (decision_times != -torch.inf)
        batch_indices = torch.arange(decision_times.size(0)).to(decision_times.device)

        # Positive decision times
        pos_decision_indices = decision_times[valid_mask & decision_times_choice].long()
        selected_batch_indices_pos = batch_indices[valid_mask & decision_times_choice]

        grads[pos_decision_indices, selected_batch_indices_pos, 0] = -1.0 / dsdt_trajectory[
            pos_decision_indices, selected_batch_indices_pos, 0]

        # Negative decision times
        neg_decision_indices = (-decision_times[valid_mask & ~decision_times_choice]).long()
        selected_batch_indices_neg = batch_indices[valid_mask & ~decision_times_choice]

        grads[neg_decision_indices, selected_batch_indices_neg, 1] = 1.0 / dsdt_trajectory[
            neg_decision_indices, selected_batch_indices_neg, 1]

        grads = grads * grad_output.unsqueeze(0).unsqueeze(-1).expand_as(grads)
        return grads, None, None

# The following WongWang Decision BP class is pytorch implementation of the original Wong-Wang model with following modifications:
# 1. The model is implemented with RTify, so that all parameters in the model are trainable with backpropagation
# 2. The model is implemented with a mixture of Euler and implicit Euler methods, so that it can be used with larger time steps
# 3. The model is implemented with batch processing, so that it can process multiple trials at once
# 4. The model makes a decision based on the S1 and S2 values, rather than the H1 and H2 values in the original model

# For the original Wong-Wang model, please refer to the original paper:
# Wong, K. F., & Wang, X. J. (2006). A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience, 26(4), 1314-1328.


class WongWangDecisionBP(nn.Module):
    def __init__(self, dt):
        super(WongWangDecisionBP, self).__init__()
        self.a = nn.Parameter(torch.tensor(270.0), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(108.0), requires_grad=False)
        self.d = nn.Parameter(torch.tensor(0.1540), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(0.641), requires_grad=False)

        self.tau_s = nn.Parameter(torch.tensor(100.0), requires_grad=False)
        self.J11 = nn.Parameter(torch.tensor(0.2609), requires_grad=True)
        self.J12 = nn.Parameter(torch.tensor(0.0497), requires_grad=True)
        self.J21 = nn.Parameter(torch.tensor(0.0497), requires_grad=True)
        self.J22 = nn.Parameter(torch.tensor(0.2609), requires_grad=True)
        self.J_ext = nn.Parameter(torch.tensor(0.0156), requires_grad=True)
        self.I_0 = nn.Parameter(torch.tensor(0.3255), requires_grad=True)
        self.noise_ampa = nn.Parameter(torch.tensor(0.02), requires_grad=True)
        self.tau_ampa = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.motor_delay = nn.Parameter(torch.tensor(0.1), requires_grad=True)

        self.dt = dt
        self.time_steps = int(2500 / self.dt)
        self.t_stimulus = int(2000 / self.dt)

    def forward(self, input_signal):
        batch_size = input_signal.shape[0]
        device = input_signal.device

        if input_signal.dim() == 3:
            assert input_signal.shape[1] == self.t_stimulus

        s_1 = torch.ones(batch_size, 1, requires_grad=False, device=device) / 10.0
        s_2 = torch.ones(batch_size, 1, requires_grad=False, device=device) / 10.0
        I_noise_1 = torch.randn(batch_size, 1, requires_grad=False, device=device) * self.noise_ampa
        I_noise_2 = torch.randn(batch_size, 1, requires_grad=False, device=device) * self.noise_ampa

        trajectory = torch.zeros((self.time_steps, batch_size, 2), device=device)
        dsdt_trajectory = torch.zeros((self.time_steps, batch_size, 2), device=device)

        for t in range(self.time_steps):
            if t < self.t_stimulus:
                I_1 = self.J_ext * (1 + input_signal / 100.0) if input_signal.dim() == 2 else self.J_ext * (1 + input_signal[:, t].unsqueeze(1) / 100.0)
                I_2 = self.J_ext * (1 - input_signal / 100.0) if input_signal.dim() == 2 else self.J_ext * (1 - input_signal[:, t].unsqueeze(1) / 100.0)

            else:
                I_1 = 0
                I_2 = 0

            x1 = self.J11 * s_1 - self.J12 * s_2 + self.I_0 + I_1 + I_noise_1
            x2 = self.J22 * s_2 - self.J21 * s_1 + self.I_0 + I_2 + I_noise_2

            H1 = F.relu((self.a * x1 - self.b) / (1 - torch.exp(-self.d * (self.a * x1 - self.b)) + 1e-6))
            H2 = F.relu((self.a * x2 - self.b) / (1 - torch.exp(-self.d * (self.a * x2 - self.b)) + 1e-6))

            ds1dt = - (s_1 / self.tau_s) + (1 - s_1) * H1 * self.gamma / 1000.0
            ds2dt = - (s_2 / self.tau_s) + (1 - s_2) * H2 * self.gamma / 1000.0

            I_noise_1 = I_noise_1.clone() * torch.exp(-self.dt / self.tau_ampa) + self.noise_ampa * torch.sqrt(
                (1 - torch.exp(-2 * self.dt / self.tau_ampa)) / 2.0) * torch.randn(batch_size, 1,
                                                                                   requires_grad=False, device=device)
            I_noise_2 = I_noise_2.clone() * torch.exp(-self.dt / self.tau_ampa) + self.noise_ampa * torch.sqrt(
                (1 - torch.exp(-2 * self.dt / self.tau_ampa)) / 2.0) * torch.randn(batch_size, 1,
                                                                                   requires_grad=False, device=device)
            s_1 = s_1.clone() + ds1dt * self.dt
            s_2 = s_2.clone() + ds2dt * self.dt

            trajectory[t, :, 0] = s_1.clone().squeeze()
            trajectory[t, :, 1] = s_2.clone().squeeze()
            dsdt_trajectory[t, :, 0] = ds1dt.clone().squeeze()
            dsdt_trajectory[t, :, 1] = ds2dt.clone().squeeze()

        decision_times = DiffDecision.apply(trajectory - self.threshold, dsdt_trajectory, self.dt)
        decision_times[decision_times < 0] = decision_times[decision_times < 0] / 1000.0 - self.motor_delay
        decision_times[decision_times > 0] = decision_times[decision_times > 0] / 1000.0 + self.motor_delay

        return decision_times

    # the following inference function is used for visualization purposes, and it returns the trajectory of the model
    # it is not used for training
    def inference(self, input_signal):
        batch_size = input_signal.shape[0]
        device = input_signal.device

        if input_signal.dim() == 3:
            assert input_signal.shape[1] == self.t_stimulus

        s_1 = torch.ones(batch_size, 1, requires_grad=False, device=device) / 10.0
        s_2 = torch.ones(batch_size, 1, requires_grad=False, device=device) / 10.0
        I_noise_1 = torch.randn(batch_size, 1, requires_grad=False, device=device) * self.noise_ampa
        I_noise_2 = torch.randn(batch_size, 1, requires_grad=False, device=device) * self.noise_ampa

        trajectory = torch.zeros((self.time_steps, batch_size, 2), device=device)
        dsdt_trajectory = torch.zeros((self.time_steps, batch_size, 2), device=device)

        for t in range(self.time_steps):
            if t < self.t_stimulus:
                I_1 = self.J_ext * (1 + input_signal / 100.0) if input_signal.dim() == 2 else self.J_ext * (1 + input_signal[:, t].unsqueeze(1) / 100.0)
                I_2 = self.J_ext * (1 - input_signal / 100.0) if input_signal.dim() == 2 else self.J_ext * (1 - input_signal[:, t].unsqueeze(1) / 100.0)

            else:
                I_1 = 0
                I_2 = 0

            x1 = self.J11 * s_1 - self.J12 * s_2 + self.I_0 + I_1 + I_noise_1
            x2 = self.J22 * s_2 - self.J21 * s_1 + self.I_0 + I_2 + I_noise_2

            H1 = F.relu((self.a * x1 - self.b) / (1 - torch.exp(-self.d * (self.a * x1 - self.b)) + 1e-6))
            H2 = F.relu((self.a * x2 - self.b) / (1 - torch.exp(-self.d * (self.a * x2 - self.b)) + 1e-6))

            ds1dt = - (s_1 / self.tau_s) + (1 - s_1) * H1 * self.gamma / 1000.0
            ds2dt = - (s_2 / self.tau_s) + (1 - s_2) * H2 * self.gamma / 1000.0

            # The following clones should not be necessary
            I_noise_1 = I_noise_1.clone() * torch.exp(-self.dt / self.tau_ampa) + self.noise_ampa * torch.sqrt(
                (1 - torch.exp(-2 * self.dt / self.tau_ampa)) / 2.0) * torch.randn(batch_size, 1,
                                                                                   requires_grad=False, device=device)
            I_noise_2 = I_noise_2.clone() * torch.exp(-self.dt / self.tau_ampa) + self.noise_ampa * torch.sqrt(
                (1 - torch.exp(-2 * self.dt / self.tau_ampa)) / 2.0) * torch.randn(batch_size, 1,
                                                                                   requires_grad=False, device=device)
            s_1 = s_1.clone() + ds1dt * self.dt
            s_2 = s_2.clone() + ds2dt * self.dt

            trajectory[t, :, 0] = s_1.clone().squeeze()
            trajectory[t, :, 1] = s_2.clone().squeeze()
            dsdt_trajectory[t, :, 0] = ds1dt.clone().squeeze()
            dsdt_trajectory[t, :, 1] = ds2dt.clone().squeeze()

        # s rather than H is used just for convenience
        decision_times = DiffDecision.apply(trajectory - self.threshold, dsdt_trajectory, self.dt)
        decision_times[decision_times < 0] = decision_times[decision_times < 0] / 1000.0 - self.motor_delay
        decision_times[decision_times > 0] = decision_times[decision_times > 0] / 1000.0 + self.motor_delay

        return decision_times, trajectory, self.threshold

# Similarly, the following DiffDecisionMultiClass function is not much different from the one in RNN_decision.py
# However, for the convenience of those who will use RTified WW directly afterwards, this function is put here anyway
class DiffDecisionMultiClass(Function):
    @staticmethod
    def forward(ctx, trajectory, dsdt_trajectory, dt, max_time):

        mask = trajectory > 0

        decision_times = mask.float().argmax(dim=1).float()
        decision_times[mask.sum(dim=1) == 0] = max_time - 1

        ctx.save_for_backward(dsdt_trajectory, decision_times)
        return decision_times * dt

    @staticmethod
    def backward(ctx, grad_output):
        dsdt_trajectory, decision_times = ctx.saved_tensors
        grads = torch.zeros_like(dsdt_trajectory)

        decision_indices = decision_times.long()

        batch_indices, class_indices = torch.meshgrid(
            torch.arange(decision_times.size(0), device=decision_times.device),
            torch.arange(decision_times.size(1), device=decision_times.device), indexing='ij')

        grads[batch_indices, decision_indices[batch_indices, class_indices], class_indices] = -1.0 / (dsdt_trajectory[
                                batch_indices, decision_indices[batch_indices, class_indices], class_indices] + 1e-6)

        grads = grads * grad_output.unsqueeze(1).expand_as(grads)
        return grads, None, None, None

# The following WongWangMultiClassDecision is a multi-class extension of the WongWangDecisionBP class
# Here N neural populations rather than 2 are used, and the decision is made when the activity of one of the populations reaches the threshold
# Similarly, the model is implemented with RTify, so that all parameters in the model are trainable with backpropagation


class WongWangMultiClassDecision(nn.Module):
    def __init__(self, dt, n_classes):
        super(WongWangMultiClassDecision, self).__init__()
        self.n_classes = n_classes

        self.a = nn.Parameter(torch.tensor(270.0), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(108.0), requires_grad=False)
        self.d = nn.Parameter(torch.tensor(0.1540), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(0.641), requires_grad=False)
        self.tau_s = nn.Parameter(torch.tensor(100.0), requires_grad=False)

        self.J_matrix = nn.Parameter(torch.ones(n_classes, n_classes) * -0.0497, requires_grad=True)
        self.J_matrix.data[range(n_classes), range(n_classes)]= 0.2609
        self.J_ext = nn.Parameter(torch.tensor(0.0156), requires_grad=True)
        self.I_0 = nn.Parameter(torch.tensor(0.3255), requires_grad=True)
        self.noise_ampa = nn.Parameter(torch.tensor(0.02), requires_grad=True)
        self.tau_ampa = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.dt = dt
        self.time_steps = int(500 / self.dt)
        self.t_stimulus = int(500 / self.dt)

    def forward(self, input):
        batch_size = input.shape[0]
        device = input.device

        s = torch.ones(batch_size, self.n_classes, requires_grad=False, device=device) / 10.0
        I_noise = torch.randn(batch_size, self.n_classes, requires_grad=False, device=device) * self.noise_ampa

        trajectory = torch.zeros((batch_size, self.time_steps, self.n_classes), device=device)
        dsdt_trajectory = torch.zeros((batch_size, self.time_steps,self.n_classes), device=device)

        for t in range(self.time_steps):
            if t < self.t_stimulus:
                I = self.J_ext * input[:, t, :] if input.dim() == 3 else self.J_ext * input
            else:
                I = torch.zeros(batch_size, self.n_classes, requires_grad=False, device=device)

            x = torch.matmul(s, self.J_matrix) + self.I_0 + I + I_noise

            H = F.relu((self.a * x - self.b) / (1 - torch.exp(-self.d * (self.a * x - self.b)) + 1e-6))

            dsdt = - (s / self.tau_s) + (1 - s) * H * self.gamma / 1000.0

            I_noise = I_noise.clone() * torch.exp(-self.dt / self.tau_ampa) + self.noise_ampa * torch.sqrt(
                (1 - torch.exp(-2 * self.dt / self.tau_ampa)) / 2.0) * torch.randn(batch_size, self.n_classes,
                                                                                   requires_grad=False, device=device)
            s = s.clone() + dsdt * self.dt

            trajectory[:, t, :] = s.clone()
            dsdt_trajectory[:, t, :] = dsdt.clone()

        decision_times_class = DiffDecisionMultiClass.apply(trajectory - self.threshold, dsdt_trajectory, self.dt, self.time_steps)
        return decision_times_class / 1000.0

    # the following inference function is used for visualization purposes, and it returns the trajectory of the model
    # it is not used for training
    def inference(self, input):
        batch_size = input.shape[0]
        device = input.device

        s = torch.ones(batch_size, self.n_classes, requires_grad=False, device=device) / 10.0
        I_noise = torch.randn(batch_size, self.n_classes, requires_grad=False, device=device) * self.noise_ampa

        trajectory = torch.zeros((batch_size, self.time_steps, self.n_classes), device=device)
        dsdt_trajectory = torch.zeros((batch_size, self.time_steps, self.n_classes), device=device)

        for t in range(self.time_steps):
            if t < self.t_stimulus:
                I = self.J_ext * input[:, t, :] if input.dim() == 3 else self.J_ext * input
            else:
                I = torch.zeros(batch_size, self.n_classes, requires_grad=False, device=device)

            x = torch.matmul(s, self.J_matrix) + self.I_0 + I + I_noise

            H = F.relu((self.a * x - self.b) / (1 - torch.exp(-self.d * (self.a * x - self.b)) + 1e-6))

            dsdt = - (s / self.tau_s) + (1 - s) * H * self.gamma / 1000.0

            I_noise = I_noise.clone() * torch.exp(-self.dt / self.tau_ampa) + self.noise_ampa * torch.sqrt(
                (1 - torch.exp(-2 * self.dt / self.tau_ampa)) / 2.0) * torch.randn(batch_size, self.n_classes,
                                                                                   requires_grad=False, device=device)
            s = s.clone() + dsdt * self.dt

            trajectory[:, t, :] = s.clone()
            dsdt_trajectory[:, t, :] = dsdt.clone()

        decision_times_class = DiffDecisionMultiClass.apply(trajectory - self.threshold, dsdt_trajectory, self.dt,
                                                            self.time_steps)
        return decision_times_class / 1000.0, trajectory, self.threshold