# This file contains the implementation of our RTify framework
# which consists of the forward function: when the model will output a decision
# and the backward function: how to backpropagate through the decision time (which is usually non-differentiable)
# The difference between the two functions is whether the dimension of the evidence function
# If the evidence function is one-dimensional, use DiffDecision.
# If the evidence function is high-dimensional with independent dimensions, use DiffDecisionMultiClass.
# If the evidence function is high-dimensional with interdependent dimensions use DiffDecision but with a nonlinear function to transform the evidence function to a one-dimensional function.


import torch
from torch.autograd import Function


class DiffDecision(Function):
    @staticmethod
    def forward(ctx, trajectory, dsdt_trajectory):

        mask = trajectory > 0
        decision_time = mask.float().argmax(dim=1).float()
        decision_time[mask.sum(dim=1) == 0] = torch.tensor(trajectory.shape[1] - 1, dtype=torch.float32)
        ctx.save_for_backward(dsdt_trajectory, decision_time)

        return decision_time

    @staticmethod
    def backward(ctx, grad_output):
        dsdt_trajectory, decision_times = ctx.saved_tensors
        grads = torch.zeros_like(dsdt_trajectory)
        batch_indices = torch.arange(decision_times.size(0)).to(decision_times.device)

        grads[batch_indices, decision_times.long()] = -1.0 / (dsdt_trajectory[
            batch_indices, decision_times.long()] + 1e-6)

        grads = grads * grad_output.unsqueeze(1).expand_as(grads)
        return grads, None


class DiffDecisionMultiClass(Function):
    @staticmethod
    def forward(ctx, trajectory, dsdt_trajectory):

        mask = trajectory > 0

        decision_times = mask.float().argmax(dim=1).float()
        decision_times[mask.sum(dim=1) == 0] = torch.tensor(trajectory.size(1) - 1, dtype=torch.float32, device=trajectory.device)

        ctx.save_for_backward(dsdt_trajectory, decision_times)
        return decision_times

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
        return grads, None
