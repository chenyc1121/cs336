from collections.abc import Callable, Iterable
from typing import Any

import torch


class adamW(torch.optim.Optimizer):
    def __init__(self,params: Iterable[torch.nn.Parameter],lr: float = 1e-3,betas: tuple[float, float] = (0.9, 0.999),eps: float = 1e-8,weight_decay: float = 1e-2):

        defaults: dict[str, Any] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad
                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # AdamW decouples weight decay from the gradient/moment update.
                parameter.mul_(1 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr / bias_correction1
                denominator = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)
                parameter.addcdiv_(exp_avg, denominator, value=-step_size)

        return loss

