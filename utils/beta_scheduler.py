from utils.common_library import *


class BetaScheduler:
    def __init__(self,
                 mode="adaptive",
                 beta_init=4.0,
                 beta_max=40.0,
                 beta_patience=100,
                 stop_patience=500,
                 min_delta=1e-3,
                 shrink_rate=0.9,
                 alpha_decay_interval=200):

        self.mode = mode

        self.beta = beta_init
        self.beta_max = beta_max

        self.alpha = 1.0 / beta_init
        self.alpha0 = 1.0 / beta_init
        self.alpha_min = 1.0 / beta_max
        self.shrink_rate = shrink_rate
        self.alpha_decay_interval = alpha_decay_interval

        self.beta_patience = beta_patience
        self.stop_patience = stop_patience
        self.min_delta = min_delta

        self.best_loss = np.inf
        self.beta_wait = 0
        self.stop_wait = 0
        self.early_stop = False

    def get_beta(self, epoch):
        if self.mode == "step":
            return self._step_schedule(epoch)

        elif self.mode == "smooth":
            return self._smooth_schedule(epoch)

        elif self.mode == "adaptive":
            return self.beta

        elif self.mode == "alpha_decay":
            return self._alpha_decay_schedule(epoch)

        else:
            raise ValueError(f"Unknown beta scheduler mode: {self.mode}")

    def update(self, current_loss, epoch=None):
        current_loss = float(current_loss)

        if current_loss < self.best_loss * (1.0 - self.min_delta):
            self.best_loss = current_loss
            self.beta_wait = 0
            self.stop_wait = 0
            return self.beta

        self.beta_wait += 1
        self.stop_wait += 1

        if self.mode == "adaptive" and self.beta_wait >= self.beta_patience:
            old_beta = self.beta
            self.alpha = max(self.alpha * self.shrink_rate, self.alpha_min)
            self.beta = 1.0 / self.alpha
            self.beta_wait = 0

            if epoch is not None:
                print(f"[Beta update] epoch {epoch}: {old_beta:.2f} -> {self.beta:.2f}")

        if self.stop_wait >= self.stop_patience:
            self.early_stop = False

        return self.beta

    def _step_schedule(self, epoch):
        if epoch < 200:
            return 2.0
        elif epoch < 400:
            return 4.0
        elif epoch < 600:
            return 6.0
        elif epoch < 800:
            return 8.0
        else:
            return 10.0

    def _smooth_schedule(self, epoch):
        start_epoch = 1
        end_epoch = 800
        beta_start = 2.0
        beta_end = 10.0
        transition_width = 50

        if epoch < start_epoch + transition_width:
            factor = (epoch - start_epoch) / transition_width
        elif epoch > end_epoch - transition_width:
            factor = (end_epoch - epoch) / transition_width
        else:
            factor = 1.0

        factor = np.clip(factor, 0.0, 1.0)
        return float(beta_start + (beta_end - beta_start) * factor)

    def _alpha_decay_schedule(self, epoch):
        n_update = (epoch - 1) // self.alpha_decay_interval

        alpha = self.alpha0 * (self.shrink_rate ** n_update)
        alpha = max(alpha, self.alpha_min)

        return float(1.0 / alpha)