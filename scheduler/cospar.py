import math
from torch.optim.lr_scheduler import _LRScheduler

class CosPar(_LRScheduler):
    """
    CosineAnnealingWarmPartialRestarts (CosPar) (260930) コスパ(日本語読み)
    最後のサイクルのみ通常のコサインカーブ(0まで減衰)、
    それ以外のサイクルは指定した最低値(min_lr_rate)までしか減衰させないリスタートスケジューラ
    デフォルトでは、中間サイクルを1/3(0.3) まで減衰させる(ユーザー指定で0.0～1.0に調整可)
    usage：
    --lr_scheduler_type=scheduler.cospar.CosPar --lr_scheduler_args "min_lr_rate=0.5"
    """
    def __init__(
        self, 
        optimizer, 
        num_warmup_steps: int= 0,       # デフォルト値
        num_training_steps: int= 100,   # デフォルト値
        num_cycles: int = 3,            # サイクル回数
        min_lr_rate: float = 0.3,       # 最終サイクル以外での最低学習率の割合 (0.0 〜 1.0)
        last_epoch: int = -1
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        
        # 0.0〜1.0の範囲に制限クランプ (安全のため)
        self.min_lr_rate = max(0.0, min(1.0, min_lr_rate))
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        
        # ウォームアップ期間
        if current_step < self.num_warmup_steps and self.num_warmup_steps > 0:
            warmup_factor = float(current_step) / float(max(1, self.num_warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # トレーニング終了後
        if current_step >= self.num_training_steps:
            return [0.0 for _ in self.base_lrs]

        # ウォームアップ終了後の進捗度 (0.0 〜 1.0)
        progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        scaled_progress = float(self.num_cycles) * progress

        # サイクル判定と倍率の計算
        if math.ceil(scaled_progress) == self.num_cycles or scaled_progress >= self.num_cycles:
            # 最後のサイクル：通常のコサイン減衰 (0.0 〜 1.0)
            multiplier = max(0.0, 0.5 * (1.0 + math.cos(math.pi * (scaled_progress % 1.0))))
        else:
            # 途中サイクル：指定された min_lr_rate を下限としてコサイン減衰
            # min_lr_rate が 0.5 の場合、変動範囲は 0.5 〜 1.0 になる
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * (scaled_progress % 1.0)))
            multiplier = self.min_lr_rate + (1.0 - self.min_lr_rate) * cosine_val

        return [base_lr * multiplier for base_lr in self.base_lrs]