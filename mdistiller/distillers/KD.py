import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from .utils import weighted_cross_entropy_loss, prune_batch_logits, prune_tensor_rows


def kd_loss(logits_student, logits_teacher, temperature, pruning_rate):
    
    # prune logit, low class value of teacher
    logits_teacher, pruned_indices = prune_batch_logits(logits_teacher, pruning_rate)
    logits_student = prune_tensor_rows(logits_student, pruned_indices)
    
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
            
        if kwargs["epoch"]<10:
            pruning_rate = 0.3
        if kwargs["epoch"] >=10 and kwargs["epoch"] <20:
            pruning_rate = 0.2
        if kwargs["epoch"]>=20:
            pruning_rate = 0.1
        
        # losses
        loss_ce, temperature_vector =  weighted_cross_entropy_loss(logits_student, target, logits_teacher, self.temperature)
        loss_ce = self.ce_loss_weight * loss_ce
        
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature, pruning_rate
        )
        loss_kd = min(kwargs["epoch"] / 20, 1.0) * loss_kd
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
