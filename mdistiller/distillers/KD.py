import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from .utils_for_paper import prune_batch_logits, prune_tensor_rows,\
weighted_cross_entropy_loss, kl_divergence_loss, custom_kl_div


def kd_loss(logits_student, logits_teacher, temperature, temperature_vector, pruning_rate):
    
    # prune logit, low class value of teacher
    logits_teacher, pruned_indices = prune_batch_logits(logits_teacher, pruning_rate)
    logits_student = prune_tensor_rows(logits_student, pruned_indices)
    
    # data adaptive temperature의 경우의 수 
    if temperature_vector is not None:
        temperature_vector = temperature_vector.to("cuda:0")
        log_pred_student = F.log_softmax(logits_student / temperature_vector.unsqueeze(1), dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature_vector.unsqueeze(1), dim=1)
    else:
        log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
    loss_kd = custom_kl_div(log_pred_student, pred_teacher)
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
            
            
        loss_ce, temperature_rate_vector =  weighted_cross_entropy_loss(logits_teacher, logits_student, target)
        # print(temperature_rate_vector)
        # epoch_adaptive = [0.2, -0.2] 
        # max_temp = self.temperature + epoch_adaptive[0] * self.temperature
        # min_temp = self.temperature + epoch_adaptive[1] * self.temperature
        # diff = max_temp - min_temp
        # temperature = max_temp - diff*(kwargs["epoch"])/80
        
        if kwargs["epoch"]<20:
            temperature_vector = None
            pruning_rate = 0.3
            temperature = 4.0
        elif kwargs["epoch"] >=20 and kwargs["epoch"] < 40:
            temperature_vector = None
            pruning_rate = 0.2
            temperature = 4.0
        elif kwargs["epoch"]>=40 and kwargs["epoch"] < 60:
            temperature_vector = None
            pruning_rate = 0.1
            temperature = 4.0
        elif kwargs["epoch"]>=40:
            temperature_vector = None
            pruning_rate = 0.0
            temperature = 4.0
        
        #
        loss_ce = self.ce_loss_weight * loss_ce
        
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, temperature, temperature_vector, pruning_rate
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
