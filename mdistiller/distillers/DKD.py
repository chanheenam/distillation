import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from .utils import weighted_cross_entropy_loss, prune_batch_logits, prune_tensor_rows, weight_row_wise_loss, weighted_l2_loss
"""
dkd의 특성상 non target class의 영향력이 늘어나는데
non target class중에서 쓰레기값이 많다는 것이 내 이론
더군다나 temperature 하이퍼 파라미터의 영향으로 쓰레기값의 value가 커질 수도 있다...
이를 위해서 하위 20%의 logit값들을 아예 지워버리고 student에서도 마찬가지로 지워 버린다.     


"""
def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature, temperature_vector, pruning_rate):
    
    # prune logit, low class value of teacher
    # logits_teacher, pruned_indices = prune_batch_logits(logits_teacher, pruning_rate)
    # logits_student = prune_tensor_rows(logits_student, pruned_indices)
    
    # get ground truth mask and opposed mask, student_logit is only for shape
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    # get smoothened softmax value
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
    # # data adaptive temperature의 경우의 수 
    # temperature_vector = temperature_vector.to("cuda:0")
    # pred_student = F.softmax(logits_student / temperature_vector.unsqueeze(1), dim=1)
    # pred_teacher = F.softmax(logits_teacher / temperature_vector.unsqueeze(1), dim=1)
    
    # [real target에 대해서 student의 output, 나머지 확률의 합]
    # e.g. tensor([[0.8000, 0.2000], [0.7000, 0.3000], [0.2000, 0.8000]])
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    
    # it is used for only tckd loss
    log_pred_student = torch.log(pred_student)
    
    # divide by batch size since kl_div is the loss sum of every batch 
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    
    # # data adaptive temperature의 경우의 수
    # pred_teacher_part2 = F.softmax(
    #     logits_teacher / temperature_vector.unsqueeze(1) - 1000.0 * gt_mask, dim=1
    # )
    # log_pred_student_part2 = F.log_softmax(
    #     logits_student / temperature_vector.unsqueeze(1) - 1000.0 * gt_mask, dim=1
    # )
    
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss, gt_mask


# target을 true로 나머지는 false로, mask shape은 logit하고 일치
def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

# target을 false로 나머지는 true로 mask shape은 logit하고 일치 
def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

# element_wise multiply -> summation 
def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def target_rate_loss(student_last_fc, teacher_last_fc, gt_mask):
    student_weight_row_sum = torch.sum(student_last_fc.weight, 1) # 사이즈: [100]
    teacher_weight_row_sum = torch.sum(teacher_last_fc.weight, 1) # 사이즈: [100]
        
    student_weight_row_rate = student_weight_row_sum / student_weight_row_sum.sum() # 사이즈: [100]
    teacher_weight_row_rate = teacher_weight_row_sum / teacher_weight_row_sum.sum() # 사이즈: [100]
        
    batch_target_num = gt_mask.sum(0, keepdims=False)
    batch_target_sum = torch.sum(batch_target_num)
    target_rate_vector = batch_target_num / batch_target_sum # 사이즈: [100]
        
    weighted_loss = weighted_l2_loss(student_weight_row_rate, 
                                     teacher_weight_row_rate, 
                                     target_rate_vector)    
    
    return weighted_loss



class DKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(DKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # cross entropy losses, target is vector
        loss_ce, temperature_vector =  weighted_cross_entropy_loss(logits_student, target, logits_teacher, self.temperature)
        loss_ce = self.ce_loss_weight * loss_ce
                
        # if kwargs["epoch"]<10:
        #     self.temperature = 4.3
        # if kwargs["epoch"] >=10 and kwargs["epoch"] <20:
        #     self.temperature = 4.15
        # if kwargs["epoch"]>=20:
        #     self.temperature = 4
        
        if kwargs["epoch"]<10:
            pruning_rate = 0.1
        if kwargs["epoch"] >=10 and kwargs["epoch"] <20:
            pruning_rate = 0.2
        if kwargs["epoch"]>=20:
            pruning_rate = 0.3
        
            
        loss_dkd, gt_mask = dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            temperature_vector,
            pruning_rate
        )
        
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * loss_dkd
        
        # rate_loss = target_rate_loss(kwargs["student_last_fc"], 
        #                              kwargs["teacher_last_fc"], 
        #                              gt_mask) 
        # rate_loss = min(kwargs["epoch"] / self.warmup, 1.0) * rate_loss
        
        # weight_row_loss = weight_row_wise_loss(kwargs["student_last_fc"], kwargs["teacher_last_fc"])
        # weight_row_loss = min(kwargs["epoch"] / self.warmup, 1.0) * weight_row_loss
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd
            # "loss_rate": rate_loss,
            # "weight_row_loss": weight_row_loss
        }

        return logits_student, losses_dict
