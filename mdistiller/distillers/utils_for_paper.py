import torch
import torch.nn.functional as F


def weighted_cross_entropy_loss(logits_teacher, logits_student, target):    
    # Compute the probabilities
    probs = torch.exp(logits_student - torch.max(logits_student, dim=1, keepdim=True)[0])
    
    # Data adaptive temperature vector with size of "batch size"
    temperature_rate_vector = data_adaptive_temperature(logits_teacher, logits_student) 
    
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    batch_size = logits_student.size(0)

    selected_probs = probs[torch.arange(batch_size), target] # shape(batch_size)
    
    # Adding a small value to avoid log(0)
    eps = 1e-10
    loss = -torch.log(selected_probs + eps).mean()

    return loss, temperature_rate_vector

def data_adaptive_temperature(teacher, student):
    # returns l2 loss from between each data of teacher logits and student logits
    teacher_student_diff = per_data_l2_loss(teacher, student)    
         
    # applies higher temperature for difficult data and lower temperature for easy data
    temperature_rate_vector = fill_values(teacher_student_diff, 0.1, 0.9, 0.95, 1.05)

    return temperature_rate_vector


def prune_batch_logits(tensor, prune_percent):
    k = max(1, int(tensor.shape[1] * prune_percent))  
    threshold_values, _ = torch.kthvalue(tensor, k, dim=1)
    for i in range(tensor.shape[0]):
        tensor[i][tensor[i] <= threshold_values[i]] = -1000

    pruned_indices = []
    for row in tensor:
        row_indices = torch.nonzero(row <= -900).squeeze(dim=1)
        pruned_indices.append(row_indices.tolist())
    return tensor, pruned_indices


def prune_tensor_rows(tensor, indices):
    # pruned_tensor = tensor.clone()  
    for i, idx in enumerate(indices):
        tensor[i][idx] = -1000  
    return tensor

def per_data_l2_loss(tensor1, tensor2):
    
    tensor1 =  F.softmax(tensor1 ,dim=1)
    tensor2 =  F.softmax(tensor2 ,dim=1)
    diff = tensor1 - tensor2
    
    squared_diff = diff.pow(2)
    
    sum_squared_diff = torch.sum(squared_diff, dim=1)
    
    epsilon = 1e-10
    sum_squared_diff = sum_squared_diff + epsilon
    
    data_adaptive_l2_loss = torch.sqrt(sum_squared_diff)
    
    return data_adaptive_l2_loss


def fill_values(lst, down_range, up_range, down_scalar, up_scalar):
    sorted_lst = sorted(lst)
    lower_threshold = sorted_lst[int(len(sorted_lst) * down_range)]
    upper_threshold = sorted_lst[int(len(sorted_lst) * up_range)]
    
    # 결과 리스트 초기화
    result = []
    
    # 각 요소에 대해 조건에 맞게 값을 채워넣습니다.
    for item in lst:
        if item <= lower_threshold:  # 하위 down_range%에 해당하는 값인 경우
            result.append(down_scalar)
        elif item >= upper_threshold:  # 상위 up_range%에 해당하는 값인 경우
            result.append(up_scalar)
        else:  # 나머지 경우
            result.append(1)
    return torch.tensor(result)


def kl_divergence_loss(p, q):
    epsilon = 1e-10
    kl_div = p * (torch.log(p + epsilon) - q)
    kl_div = kl_div.sum(dim=1).sum()
    
    return kl_div


def custom_kl_div(log_pred_student, pred_teacher):
    # Compute the KL divergence for each element
    kl_div = pred_teacher * (torch.log(pred_teacher + 1e-10) - log_pred_student)
    
    # Sum over the last dimension and then take the mean
    loss = kl_div.sum(dim=1).mean()
    return loss