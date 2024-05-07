import torch
import torch.nn.functional as F

"""
포인트: 
선생이 폐급답변을 내놓으면 그냥 label 학습 비중을 더 높이는 것이 이득  
선생이 틀린답변을 내놓으면 label학습 비중을 살짝 높힘
선생이 옳든 답변을 내놓으면 선생의 confidence에 비례해서 confidence가 높을수록 label학습 비중을 줄이고 confidence가 낮을수록 
그런데 이렇게 하는 것이 맞을까? 20 20 20 21 19 예보다는 0 0 0 49 51 얘가 훨씬 더 잘 학습한 것이 아닐까? 그렇다면 결국 
logits teacher에서의 정답을 구해야지 
"""
def weighted_cross_entropy_loss(logits_student, target, logits_teacher, temperature):    
    probs = torch.exp(logits_student - torch.max(logits_student, dim=1, keepdim=True)[0])
    
    # data adaptive temperature vector with size of "batch size"
    temperature_vector = get_temperature_rate(logits_teacher, target, temperature) 
    
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    batch_size = logits_student.size(0)

    selected_probs = probs[torch.arange(batch_size), target] # shape(batch_size)
    
    loss = -torch.log(selected_probs).mean()

    return loss, temperature_vector


def get_fatal_idx(logits_teacher, target, allowed_number):
    fatal_idx_list = list()
    top_values, top_indices = torch.topk(logits_teacher, allowed_number, dim=1)
    for idx, label_and_top_index in enumerate(zip(target, top_indices)):
        label = label_and_top_index[0]
        top_index = label_and_top_index[1]
        if label not in top_index:
            fatal_idx_list.append(idx)
            
    return fatal_idx_list

# 해당 배치에 대해서 뺑이 돌면서 정답(target)에 대한 confidence score구함 
def compare_logit_target(logits, target):
    logits = F.softmax(logits, dim=1)
    
    # logit이 target에 가지는 confidence score을 저장하는 리스트
    logit_target_list = list()
    for batch_idx, single_target in enumerate(target):
        logit_target_list.append(logits[batch_idx][single_target])
        
    return logit_target_list

def fill_values(lst, down_range, up_range, down_scalar, up_scalar):
    # 리스트를 정렬하여 상위 30%와 하위 30%의 위치를 찾습니다.
    sorted_lst = sorted(lst)
    lower_threshold = sorted_lst[int(len(sorted_lst) * down_range)]
    upper_threshold = sorted_lst[int(len(sorted_lst) * up_range)]
    
    # 결과 리스트 초기화
    result = []
    
    # 각 요소에 대해 조건에 맞게 값을 채워넣습니다.
    for item in lst:
        if item <= lower_threshold:  # 하위 30%에 해당하는 값인 경우
            result.append(down_scalar)
        elif item >= upper_threshold:  # 상위 30%에 해당하는 값인 경우
            result.append(up_scalar)
        else:  # 나머지 경우
            result.append(1)
    return torch.tensor(result)


def prune_batch_logits(tensor, prune_percent):
    k = max(1, int(tensor.shape[1] * prune_percent))  
    threshold_values, _ = torch.kthvalue(tensor, k, dim=1)
    for i in range(tensor.shape[0]):
        tensor[i][tensor[i] <= threshold_values[i]] = -200

    pruned_indices = []
    for row in tensor:
        row_indices = torch.nonzero(row <= -190).squeeze(dim=1)
        pruned_indices.append(row_indices.tolist())
    return tensor, pruned_indices


def prune_tensor_rows(pruned_tensor, indices):
    # pruned_tensor = tensor.clone()  
    for i, idx in enumerate(indices):
        pruned_tensor[i][idx] = -200  
    return pruned_tensor


# 이 부분은 현재 teacher logit과 target을 보고 있다. target이 아니라 stduent logit으로 
def get_temperature_rate(logits_teacher, target, temperature):
    logit_target_list = compare_logit_target(logits_teacher, target)
    temperature_vector = fill_values(logit_target_list, 0.1, 0.9, 1.05, 0.95) * temperature
    return temperature_vector


def weighted_l2_loss(tensor1, tensor2, target_rate_vector = None):
    diff = tensor1 - tensor2
    
    squared_diff = diff.pow(2)
    
    if target_rate_vector is not None:
        target_rate_vector = target_rate_vector.to("cuda:0")
        squared_diff = squared_diff * target_rate_vector
        
    # 각 요소의 제곱을 합하여 총 제곱 합을 계산합니다.
    sum_squared_diff = squared_diff.sum()
    
    # 제곱 합의 제곱근을 계산하여 L2 손실을 얻습니다.
    weighted_l2_loss = torch.sqrt(sum_squared_diff)
    
    return weighted_l2_loss


def weight_row_wise_loss(student_last_fc, teacher_last_fc):
    num_classes = student_last_fc.weight.shape[0] # weight is 100 * 64
    
    row_loss_sum = 0

    for column in range(num_classes):
        student_weight_row = student_last_fc.weight[column, :] # 1 * 64
        teacher_weight_row = teacher_last_fc.weight[column, :] # 1 * 64
        
        row_loss = weighted_l2_loss(teacher_weight_row, student_weight_row)
        
        row_loss_sum += row_loss
    
    return row_loss_sum