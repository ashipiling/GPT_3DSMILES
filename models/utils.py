
def accuracy2(outputs, targets, ignore=-100):
    mask = targets.ne(ignore)
    pred_id = outputs[mask].argmax(-1)
    targets = targets[mask]
    masked_hit = (pred_id == targets).long().sum()
    masked_cnt = mask.long().sum()
    hit_rate = masked_hit/masked_cnt
    return hit_rate
