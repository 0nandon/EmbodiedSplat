'''IoU'''
import numpy as np
from database.class_labels import SCANNET20_CLASS_LABELS, SCANNET200_CLASS_LABELS, REPLICA_CLASS_LABELS, SCANNET200_IGNORE
from database.class_labels import SCANNET10_CLASS_LABELS, SCANNET15_CLASS_LABELS, SCANNETPP_CLASS_LABELS, SCANNETPP_IGNORE

UNKNOWN_ID = -1

def confusion_matrix(pred_ids, gt_ids, num_classes):
    '''calculate the confusion matrix.'''

    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids != UNKNOWN_ID

    return np.bincount(
        pred_ids[idxs] * num_classes + gt_ids[idxs],
        minlength=num_classes**2).reshape((
        num_classes, num_classes)).astype(np.ulonglong)


def get_iou(label_id, confusion):
    '''calculate IoU.'''

    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom, tp, denom


def evaluate(pred_ids, gt_ids, stdout=False, dataset='scannet20'):
    if stdout:
        print('evaluating', gt_ids.size, 'points...')
    if 'scannet20' == dataset:
        CLASS_LABELS = SCANNET20_CLASS_LABELS
    elif 'scannet15' == dataset:
        CLASS_LABELS = SCANNET15_CLASS_LABELS
    elif 'scannet10' == dataset:
        CLASS_LABELS = SCANNET10_CLASS_LABELS
    elif 'scannet200' == dataset:
        CLASS_LABELS = [cls for cls in SCANNET200_CLASS_LABELS if cls not in SCANNET200_IGNORE]
    elif 'replica' == dataset:
        CLASS_LABELS = REPLICA_CLASS_LABELS
    elif 'scannetpp' == dataset:
        CLASS_LABELS = [cls for cls in SCANNETPP_CLASS_LABELS if cls not in SCANNETPP_IGNORE] 
        
    N_CLASSES = len(CLASS_LABELS)
    confusion = confusion_matrix(pred_ids, gt_ids, N_CLASSES)
    class_ious = {}
    class_accs = {}
    mean_iou = 0
    mean_acc = 0

    count = 0
    not_found = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        if (gt_ids==i).sum() == 0: # at least 1 point needs to be in the evaluation for this class
            not_found += 1
            continue

        class_ious[label_name] = get_iou(i, confusion)
        class_accs[label_name] = class_ious[label_name][1] / (gt_ids==i).sum()
        count+=1

        mean_iou += class_ious[label_name][0]
        mean_acc += class_accs[label_name]

    mean_iou /= (N_CLASSES - not_found)
    mean_acc /= (N_CLASSES - not_found)
    if stdout:
        print('classes          IoU')
        print('----------------------------')
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            try:
                if 'matterport' in dataset:
                    print('{0:<14s}: {1:>5.3f}'.format(label_name, class_accs[label_name]))

                else:
                    print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2]))
            except:
                print(label_name + ' error!')
                continue
        print('Mean IoU', mean_iou)
        print('Mean Acc', mean_acc)
    return mean_iou
