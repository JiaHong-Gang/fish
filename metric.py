from tensorflow.keras import backend as K

def iou_metric(y_pre, y_ture):
    smooth =1e-6
    intersection = K.sum(y_pre * y_ture)
    union = K.sum(y_pre) + K.sum(y_ture) - intersection
    iou = (smooth + intersection) / (union +smooth)

    return iou
