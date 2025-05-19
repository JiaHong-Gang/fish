from tensorflow.keras import backend as K
import tensorflow as tf

def iou_metric(y_pred, y_true, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    ious = []
    for i in range(y_true.shape[-1]):
        y_t = y_true[..., i]
        y_p = y_pred[..., i]
        smooth = 1e-7
        intersection = tf.reduce_sum(y_t * y_p, axis=[1, 2])  # batch-wise
        union = tf.reduce_sum(y_t + y_p, axis=[1, 2]) - intersection
        iou = (intersection + smooth) / (union +smooth)
        ious.append(tf.reduce_mean(iou))  # mean over batch

    return tf.reduce_mean(tf.stack(ious))  # mean over channels


