from __future__ import print_function

import tensorflow as tf

class SSDLoss:

    def __init__(self, neg_pos_ratio=3, n_neg_min=0, alpha=1.0):
        self.neg_pos_ratio = tf.constant(neg_pos_ratio)
        self.n_neg_min = tf.constant(n_neg_min)
        self.alpha = tf.constant(alpha)

    def smooth_L1_loss(self, y_true, y_predict):
        absolute_loss = tf.abs(y_true - y_predict)
        square_loss = 0.5 * (y_true - y_predict) ** 2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def cross_entropy_loss(self, y_true, y_predict):
        y_predict = tf.maximum(y_predict, 1e-15)
        log_loss = - tf.reduce_sum(y_true * tf.log(y_predict), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_predict):
        batch_size = tf.shape(y_predict)[0]
        n_boxes = tf.shape(y_predict)[1]

        classification_loss = tf.to_float(self.cross_entropy_loss(y_true[:,:,:-12], y_predict[:,:,:-12]))
        localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:,:,-12:-8], y_predict[:,:,-12:-8]))

        negatives = y_true[:,:,0]
        positives = tf.to_float(tf.reduce_max(y_true[:,:,1:-12], axis=-1))

        n_positives = tf.reduce_sum(positives)
        pos_class_loss = tf.reduce_sum(classification_loss * positives , axis=-1)

        neg_class_loss_all = classification_loss * negatives
        n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32)
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positives), self.n_neg_min), n_neg_losses)

        def f1():
            return tf.zeros([batch_size])
        def f2():
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])
            values, indices = tf.nn.top_k(neg_class_loss_all_1D, n_negative_keep, False)

            negatives_keep = tf.scatter_nd(tf.expand_dims(indices, axis=1), updates=tf.ones_like(indices, dtype=tf.int32), shape=tf.shape(neg_class_loss_all_1D))
            negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes]))

            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss

        # compute the localization loss for the positive targets
        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)

        # compute the total loss
        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positives)

        return total_loss
