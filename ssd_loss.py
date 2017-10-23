from __future__ import print_function

import tensorflow as tf

class SSDLoss:
    """
    The SSD training objective is derived from the MultiBox objective but is extended to handle multiple object categories.
    The overall objective loss function is a weighted sum of the localization loss and the confidence loss.
    The confidence loss is the softmax loss over multiple classes confidences
    The localization loss is a smooth L1 loss
    """
    def __init__(self, neg_pos_ratio=3, n_neg_min=0, alpha=1.0):
        """
        Arguments:
        neg_pos_ratio (int, optional): The maximum ratio of negative to positive ground truth boxes to include in the loss computation.
            `y_true` contains anchor boxes labeled with the background class. 
            Since the number of background boxes in `y_true` will usually exceed positive labels,
            it is necessary to balance their influence on the loss
        n_neg_min (int, optional): 
        alpha (float, optional): A hyperparameter to weight the localization loss in the computation.
        """
        self.neg_pos_ratio = tf.constant(neg_pos_ratio)
        self.n_neg_min = tf.constant(n_neg_min)
        self.alpha = tf.constant(alpha)

    def smooth_L1_loss(self, y_true, y_predict):
        absolute_loss = tf.abs(y_true - y_predict)
        square_loss = 0.5 * (y_true - y_predict) ** 2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def cross_entropy_loss(self, y_true, y_predict):
        """
        Compute the cross_entropy loss over classification softmax score
        Arguments:
        y_true (keras tensor): A keras tensor has shape (batch_size, num_of_boxes, num_of_classes),
            the last axis contain the one-hot encoding for the classes.
        y_predict (keras tensor): A keras tensor has the identical structure to y_true.
        """
        y_predict = tf.maximum(y_predict, 1e-15)
        log_loss = - tf.reduce_sum(y_true * tf.log(y_predict), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_predict):
        """
        Compute the loss of SSD model prediction against the ground truth
        Arguments:
        y_true (array): A numpy array of shape '(batch_size, num_of_boxes, num_of_classes+12)',
            Be careful to make sure that the index of each given box in `y_true` is the same as the index for 
            corresponding box in `y_predict`.The last axis must have length of `num_of_classes + 12`,
            The 12 last elements are 4 groud truth box coordinates (4*2) and 4 arbitrary entries.
            The last four entries of the last axis are not used by this function and therefore their content are irrelevant in this case.
            Import: Boxes that you want the cost function to ignore need to have a one-hot class vector of all zeros.
        y_predict (keras tensor): The model prediction. The shape is identical to that of 'y_true'

        Returns:
        A scalar, the total multitask loss for classification and localization.
        """
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
