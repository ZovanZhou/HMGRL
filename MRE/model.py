import tensorflow as tf
from framework import ZeroShotMRelModel
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Dropout, Layer


class Proto(ZeroShotMRelModel):
    def __init__(self, encoder, use_img: bool = False, d: int = 64):
        ZeroShotMRelModel.__init__(self, encoder, use_img)
        self.ln1 = Dense(d)
        self.ln2 = Dense(d)
        self.dropout = Dropout(0.1)

    @tf.function
    def call(self, data, n_class, training=False):
        data = self.unpack_data(data)
        _, h_entity, h_label = self.encoder(*data)

        hidden_size = h_label.get_shape().as_list()[-1]
        h_label_emb = tf.reshape(h_label, (-1, n_class, hidden_size))

        h_emb = h_entity

        if training:
            h_emb = self.dropout(h_emb, training=training)
            h_label_emb = self.dropout(h_label_emb, training=training)
        h_emb = self.ln1(h_emb)
        h_label_emb = self.ln2(h_label_emb)

        logits = self._batch_dist(h_emb, h_label_emb)
        pred = tf.argmax(logits, axis=-1)
        aux_loss = 0.0
        return logits, pred, aux_loss

class MMProto(ZeroShotMRelModel):
    def __init__(
        self, encoder, use_img: bool = True, d: int = 64, hidden_size: int = 768
    ):
        ZeroShotMRelModel.__init__(self, encoder, use_img)
        self.ln1 = Dense(d)
        self.ln2 = Dense(d)
        self.ln3 = Dense(hidden_size, activation="tanh")
        self.dropout = Dropout(0.1)

    @tf.function
    def call(self, data, n_class, training=False):
        data = self.unpack_data(data)
        h_sentence, h_image, h_entity, h_label, vibmoe_loss = self.encoder(*data)

        hidden_size = h_label.get_shape().as_list()[-1]
        h_label_emb = tf.reshape(h_label, (-1, n_class, hidden_size))

        h_emb = self.ln3(h_entity)

        if training:
            h_emb = self.dropout(h_emb, training=training)
            h_label_emb = self.dropout(h_label_emb, training=training)
        h_emb = self.ln1(h_emb)
        h_label_emb = self.ln2(h_label_emb)

        logits = self._batch_dist(h_emb, h_label_emb)
        pred = tf.argmax(logits, axis=-1)

        aux_loss = vibmoe_loss
        return logits, pred, aux_loss


class MMBilinearProto(ZeroShotMRelModel):
    def __init__(
        self, encoder, use_img: bool = True, d: int = 768, eta: float = 10.0, zeta: float = 10.0
    ):
        ZeroShotMRelModel.__init__(self, encoder, use_img)
        self.ln1 = Dense(d)
        self.ln2 = Dense(d)
        self.dropout = Dropout(0.1)
        self.zeta = zeta
        self.eta = eta

    @tf.function
    def unseen_loss(self, n_class, h_entity, h_label):
        one_hot_labels = tf.eye(n_class)
        logits = h_entity @ tf.transpose(h_label)
        loss = tf.losses.categorical_crossentropy(one_hot_labels, logits, from_logits=True)

        true_logits = tf.reduce_max(logits * one_hot_labels, axis=-1)
        unseen_logits_distribution = tf.nn.softmax(true_logits)
        unseen_logits_target = (1.0 / n_class) * tf.ones_like(unseen_logits_distribution)
        kl_loss = tf.losses.kl_divergence(unseen_logits_target, unseen_logits_distribution)
        return self.eta * tf.reduce_mean(loss) + self.zeta * kl_loss

    @tf.function
    def construct_syntactic_data(self, n_class, h_label, h_entity):
        entity_hidden_size = h_entity.get_shape().as_list()[-1]
        h_entity = tf.reshape(h_entity, shape=(-1, n_class, entity_hidden_size))
        h_entity = tf.reduce_mean(h_entity, axis=0)
        h_label_emb = h_label[0, :, :]
        h_entity = self.ln1(h_entity)
        h_label_emb = self.ln2(h_label_emb)
        return h_entity, h_label_emb

    @tf.function
    def call(self, data, n_seen_class, n_unseen_class, training=False):
        data = self.unpack_data(data)
        _, _, h_mm_entity, h_seen_label, h_unseen_label, h_seen_entity, h_unseen_entity, overall_loss = self.encoder(*data, training=training)

        label_hidden_size = h_seen_label.get_shape().as_list()[-1]
        raw_h_seen_label_emb = tf.reshape(h_seen_label, (-1, n_seen_class, label_hidden_size))
        raw_h_unseen_label_emb = tf.reshape(h_unseen_label, (-1, n_unseen_class, label_hidden_size))
        if training:
            h_seen_emb = self.dropout(h_mm_entity, training=training)
            h_seen_label_emb = self.dropout(raw_h_seen_label_emb, training=training)
            h_seen_emb = self.ln1(h_seen_emb)
            h_seen_label_emb = self.ln2(h_seen_label_emb)
            seen_logits = self._batch_dist(h_seen_emb, h_seen_label_emb)
            pred = tf.argmax(seen_logits, axis=-1)

            h_unseen_sync_entity, h_unseen_sync_label = self.construct_syntactic_data(n_unseen_class, raw_h_unseen_label_emb, h_unseen_entity)
            unseen_loss = self.unseen_loss(n_unseen_class, h_unseen_sync_entity, h_unseen_sync_label)

            overall_loss += unseen_loss

            return seen_logits, pred, overall_loss
        else:
            h_label_emb = tf.concat([raw_h_seen_label_emb, raw_h_unseen_label_emb], axis=1)
            h_seen_emb = self.ln1(h_mm_entity)
            h_label_emb = self.ln2(h_label_emb)
            logits = self._batch_dist(h_seen_emb, h_label_emb)
            return logits