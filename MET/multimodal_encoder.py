import tensorflow as tf
from vit_keras import vit
from transformer import EncoderLayer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from attention import AttentionWeightedAverage
from lorendz import LorendzLinearLayer


class SimpleMultimodalEncoder(tf.keras.models.Model):
    def __init__(self, sentence_encoder):
        super(SimpleMultimodalEncoder, self).__init__()
        self.sentence_encoder = sentence_encoder
        self.image_encoder = vit.vit_b16(
            image_size=(384, 384),
            pretrained=True,
            include_top=False,
            pretrained_top=False,
        )
        for l in self.image_encoder.layers:
            l.trainable = False

    @tf.function
    def call(
        self,
        s_ind,
        s_seg,
        l_ind,
        l_seg,
        mask_idx,
        img,
        training=False,
    ):
        h_sentence, h_entity, h_label = self.sentence_encoder(
            s_ind, s_seg, l_ind, l_seg, mask_idx
        )
        h_image = self.image_encoder(img)
        h_image = tf.reduce_mean(h_image, axis=1)
        h_entity = tf.concat([h_entity, h_image], axis=-1)
        aux_loss = 0.0
        return h_sentence, h_image, h_entity, h_label, aux_loss

class VariationalInformationBottleneck(tf.keras.models.Model):
    def __init__(self, input_size: int = 768, hidden_size: int = 768, latent_size: int = 768):
        super(VariationalInformationBottleneck, self).__init__()
        self.LL4mu = LorendzLinearLayer(input_size, latent_size)
        self.dense4mu = Dense(hidden_size)
        self.LL4logvar = LorendzLinearLayer(input_size, latent_size)
        self.dense4logvar = Dense(hidden_size)

    @tf.function
    def kl_loss(self, mu, logvar):
        _kl_loss = -0.5 * tf.reduce_sum(
            1.0 + logvar - tf.math.square(mu) - tf.math.exp(logvar), axis=-1
        )
        return tf.reduce_mean(_kl_loss)

    @tf.function
    def call(self, x):
        mu = self.dense4mu(self.LL4mu(x))
        # mu = self.dense4mu(x)
        logvar = self.dense4logvar(self.LL4logvar(x))
        # logvar = self.dense4logvar(x)
        kl_loss = self.kl_loss(mu, logvar)
        var = tf.math.exp(logvar)
        epsilon = tf.random.normal(tf.shape(mu))
        N = mu + tf.math.sqrt(var) * epsilon
        return N, kl_loss

class VariationalAutoencoder(tf.keras.models.Model):
    def __init__(self, input_size: int = 768 * 3, hidden_size: int = 768, latent_size: int = 768):
        super(VariationalAutoencoder, self).__init__()
        self.lll_hidden_size = latent_size
        lll_latent_size = self.lll_hidden_size + hidden_size
        self.dense4decoder = Sequential([
            Dense(lll_latent_size, activation=tf.nn.leaky_relu),
            Dense(lll_latent_size, activation=tf.nn.leaky_relu),
            Dense(input_size)
        ])
        self.dense4mu = LorendzLinearLayer(input_size, self.lll_hidden_size)
        self.dense4logvar = LorendzLinearLayer(input_size, self.lll_hidden_size)

    @tf.function
    def kl_loss(self, mu, logvar):
        _kl_loss = -0.5 * tf.reduce_sum(
            1.0 + logvar - tf.math.square(mu) - tf.math.exp(logvar), axis=-1
        )
        return tf.reduce_mean(_kl_loss)

    @tf.function
    def encode(self, x):
        mu = self.dense4mu(x)
        logvar = self.dense4logvar(x)
        kl_loss = self.kl_loss(mu, logvar)
        var = tf.math.exp(logvar)
        epsilon = tf.random.normal(tf.shape(mu))
        N = mu + tf.math.sqrt(var) * epsilon
        return N, kl_loss

    @tf.function
    def decode(self, p):
        p = tf.stop_gradient(p)
        epsilon_shape = p.get_shape().as_list()[:-1] + [self.lll_hidden_size]
        epsilon = tf.random.normal(epsilon_shape)
        h_latent = tf.concat([epsilon, p], axis=-1)
        r_x = self.dense4decoder(h_latent)
        return r_x

    @tf.function
    def call(self, x, p):
        x = tf.stop_gradient(x)
        p = tf.stop_gradient(p)
        N, kl_loss = self.encode(x)
        h_latent = tf.concat([N, p], axis=-1)
        r_x = self.dense4decoder(h_latent)
        r_loss = tf.reduce_mean(tf.losses.mean_squared_error(x, r_x))
        overall_loss = r_loss + kl_loss
        return r_x, overall_loss

# TODO: w/o multimodal VIB
# class MultimodalEncoder(tf.keras.models.Model):
#     def __init__(self, sentence_encoder, seen_class):
#         super(MultimodalEncoder, self).__init__()
#         self.sentence_encoder = sentence_encoder
#         self.image_encoder = vit.vit_b16(
#             image_size=(384, 384),
#             pretrained=True,
#             include_top=False,
#             pretrained_top=False,
#         )
#         # self.vib = VariationalInformationBottleneck()
#         self.vae = VariationalAutoencoder()
#         for l in self.image_encoder.layers:
#             l.trainable = False

#         self.ln = Dense(1)

#     def attention(self, q, v):
#         max_len = tf.shape(v)[1]
#         h_q = tf.tile(tf.expand_dims(q, axis=1), [1, max_len, 1])
#         logits = tf.nn.softmax(self.ln(tf.concat([h_q, v], axis=-1)), axis=1)
#         h_v = tf.reduce_sum(logits * v, axis=1)
#         return h_v

#     @tf.function
#     def infonec_loss(self, h_mm_text, h_mm_image):
#         bs = tf.shape(h_mm_text)[0]
#         labels = tf.eye(bs)
#         logits = h_mm_text @ tf.transpose(h_mm_image)
#         prob1 = tf.nn.softmax(logits, axis=0)
#         prob2 = tf.nn.softmax(logits, axis=1)
#         loss1 = tf.losses.categorical_crossentropy(labels, tf.transpose(prob1))
#         loss2 = tf.losses.categorical_crossentropy(labels, prob2)
#         return tf.reduce_mean((loss1 + loss2) / 2.0)

#     @tf.function
#     def call(
#         self,
#         s_ind,
#         s_seg,
#         s_l_ind,
#         s_l_seg,
#         u_l_ind,
#         u_l_seg,
#         mask_idx,
#         img,
#         training=False,
#     ):
#         h_sentence, h_entity, h_seen_label, h_unseen_label = self.sentence_encoder(
#             s_ind, s_seg, s_l_ind, s_l_seg, u_l_ind, u_l_seg, mask_idx
#         )
#         h_image = self.image_encoder(img)
#         h_unseen_label = tf.stop_gradient(h_unseen_label)

#         # h_mm_text, vib_loss4text = self.vib(h_sentence)
#         # h_mm_image, vib_loss4image = self.vib(h_image)
#         # vib_loss = (vib_loss4text + vib_loss4image) / 2.0

#         h_mm_text = h_sentence
#         h_mm_image = h_image
#         global_h_mm_text = tf.reduce_mean(h_mm_text, axis=1)
#         global_h_mm_image = tf.reduce_mean(h_mm_image, axis=1)
#         infonec_loss = self.infonec_loss(global_h_mm_text, global_h_mm_image)

#         global_h_mm = tf.concat([h_mm_text, h_mm_image], axis=1)
#         h_cls = h_sentence[:, 0, :]
#         h_mm_e = self.attention(tf.concat([h_cls, h_entity], axis=-1), global_h_mm)
#         h_mm = tf.concat([h_cls, h_entity, h_mm_e], axis=-1)

#         _, vae_loss = self.vae(h_mm, h_entity)
#         h_unseen_entity = self.vae.decode(h_unseen_label)
#         h_seen_entity = self.vae.decode(h_seen_label)

#         overall_loss = vae_loss + infonec_loss

#         return h_sentence, h_image, h_entity, h_mm, h_seen_label, h_unseen_label, h_seen_entity, h_unseen_entity, overall_loss


# TODO: overall model
class MultimodalEncoder(tf.keras.models.Model):
    def __init__(self, sentence_encoder, vib_latent_size: int = 768, vae_latent_size: int = 768):
        super(MultimodalEncoder, self).__init__()
        self.sentence_encoder = sentence_encoder
        self.image_encoder = vit.vit_b16(
            image_size=(384, 384),
            pretrained=True,
            include_top=False,
            pretrained_top=False,
        )
        self.vib = VariationalInformationBottleneck(latent_size=vib_latent_size)
        self.vae = VariationalAutoencoder(latent_size=vae_latent_size)
        for l in self.image_encoder.layers:
            l.trainable = False
        self.ln = Dense(1)

    def attention(self, q, v):
        max_len = tf.shape(v)[1]
        h_q = tf.tile(tf.expand_dims(q, axis=1), [1, max_len, 1])
        logits = tf.nn.softmax(self.ln(tf.concat([h_q, v], axis=-1)), axis=1)
        h_v = tf.reduce_sum(logits * v, axis=1)
        return h_v

    @tf.function
    def infonec_loss(self, h_mm_text, h_mm_image):
        bs = tf.shape(h_mm_text)[0]
        labels = tf.eye(bs)
        logits = h_mm_text @ tf.transpose(h_mm_image)
        prob1 = tf.nn.softmax(logits, axis=0)
        prob2 = tf.nn.softmax(logits, axis=1)
        loss1 = tf.losses.categorical_crossentropy(labels, tf.transpose(prob1))
        loss2 = tf.losses.categorical_crossentropy(labels, prob2)
        return tf.reduce_mean((loss1 + loss2) / 2.0)

    @tf.function
    def call(
        self,
        s_ind,
        s_seg,
        s_l_ind,
        s_l_seg,
        u_l_ind,
        u_l_seg,
        mask_idx,
        img,
        training=False,
    ):
        h_sentence, h_entity, h_seen_label, h_unseen_label = self.sentence_encoder(
            s_ind, s_seg, s_l_ind, s_l_seg, u_l_ind, u_l_seg, mask_idx
        )
        h_image = self.image_encoder(img)
        h_unseen_label = tf.stop_gradient(h_unseen_label)

        h_mm_text, vib_loss4text = self.vib(h_sentence)
        h_mm_image, vib_loss4image = self.vib(h_image)
        vib_loss = (vib_loss4text + vib_loss4image) / 2.0

        global_h_mm_text = tf.reduce_mean(h_mm_text, axis=1)
        global_h_mm_image = tf.reduce_mean(h_mm_image, axis=1)
        infonec_loss = self.infonec_loss(global_h_mm_text, global_h_mm_image)

        global_h_mm = tf.concat([h_mm_text, h_mm_image], axis=1)
        h_cls = h_sentence[:, 0, :]
        h_mm_e = self.attention(tf.concat([h_cls, h_entity], axis=-1), global_h_mm)
        h_mm = tf.concat([h_cls, h_entity, h_mm_e], axis=-1)

        _, vae_loss = self.vae(h_mm, h_entity)
        h_unseen_entity = self.vae.decode(h_unseen_label)
        h_seen_entity = self.vae.decode(h_seen_label)

        # overall_loss = vib_loss + infonec_loss
        # overall_loss = vae_loss + vib_loss
        overall_loss = vae_loss + vib_loss + infonec_loss

        return h_sentence, h_image, h_entity, h_mm, h_seen_label, h_unseen_label, h_seen_entity, h_unseen_entity, overall_loss