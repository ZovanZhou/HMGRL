import tensorflow as tf
from vit_keras import vit
from transformer import EncoderLayer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from attention import AttentionWeightedAverage


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
        head_idx,
        tail_idx,
        img,
        training=False,
    ):
        h_sentence, h_entity, h_label = self.sentence_encoder(
            s_ind, s_seg, l_ind, l_seg, head_idx, tail_idx
        )
        h_image = self.image_encoder(img)
        h_image = tf.reduce_mean(h_image, axis=1)
        h_entity = tf.concat([h_entity, h_image], axis=-1)
        aux_loss = 0.0
        return h_sentence, h_image, h_entity, h_label, aux_loss

class InstancePrototypeDictionaryLearningLayer(tf.keras.models.Model):
    def __init__(self, seen_class, hidden_size: int = 768):
        super(InstancePrototypeDictionaryLearningLayer, self).__init__()
        self.seen_class = seen_class
        self.D = self.add_weight(shape=[hidden_size, seen_class], trainable=True, initializer=initializers.he_normal)

    @tf.function
    def call(self, x):
        x = tf.stop_gradient(x)
        prob = tf.nn.softmax(x @ self.D, axis=-1)
        label = tf.one_hot(tf.argmax(prob, axis=-1), depth=self.seen_class)
        h_o = label @ tf.transpose(self.D)
        entropy = - tf.reduce_sum(prob * tf.math.log(prob), axis=-1)
        dict_loss = tf.losses.mean_squared_error(x, h_o)
        return h_o, dict_loss, entropy

class Lorentzian(tf.keras.models.Model):
    def __init__(self, c):
        super(Lorentzian, self).__init__()
        self.c = c
        self.max_norm = 1000
    
    @tf.function
    def exp_map_x(self, p, dp, is_res_normalize=True, is_dp_normalize=True):
        if is_dp_normalize:
            dp = self.normalize_tangent(p, dp)
        dp_lnorm = self.scalar_product(dp, dp, keepdims=True)
        dp_lnorm = tf.math.sqrt(tf.maximum(dp_lnorm + 1e-6, 1e-6))
        dp_lnorm_cut = tf.minimum(dp_lnorm, 50.0)
        sqrt_c = tf.math.sqrt(self.c)
        res = (tf.math.cosh(dp_lnorm_cut / sqrt_c) * p) + sqrt_c * (tf.math.sinh(dp_lnorm_cut / sqrt_c) * dp / dp_lnorm)
        if is_res_normalize:
            res = self.normalize(res)
        return res

    @tf.function
    def exp_map_zero(self, dp, is_res_normalize=True, is_dp_normalize=True):
        ones = tf.ones_like(dp[..., 0]) * tf.math.sqrt(self.c)
        zeros = tf.zeros_like(dp[..., 1:])
        zeros = tf.concat([tf.expand_dims(ones, axis=-1), zeros], axis=-1)
        return self.exp_map_x(zeros, dp, is_res_normalize, is_dp_normalize)

    @tf.function
    def normalize(self, p):
        features = p[..., 1:]
        if self.max_norm:
            features = tf.clip_by_norm(features, clip_norm=self.max_norm, axes=-1)
        first = self.c + tf.reduce_sum(tf.math.pow(features, 2), axis=-1, keepdims=True)
        first = tf.math.sqrt(first)
        return tf.concat([first, features], axis=-1)

    @tf.function
    def normalize_tangent(self, p, p_tan):
        p_tail = p[..., 1:]
        p_tan_tail = p_tan[..., 1:]
        ptpt = tf.reduce_sum(p_tail * p_tan_tail, axis=-1, keepdims=True)
        p_head = tf.math.sqrt(self.c + tf.reduce_sum(tf.math.pow(p_tail, 2), axis=-1, keepdims=True) + 1e-6)
        return tf.concat((ptpt / p_head, p_tan_tail), axis=-1)
    
    @tf.function
    def normalize_input(self, x):
        shape = x.get_shape().as_list()[:-1]
        zeros = tf.zeros(shape=[*shape, 1])
        norm_x = tf.concat([zeros, x], axis=-1)
        return self.exp_map_zero(norm_x)

    @tf.function
    def scalar_product(self, x, y, keepdims=False):
        xy = x * y
        xy = tf.concat([tf.expand_dims(0.0 - xy[..., 0], axis=-1), xy[..., 1:]], axis=-1)
        return tf.reduce_sum(xy, axis=-1, keepdims=keepdims)

    @tf.function
    def lorentzian_distance(self, x, y):
        xy = self.scalar_product(x, y)
        return -2.0 * (self.c + xy)

    @tf.function
    def induced_distance(self, x, y):
        xy = self.scalar_product(x, y)
        sqrt_c = tf.math.sqrt(self.c)
        return sqrt_c * tf.math.acosh(- xy / self.c + 1e-6)

    @tf.function
    def log_map_x(self, x, y, is_tan_normalize=True):
        xy_distance = self.induced_distance(x, y)
        tmp_vector = y + self.scalar_product(x, y, keepdims=True) / self.c * x
        tmp_norm = tf.math.sqrt(self.scalar_product(tmp_vector, tmp_vector) + 1e-6)
        y_tan = tf.expand_dims(xy_distance, axis=-1) / tf.expand_dims(tmp_norm, axis=-1) * tmp_vector
        if is_tan_normalize:
            y_tan = self.normalize_tangent(x, y_tan)
        return y_tan

    @tf.function
    def log_map_zero(self, y, is_tan_normalize=True):
        ones = tf.ones_like(y[..., 0]) * tf.math.sqrt(self.c)
        zeros = tf.zeros_like(y[..., 1:])
        zeros = tf.concat([tf.expand_dims(ones, axis=-1), zeros], axis=-1)
        return self.log_map_x(zeros, y, is_tan_normalize)
    
    @tf.function
    def normalize_tangent_zero(self, p_tan):
        ones = tf.ones_like(p_tan[..., 0]) * tf.math.sqrt(self.c)
        zeros = tf.zeros_like(p_tan[..., 1:])
        zeros = tf.concat([tf.expand_dims(ones, axis=-1), zeros], axis=-1)
        return self.normalize_tangent(zeros, p_tan)

    @tf.function
    def matvec_regular(self, m, x, b):
        x_tan = self.log_map_zero(x)
        x_head = tf.expand_dims(x_tan[..., 0], axis=-1)
        x_tail = x_tan[..., 1:]
        mx = x_tail @ tf.transpose(m)
        mx_b = mx + b
        mx = tf.concat([x_head, mx_b], axis=-1)
        mx = self.normalize_tangent_zero(mx)
        mx = self.exp_map_zero(mx)
        # mask = tf.cast(tf.equal(mx, 0), tf.float32)
        # cond = tf.reduce_prod(mask, axis=-1, keepdims=True)
        cond = tf.reduce_prod(mx, axis=-1, keepdims=True)
        zeros = tf.zeros_like(mx)
        res = tf.where(tf.equal(cond, 0.0), zeros, mx)
        return res

class LorendzLinearLayer(tf.keras.models.Model):
    def __init__(self, in_size, out_size):
        super(LorendzLinearLayer, self).__init__()
        self.weight = self.add_weight(shape=[out_size - 1, in_size], initializer=initializers.he_uniform, name="{}_w".format(self.name))
        self.bias = self.add_weight(shape=[1, out_size - 1], initializer=initializers.constant, name="{}_b".format(self.name))
        self.lorendz_model = Lorentzian(1.0)
    
    @tf.function
    def call(self, x):
        x_loren = self.lorendz_model.normalize_input(x)
        x_w = self.lorendz_model.matvec_regular(self.weight, x_loren, self.bias)
        x_tan = self.lorendz_model.log_map_zero(x_w)
        return x_tan

class VariationalAutoencoder(tf.keras.models.Model):
    def __init__(self, input_size: int = 768 * 4, hidden_size: int = 768, latent_size: int = 768):
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
#         head_idx,
#         tail_idx,
#         img,
#         training=False,
#     ):
#         h_sentence, h_entity, h_relation, h_seen_label, h_unseen_label = self.sentence_encoder(
#             s_ind, s_seg, s_l_ind, s_l_seg, u_l_ind, u_l_seg, head_idx, tail_idx
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

#         _, vae_loss = self.vae(h_mm, h_relation)
#         h_unseen_entity = self.vae.decode(h_unseen_label)
#         h_seen_entity = self.vae.decode(h_seen_label)

#         overall_loss = vae_loss + infonec_loss

#         return h_sentence, h_image, h_mm, h_seen_label, h_unseen_label, h_seen_entity, h_unseen_entity, overall_loss


# TODO: overall model
class MultimodalEncoder(tf.keras.models.Model):
    def __init__(self, sentence_encoder, vib_latent_size, vae_latent_size):
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
        head_idx,
        tail_idx,
        img,
        training=False,
    ):
        h_sentence, h_entity, h_relation, h_seen_label, h_unseen_label = self.sentence_encoder(
            s_ind, s_seg, s_l_ind, s_l_seg, u_l_ind, u_l_seg, head_idx, tail_idx
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

        _, vae_loss = self.vae(h_mm, h_relation)
        h_unseen_entity = self.vae.decode(h_unseen_label)
        h_seen_entity = self.vae.decode(h_seen_label)

        # overall_loss = vib_loss + infonec_loss
        # overall_loss = vae_loss + vib_loss
        overall_loss = vae_loss + vib_loss + infonec_loss

        return h_sentence, h_image, h_mm, h_seen_label, h_unseen_label, h_seen_entity, h_unseen_entity, overall_loss