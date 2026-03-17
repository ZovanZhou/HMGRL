import tensorflow as tf
from tensorflow.keras import initializers

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