import tensorflow as tf

def random_t(batch_size, t_max = 1000):
    t = tf.random.uniform((batch_size,), minval = 0, maxval = t_max, dtype = tf.int32) # random t
    return t

def linear_beta_schedule(t, beta_start = 0.0001, beta_end = 0.02, T = 1000):
    t_float = tf.cast(t, tf.float32)
    beta = beta_start + (beta_end - beta_start) * (t_float/(T - 1.))  #culculate beta
    return beta

def sample(x0, t):
    beta = linear_beta_schedule(t)
    alpha = 1.0 - beta
    sqrt_alpha = tf.sqrt(alpha)
    one_min_alpha = tf.sqrt(1.0 - alpha)
    #reshape
    sqrt_alpha = sqrt_alpha[:, None, None, None]
    one_min_alpha = one_min_alpha[:, None, None, None]

    # noisy
    noisy = tf.random.normal(tf.shape(x0))

    #add noisy
    x_t = sqrt_alpha * x0 + one_min_alpha *noisy

    return x_t

def map_func(x0):
    batch = tf.shape(x0)[0]
    t = random_t(batch)
    x_t = sample(x0, t)
    result = ({"input_image": x_t, "time_input": t}, x0)

    return result



