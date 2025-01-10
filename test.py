import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# 定义线性 beta 计算函数
def linear_beta_schedule(t, beta_start=0.0001, beta_end=0.02, T=1000):
    t_float = tf.cast(t, tf.float32)
    beta = beta_start + (beta_end - beta_start) * (t_float / (T - 1.))
    return beta

# 定义 sample 函数
def sample(x0, t):
    # 如果 t 是标量张量，确保转换为 1D 张量
    if tf.rank(t) == 0:
        t = tf.reshape(t, [1])

    beta = linear_beta_schedule(t)
    alpha = 1.0 - beta
    sqrt_alpha = tf.sqrt(alpha)
    sqrt_alpha = tf.cast(sqrt_alpha, tf.float32)
    one_min_alpha = tf.sqrt(1.0 - alpha)
    one_min_alpha = tf.cast(one_min_alpha, tf.float32)

    # 保证维度正确
    sqrt_alpha = sqrt_alpha[:, None, None, None]
    one_min_alpha = one_min_alpha[:, None, None, None]

    # 添加噪声
    noisy = tf.random.normal(tf.shape(x0))
    noisy = tf.cast(noisy, tf.float32)

    # 计算噪声图像
    x_t = sqrt_alpha * x0 + one_min_alpha * noisy
    return x_t

# 加载图像
train_data_path = "/Users/gangjiahong/Downloads/data1/3.県品評会/第14回千葉県若鯉品評会/1.15部総合優勝.jpg"
train_data = cv2.imread(train_data_path)
train_data = cv2.cvtColor(train_data, cv2.COLOR_BGR2RGB)
ht_img, wd_img = 512, 512
train_data = cv2.resize(train_data, (ht_img, wd_img))
train_data = tf.convert_to_tensor(train_data, dtype=tf.float32) / 255.0

# 确保 train_data 为 4D 张量
train_data = tf.expand_dims(train_data, axis=0)

# 正确传递 t (作为标量张量)
t = tf.constant(10000, dtype=tf.float32)  # 正确的标量张量

# 调用 sample 函数
train_data_noisy = sample(train_data, t)

# 显示图像
train_data_noisy = tf.squeeze(train_data_noisy)  # 去掉 batch 维度
train_data_noisy = tf.clip_by_value(train_data_noisy, 0.0, 1.0)

plt.figure(figsize=(10, 8))
plt.title("Noisy Image")
plt.imshow(train_data_noisy.numpy())
plt.show()

















