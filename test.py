import tensorflow as tf

# 创建测试张量
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 调整 b 的形状
b = tf.transpose(b)  # 将 b 转置以获得形状 [3, 2]

# 执行矩阵乘法
c = tf.matmul(a, b)
print("Tensor computation result:", c)

