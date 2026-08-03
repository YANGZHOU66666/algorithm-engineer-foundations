# numpy

## numpy.array

### 创建数组

```python
import numpy as np
# 从python原生数组创建
a = np.array([1, 2, 3])           # 一维数组
b = np.array([[1, 2], [3, 4]])    # 二维数组

# 内置构造函数创建
np.zeros((2, 3))       # 创建全0数组
np.ones((2, 3))        # 创建全1数组
np.full((2, 3), 7)     # 创建值为7的数组
np.eye(3)              # 单位矩阵
np.arange(0, 10, 2)    # 从0到10（不含10），步长为2
np.linspace(0, 1, 5)   # 线性等间隔生成5个数

```



### 数组计算

- `np.dot()`

```python
import numpy as np
# 向量*向量: 点积
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)   # 1*4 + 2*5 + 3*6 = 32

# 矩阵·向量: 注意向量是竖着的
A = np.array([[1, 2], [3, 4]])
v = np.array([5, 6]) # 竖着的向量, shape为(2,), 如果要横着的应该是[[5, 6]], shape为(1,2)
np.dot(A, v)  # 2x2 矩阵 × 2维向量 → 2维向量

# 矩阵·矩阵
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
np.dot(A, B)


# 广播机制
num = 5
np_arr4 = num+np_arr3
# 对np.ndarray可以直接进行常规运算，等价于每个数字进行运算
np_arr5 = (np_arr4)**2 # 乘方
def sigmoid(arr):
    return 1/(1+np.exp(-arr))
np_arr6 = sigmoid(np_arr4)
```

- `np.dot()`和`np.matmul()`区别

```python
# 对三维数组，np.matmul()计算的是相对标注的按后两维矩阵相乘
import numpy as np
a = np.ones((2, 3, 4))
b = np.ones((2, 4, 5))
print(np.matmul(a, b))
'''结果:
[[[4. 4. 4. 4. 4.]
  [4. 4. 4. 4. 4.]
  [4. 4. 4. 4. 4.]]

 [[4. 4. 4. 4. 4.]
  [4. 4. 4. 4. 4.]
  [4. 4. 4. 4. 4.]]]
'''

# np.dot()不知道算的是什么
import numpy as np
a = np.ones((2, 3, 4))
b = np.ones((2, 4, 5))
print(np.dot(a, b))  # 错！维度不允许这样直接乘
'''结果:
[[[[4. 4. 4. 4. 4.]
   [4. 4. 4. 4. 4.]]

  [[4. 4. 4. 4. 4.]
   [4. 4. 4. 4. 4.]]

  [[4. 4. 4. 4. 4.]
   [4. 4. 4. 4. 4.]]]


 [[[4. 4. 4. 4. 4.]
   [4. 4. 4. 4. 4.]]

  [[4. 4. 4. 4. 4.]
   [4. 4. 4. 4. 4.]]

  [[4. 4. 4. 4. 4.]
   [4. 4. 4. 4. 4.]]]]
'''
```

- 广播机制

```python
import numpy as np

a = np.array([1, 2, 3])
b = 10

result = a + b   # 自动把 10 广播为 [10, 10, 10]
print(result)    # 输出: [11 12 13]

# 矩阵和标量的广播机制：又变成横着的向量了
import numpy as np
a = np.array([[1,2],[1,2]])
b = np.array([1,2])
print(a+b)
'''
[[2 4]
 [2 4]]
'''
```



## 计算

一些内置的公式，可以用在矩阵/向量/标量上：

```python
import numpy as np
np.exp(1)
np.log(10) # 以e为底
np.log2(4)
np.log10(1000)
```

求和np.sum()/平均np.mean()：

```python
import numpy as np
a = np.array([[1,2],[1,2]])
print(np.sum(a)) # 6
print(np.sum(a,axis=0)) # [2, 4]
print(np.sum(a,axis=1)) # [3, 3]

print(np.mean(a)) # 1.5
print(np.mean(a,axis=0)) # [1, 2]
print(np.mean(a,axis=1)) # [1.5, 1.5]
```

