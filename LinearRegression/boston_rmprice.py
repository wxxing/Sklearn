import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


boston = load_boston()
bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target

X = bos.drop('PRICE', axis=1)
lm = LinearRegression()

lm.fit(X, bos.PRICE)
print('线性回归算法w值：', lm.coef_)
print('线性回归算法b值: ', lm.intercept_)

plt.scatter(bos.RM, bos.PRICE)
plt.xlabel(u'住宅平均房间数')
plt.ylabel(u'房屋价格')
plt.title(u'RM与PRICE的关系')
plt.show()


print(lm.predict(X)[0:5])
mse = np.mean((bos.PRICE - lm.predict(X)) ** 2)
print(mse)