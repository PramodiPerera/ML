import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



def gradient_descent(x,y):
    # starting parameter values
    m_curr = b_curr = 0

    learning_rate = 0.0002
    m = len(x)

    pre_cost = cost = 0
    i = 0
    while(True):
        y_predict = m_curr * x + b_curr
        cost = (1/m) * sum([val**2 for val in (y-y_predict)])

        # stop iterations when cost is close to pre cost value
        if i != 0:
            if math.isclose(cost, pre_cost):
                break

        # calculate derivatives
        m_derv = -(2/m)*sum(x*(y-y_predict))
        b_derv = -(2/m)*sum(y-y_predict)

        # record prev_cost
        pre_cost = cost

        # update parameters
        m_curr = m_curr - learning_rate * m_derv
        b_curr = b_curr - learning_rate * b_derv

        i = i+1
        print("m {}, b {}, cost {}, iterations {}". format(m_curr, b_curr, cost, i))


# load data
df = pd.read_csv('test_scores.csv')
x = np.array(df['math'])
y = np.array(df['cs'])

gradient_descent(x, y)




