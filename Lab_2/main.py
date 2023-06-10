import sys
from PyQt5.QtWidgets import QApplication
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
J = 2000
M = 50
x0 = 0.35
y0 = 0.17
k = M / J
Vp = 0 #линия переключения
x = x0
y = y0
x1 = x0
x2 = y0
k1 = 0
k2 = 0
k3 = 0
k4 = 0
m1 = 0
m2 = 0
m3 = 0
m4 = 0
t = 0
t1 = 0
t2 = 0
dt = 0.1
u = 1
tVect = []
xVect = []
yVect = []
######################################
while abs(x1) > 0.1 or abs(x2) > 0.1:
    if x > 0:
        Vp = -np.sqrt(2*k*np.abs(x))
    else:
        Vp = np.sqrt(2*k*np.abs(x))

    if x2 < Vp or (x2 == Vp) < 0:
        u = +1
    else:
        u = -1
        
    k1 = x2 * dt
    m1 = u * k * dt
    k2 = (x2 + m1 / 2) * dt
    m2 = u * k * dt
    k3 = (x2 + m2 / 2) * dt
    m3 = u * k * dt
    k4 = (x2 + m3) * dt
    m4 = u * k * dt

    x1 += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    x2 += (m1 + 2 * m2 + 2 * m3 + m4) / 6

    x = x1
    y = x2
    tVect.append(t)
    xVect.append(x)
    yVect.append(y)

    if u == -1:
        t1 += dt
    else:
        t2 += dt
    #print(t1, t2)
    t += dt
print("Время переключения t1:", t1)
print("Время переключения t2:", t2)
print("Время перехода спутника в нулевое положение:", t)
fig = go.Figure()
fig.add_trace(go.Scatter(x=xVect, y=yVect))
fig.show()
####
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=tVect, y=xVect))# зависимость отклонения от времени 
fig1.show()
####
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=tVect, y=yVect))# Зависимость угловой скорости от времени 
fig2.show()
