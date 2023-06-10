import numpy as np
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
def calc(x, z1, z2):
    T1 = 1
    T2 = 3
    k = 1
    dt = 0.01
    y1 = z1
    k1 = dt*((k/T1*x) - (y1/T1))
    k2 = dt*((k/T1*x) - 1/T1*(y1+k1/2))
    k3 = dt*((k/T1*x) - 1/T1*(y1+k2/2))
    k4 = dt*((k/T1*x) - 1/T1*(y1+k3))
    z1 = z1 + 1/6*(k1+2*k2+2*k3+k4)
    y2=z2
    m1 = dt*(1/T2*y1 - 1/T2*y2)
    m2 = dt*(1/T2*y1 - 1/T2*(y2+m1/2))
    m3 = dt*(1/T2*y1 - 1/T2*(y2+m2/2))
    m4 = dt*(1/T2*y1 - 1/T2*(y2+m3))
    z2 = z2 + 1/6*(m1 + 2*m2 + 2*m3 + m4)
    return y2, z1, z2g = 1
###############################
L = 10
y, t_, x_prev = 0, 0, 0
y2, z1, z2 = 0, 0, 0
dt = 0.01
krit, prev_krit = 0, 0

#массивы для графиков
Y = []#ось у
X = []#ось х
H = []#запаздывание
G = []#без запаздывания
Krit = []#модульный критерий
L_mas = []#тактовый шаг
#графики q для первого опыта
# q(номер параметра)_номер опыта
q1_1 = []
q2_1 = []
q3_1 = []

q1_2 = []
q2_2 = []
q3_2 = []

q1_3 = []
q2_3 = []
q3_3 = []
#запаздывание
# i, j - цикл для моделирования запаздывания
# l - итератор для внешнего цикла
i, j, l, ns, tau, h = 0, 0, 0, 0, 0.3, 0.3
ns=tau/dt
# И. и Д. регуляторы
integ, dx = 0, 0
q1, q2, q3 = 2, 0.5, 1.5#эталонный
#q1, q2, q3 = 5, 4, 3
prev_q1, prev_q2, prev_q3 = 0, 0, 0
Q1 = []
Q2 = []
Q3 = []

xi_1, xi_2, xi_3 = 0, 0, 0
integ_xi1, integ_xi2, integ_xi3 = 0, 0, 0
Xi1 = []
Xi1_graph = []
Xi2 = []
Xi2_graph = []
Xi3 = []
Xi3_graph = []
prev_xi1, prev_xi2, prev_xi3 = 0, 0, 0
y2_xi1, z1_xi1, z2_xi1 = 0, 0, 0
y2_xi2, z1_xi2, z2_xi2 = 0, 0, 0
y2_xi3, z1_xi3, z2_xi3 = 0, 0, 0

dI1, dI2, dI3 = 0, 0, 0
Di1 = []
Di2 = []
Di3 = []

#попробуй dI1**2 + dI2**2 + dI3**2 > 0.001

#эталонное моделирование
epsilon = 1
while (epsilon > 0.001):
    i, j = 0, 0
    Krit.clear()
    X.clear()
    Y.clear
    H.clear()
    G.clear()
    Xi1.clear()
    Xi2.clear()
    Xi3.clear()
    t_ = 0
    y, x, x_prev = 0, 0, 0
    krit = 0
    integ, diff = 0, 0
    integ_xi1, integ_xi2, integ_xi3 = 0, 0, 0
    diff_xi1, diff_xi2, diff_xi3 = 0, 0, 0
    pid = 0
    du_1, du_2, du_3 = 0, 0, 0
    xi_1, xi_2, xi_3 = 0, 0, 0
    y2, z1, z2 = 0, 0, 0
    y2_xi1, z1_xi1, z2_xi1 = 0, 0, 0
    y2_xi2, z1_xi2, z2_xi2 = 0, 0, 0
    y2_xi3, z1_xi3, z2_xi3 = 0, 0, 0
    prev_xi1, prev_xi2, prev_xi3 = 0, 0, 0
    dI1, dI2, dI3 = 0, 0, 0
###############################################################################################
    while(t_ <= L):
        integ_xi1 += xi_1*dt
        integ_xi2 += xi_2*dt
        integ_xi3 += xi_3*dt
        diff_xi1 = (xi_1 - prev_xi1)/dt
        diff_xi2 = (xi_2 - prev_xi2)/dt
        diff_xi3 = (xi_3 - prev_xi3)/dt

        x = g - y
        integ += x*dt
        diff = (x - x_prev)/dt
        pid = q1*x + q2*integ + q3*diff

        du_1 = x - q1*xi_1 - q2*integ_xi1 - q3*diff_xi1
        du_2 = -q1*xi_2 + integ - q2*integ_xi2 - q3*diff_xi2
        du_3 = -q1*xi_3 - q2*integ_xi3 + diff - q3*diff_xi3

        y2, z1, z2 = calc(pid, z1, z2)
        y2_xi1, z1_xi1, z2_xi1 = calc(du_1, z1_xi1, z2_xi1)
        y2_xi2, z1_xi2, z2_xi2 = calc(du_2, z1_xi2, z2_xi2)
        y2_xi3, z1_xi3, z2_xi3 = calc(du_3, z1_xi3, z2_xi3)

        krit += abs(x)*dt

        prev_xi1 = xi_1
        prev_xi2 = xi_2
        prev_xi3 = xi_3

        if i > ns:
            if j >= ns:
                j = 0
            j += 1
            y = Y[j]
            Y[j] = y2
        #
            xi_1 = Xi1[j]
            Xi1[j] = y2_xi1
        #
            xi_2 = Xi2[j]
            Xi2[j] = y2_xi2
        #
            xi_3 = Xi3[j]
            Xi3[j] = y2_xi3
        else:
            Y.insert(i, y2)
            y = 0
        #
            Xi1.insert(i, y2_xi1)
            xi_1 = 0
        #
            Xi2.insert(i, y2_xi2)
            xi_2 = 0
        #
            Xi3.insert(i, y2_xi3)
            xi_3 = 0
        X.insert(i, t_)
        H.insert(i, y)
        G.insert(i, y2)
        Xi1.insert(i, xi_1)
        Xi2.insert(i, xi_2)
        Xi3.insert(i, xi_3)
        Krit.insert(i, krit)
        i += 1

        dI1 -= xi_1 * np.sign(x) * dt
        dI2 -= xi_2 * np.sign(x) * dt
        dI3 -= xi_3 * np.sign(x) * dt
        x_prev = x
        t_ += dt
##################################################################
    prev_q1 = q1
    prev_q2 = q2
    prev_q3 = q3
    
    if (krit <= prev_krit):#это до вычисления q
        h *= 1.2
    else:
        h *= 0.5
    prev_krit = krit

    q1 = prev_q1 - h*np.sign(dI1)
    q2 = prev_q2 - h*np.sign(dI2)   
    q3 = prev_q3 - h*np.sign(dI3)
    if (q1 < 0):
        q1 = prev_q1
    if (q2 < 0):
        q2 = prev_q2
    if (q3 < 0):
        q3 = prev_q3
    
    Di1.insert(l, dI1)
    Di2.insert(l, dI2)
    Di3.insert(l, dI3) 
    
    Q1.insert(l, q1)
    q1_1.insert(l, q1)

    Q2.insert(l, q2)
    q2_1.insert(l, q2)

    Q3.insert(l, q3)
    q3_1.insert(l, q3)
    
    epsilon = dI1**2 + dI2**2 + dI3**2
    print(epsilon)
    l += 1
    print(l)
    L_mas.append(l)

fig = go.Figure()
fig.add_trace(go.Scatter(x = X, y = H, name = 'с запаздыванием'))
fig.add_trace(go.Scatter(x = X, y = G, name = 'без запаздывания'))
fig.add_trace(go.Scatter(x = X, y = Krit, name = 'крит. (модул.)'))
fig.show()
########################
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x = X, y = Xi1, name = 'xi1'))
fig1.add_trace(go.Scatter(x = X, y = Xi2, name = 'xi2'))
fig1.add_trace(go.Scatter(x = X, y = Xi3, name = 'xi3'))
fig1.show()
########################
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x = L_mas, y = Q1, name = 'q1'))
fig2.add_trace(go.Scatter(x = L_mas, y = Q2, name = 'q2'))
fig2.add_trace(go.Scatter(x = L_mas, y = Q3, name = 'q3'))
fig2.show()
##################################
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x = L_mas, y = Di1, name = 'dI1'))
fig3.add_trace(go.Scatter(x = L_mas, y = Di2, name = 'dI2'))
fig3.add_trace(go.Scatter(x = L_mas, y = Di3, name = 'dI3'))
fig3.show()

#значения бóльших параметров q
q1, q2, q3 = 4, 2, 3
prev_q1, prev_q2, prev_q3 = 0, 0, 0
x = 0 
l = 0
epsilon = 1
L_mas.clear()
Di1.clear()
Di2.clear()
Di3.clear()
Q1.clear()
Q2.clear()
Q3.clear()
while (epsilon > 0.001):
    i, j = 0, 0
    Krit.clear()
    X.clear()
    Y.clear
    H.clear()
    G.clear()
    Xi1.clear()
    Xi2.clear()
    Xi3.clear()
    t_ = 0
    y, x, x_prev = 0, 0, 0
    krit = 0
    integ, diff = 0, 0
    integ_xi1, integ_xi2, integ_xi3 = 0, 0, 0
    diff_xi1, diff_xi2, diff_xi3 = 0, 0, 0
    pid = 0
    du_1, du_2, du_3 = 0, 0, 0
    xi_1, xi_2, xi_3 = 0, 0, 0
    y2, z1, z2 = 0, 0, 0
    y2_xi1, z1_xi1, z2_xi1 = 0, 0, 0
    y2_xi2, z1_xi2, z2_xi2 = 0, 0, 0
    y2_xi3, z1_xi3, z2_xi3 = 0, 0, 0
    prev_xi1, prev_xi2, prev_xi3 = 0, 0, 0
    dI1, dI2, dI3 = 0, 0, 0
###############################################################################################
    while(t_ <= L):
        integ_xi1 += xi_1*dt
        integ_xi2 += xi_2*dt
        integ_xi3 += xi_3*dt
        diff_xi1 = (xi_1 - prev_xi1)/dt
        diff_xi2 = (xi_2 - prev_xi2)/dt
        diff_xi3 = (xi_3 - prev_xi3)/dt

        x = g - y
        integ += x*dt
        diff = (x - x_prev)/dt
        pid = q1*x + q2*integ + q3*diff

        du_1 = x - q1*xi_1 - q2*integ_xi1 - q3*diff_xi1
        du_2 = -q1*xi_2 + integ - q2*integ_xi2 - q3*diff_xi2
        du_3 = -q1*xi_3 - q2*integ_xi3 + diff - q3*diff_xi3

        y2, z1, z2 = calc(pid, z1, z2)
        y2_xi1, z1_xi1, z2_xi1 = calc(du_1, z1_xi1, z2_xi1)
        y2_xi2, z1_xi2, z2_xi2 = calc(du_2, z1_xi2, z2_xi2)
        y2_xi3, z1_xi3, z2_xi3 = calc(du_3, z1_xi3, z2_xi3)

        krit += abs(x)*dt

        prev_xi1 = xi_1
        prev_xi2 = xi_2
        prev_xi3 = xi_3

        if i > ns:
            if j >= ns:
                j = 0
            j += 1
            y = Y[j]
            Y[j] = y2
        #
            xi_1 = Xi1[j]
            Xi1[j] = y2_xi1
        #
            xi_2 = Xi2[j]
            Xi2[j] = y2_xi2
        #
            xi_3 = Xi3[j]
            Xi3[j] = y2_xi3
        else:
            Y.insert(i, y2)
            y = 0
        #
            Xi1.insert(i, y2_xi1)
            xi_1 = 0
        #
            Xi2.insert(i, y2_xi2)
            xi_2 = 0
        #
            Xi3.insert(i, y2_xi3)
            xi_3 = 0
        X.insert(i, t_)
        H.insert(i, y)
        G.insert(i, y2)
        Xi1.insert(i, xi_1)
        Xi2.insert(i, xi_2)
        Xi3.insert(i, xi_3)
        Krit.insert(i, krit)
        i += 1

        dI1 -= xi_1 * np.sign(x) * dt
        dI2 -= xi_2 * np.sign(x) * dt
        dI3 -= xi_3 * np.sign(x) * dt
        x_prev = x
        t_ += dt
##################################################################
    prev_q1 = q1
    prev_q2 = q2
    prev_q3 = q3
    
    if (krit <= prev_krit):#это до вычисления q
        h *= 1.2
    else:
        h *= 0.5
    prev_krit = krit

    q1 = prev_q1 - h*np.sign(dI1)
    q2 = prev_q2 - h*np.sign(dI2)   
    q3 = prev_q3 - h*np.sign(dI3)
    if (q1 < 0):
        q1 = prev_q1
    if (q2 < 0):
        q2 = prev_q2
    if (q3 < 0):
        q3 = prev_q3
    
    Di1.insert(l, dI1)
    Di2.insert(l, dI2)
    Di3.insert(l, dI3) 
    
    Q1.insert(l, q1)
    q1_2.insert(l, q1)

    Q2.insert(l, q2)
    q2_2.insert(l, q2)

    Q3.insert(l, q3)
    q3_2.insert(l, q3)

    epsilon = dI1**2 + dI2**2 + dI3**2
    print(epsilon)
    l += 1
    print(l)
    L_mas.append(l)

fig = go.Figure()
fig.add_trace(go.Scatter(x = X, y = H, name = 'с запаздыванием'))
fig.add_trace(go.Scatter(x = X, y = G, name = 'без запаздывания'))
fig.add_trace(go.Scatter(x = X, y = Krit, name = 'крит. (модул.)'))
fig.show()
########################
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x = X, y = Xi1, name = 'xi1'))
fig1.add_trace(go.Scatter(x = X, y = Xi2, name = 'xi2'))
fig1.add_trace(go.Scatter(x = X, y = Xi3, name = 'xi3'))
fig1.show()
########################
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x = L_mas, y = Q1, name = 'q1'))
fig2.add_trace(go.Scatter(x = L_mas, y = Q2, name = 'q2'))
fig2.add_trace(go.Scatter(x = L_mas, y = Q3, name = 'q3'))
fig2.show()
##################################
axis_dI1 = np.array(Di1)
axis_dI2 = np.array(Di2)
axis_dI3 = np.array(Di3)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x = L_mas, y = Di1, name = 'dI1'))
fig3.add_trace(go.Scatter(x = L_mas, y = Di2, name = 'dI2'))
fig3.add_trace(go.Scatter(x = L_mas, y = Di3, name = 'dI3'))
fig3.show()

#значения меньших параметров q
q1, q2, q3 = 1, 0.1, 0.8
prev_q1, prev_q2, prev_q3 = 0, 0, 0
x = 0 
l = 0
epsilon = 1
L_mas.clear()
Di1.clear()
Di2.clear()
Di3.clear()
Q1.clear()
Q2.clear()
Q3.clear()
while (epsilon > 0.001):
    i, j = 0, 0
    Krit.clear()
    X.clear()
    Y.clear
    H.clear()
    G.clear()
    Xi1.clear()
    Xi2.clear()
    Xi3.clear()
    t_ = 0
    y, x, x_prev = 0, 0, 0
    krit = 0
    integ, diff = 0, 0
    integ_xi1, integ_xi2, integ_xi3 = 0, 0, 0
    diff_xi1, diff_xi2, diff_xi3 = 0, 0, 0
    pid = 0
    du_1, du_2, du_3 = 0, 0, 0
    xi_1, xi_2, xi_3 = 0, 0, 0
    y2, z1, z2 = 0, 0, 0
    y2_xi1, z1_xi1, z2_xi1 = 0, 0, 0
    y2_xi2, z1_xi2, z2_xi2 = 0, 0, 0
    y2_xi3, z1_xi3, z2_xi3 = 0, 0, 0
    prev_xi1, prev_xi2, prev_xi3 = 0, 0, 0
    dI1, dI2, dI3 = 0, 0, 0
###############################################################################################
    while(t_ <= L):
        integ_xi1 += xi_1*dt
        integ_xi2 += xi_2*dt
        integ_xi3 += xi_3*dt
        diff_xi1 = (xi_1 - prev_xi1)/dt
        diff_xi2 = (xi_2 - prev_xi2)/dt
        diff_xi3 = (xi_3 - prev_xi3)/dt

        x = g - y
        integ += x*dt
        diff = (x - x_prev)/dt
        pid = q1*x + q2*integ + q3*diff

        du_1 = x - q1*xi_1 - q2*integ_xi1 - q3*diff_xi1
        du_2 = -q1*xi_2 + integ - q2*integ_xi2 - q3*diff_xi2
        du_3 = -q1*xi_3 - q2*integ_xi3 + diff - q3*diff_xi3

        y2, z1, z2 = calc(pid, z1, z2)
        y2_xi1, z1_xi1, z2_xi1 = calc(du_1, z1_xi1, z2_xi1)
        y2_xi2, z1_xi2, z2_xi2 = calc(du_2, z1_xi2, z2_xi2)
        y2_xi3, z1_xi3, z2_xi3 = calc(du_3, z1_xi3, z2_xi3)

        krit += abs(x)*dt

        prev_xi1 = xi_1
        prev_xi2 = xi_2
        prev_xi3 = xi_3

        if i > ns:
            if j >= ns:
                j = 0
            j += 1
            y = Y[j]
            Y[j] = y2
        #
            xi_1 = Xi1[j]
            Xi1[j] = y2_xi1
        #
            xi_2 = Xi2[j]
            Xi2[j] = y2_xi2
        #
            xi_3 = Xi3[j]
            Xi3[j] = y2_xi3
        else:
            Y.insert(i, y2)
            y = 0
        #
            Xi1.insert(i, y2_xi1)
            xi_1 = 0
        #
            Xi2.insert(i, y2_xi2)
            xi_2 = 0
        #
            Xi3.insert(i, y2_xi3)
            xi_3 = 0
        X.insert(i, t_)
        H.insert(i, y)
        G.insert(i, y2)
        Xi1.insert(i, xi_1)
        Xi2.insert(i, xi_2)
        Xi3.insert(i, xi_3)
        Krit.insert(i, krit)
        i += 1

        dI1 -= xi_1 * np.sign(x) * dt
        dI2 -= xi_2 * np.sign(x) * dt
        dI3 -= xi_3 * np.sign(x) * dt
        x_prev = x
        t_ += dt
##################################################################
    prev_q1 = q1
    prev_q2 = q2
    prev_q3 = q3
    
    if (krit <= prev_krit):#это до вычисления q
        h *= 1.2
    else:
        h *= 0.5
    prev_krit = krit

    q1 = prev_q1 - h*np.sign(dI1)
    q2 = prev_q2 - h*np.sign(dI2)   
    q3 = prev_q3 - h*np.sign(dI3)
    if (q1 < 0):
        q1 = prev_q1
    if (q2 < 0):
        q2 = prev_q2
    if (q3 < 0):
        q3 = prev_q3
    
    Di1.insert(l, dI1)
    Di2.insert(l, dI2)
    Di3.insert(l, dI3) 
    
    Q1.insert(l, q1)
    q1_3.insert(l, q1)

    Q2.insert(l, q2)
    q2_3.insert(l, q2)

    Q3.insert(l, q3)
    q3_3.insert(l, q3)

    epsilon = dI1**2 + dI2**2 + dI3**2
    print(epsilon)
    l += 1
    print(l)
    L_mas.append(l)

fig = go.Figure()
fig.add_trace(go.Scatter(x = X, y = H, name = 'с запаздыванием'))
fig.add_trace(go.Scatter(x = X, y = G, name = 'без запаздывания'))
fig.add_trace(go.Scatter(x = X, y = Krit, name = 'крит. (модул.)'))
fig.show()
########################
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x = X, y = Xi1, name = 'xi1'))
fig1.add_trace(go.Scatter(x = X, y = Xi2, name = 'xi2'))
fig1.add_trace(go.Scatter(x = X, y = Xi3, name = 'xi3'))
fig1.show()
########################
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x = L_mas, y = Q1, name = 'q1'))
fig2.add_trace(go.Scatter(x = L_mas, y = Q2, name = 'q2'))
fig2.add_trace(go.Scatter(x = L_mas, y = Q3, name = 'q3'))
fig2.show()
##################################
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x = L_mas, y = Di1, name = 'dI1'))
fig3.add_trace(go.Scatter(x = L_mas, y = Di2, name = 'dI2'))
fig3.add_trace(go.Scatter(x = L_mas, y = Di3, name = 'dI3'))
fig3.show()

#графики изменений q
fig_1 = go.Figure()
fig_1.add_trace(go.Scatter(x = X, y = q1_1, name = 'q1 в 1-ом опыте'))
fig_1.add_trace(go.Scatter(x = X, y = q1_2, name = 'q1 в 2-ом опыте'))
fig_1.add_trace(go.Scatter(x = X, y = q1_3, name = 'q1 в 3-ем опыте'))
fig_1.show()
###################
fig_2 = go.Figure()
fig_2.add_trace(go.Scatter(x = X, y = q2_1, name = 'q2 в 1-ом опыте'))
fig_2.add_trace(go.Scatter(x = X, y = q2_2, name = 'q2 в 2-ом опыте'))
fig_2.add_trace(go.Scatter(x = X, y = q2_3, name = 'q2 в 3-ем опыте'))
fig_2.show()
###################
fig_3 = go.Figure()
fig_3.add_trace(go.Scatter(x = X, y = q3_1, name = 'q3 в 1-ом опыте'))
fig_3.add_trace(go.Scatter(x = X, y = q2_2, name = 'q3 в 2-ом опыте'))
fig_3.add_trace(go.Scatter(x = X, y = q3_3, name = 'q3 в 3-ем опыте'))
fig_3.show()
