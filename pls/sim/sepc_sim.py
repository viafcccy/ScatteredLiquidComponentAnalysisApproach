import numpy as np

# 计算SID
def SID(x,y):
    # print(np.sum(x))
    p = np.zeros_like(x,dtype=np.float)
    q = np.zeros_like(y,dtype=np.float)
    Sid = 0
    for i in range(len(x)):
        p[i] = x[i]/np.sum(x)
        q[i] = y[i]/np.sum(y)
        # print(p[i],q[i])
    for j in range(len(x)):
        Sid += p[j]*np.log10(p[j]/q[j])+q[j]*np.log10(q[j]/p[j])
    return Sid

# 计算SAM
def SAM(x,y):
    s = np.sum(np.dot(x,y))
    t = np.sqrt(np.sum(x**2))*np.sqrt(np.sum(y**2))
    th = np.arccos(s/t)
    # print(s,t)
    return th