import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    

def gen_xor(quad_var=1,n=100):
    x0=np.zeros(n*4)
    x1=np.zeros(n*4)
    y=np.zeros(n*4)
    means_x0=[0,-1,-1,0]
    means_x1=[0,0,-1,-1]
    val_y=[1,0,1,0]
    for i in range(4):
        x0[(i*n):((i+1)*n)]=(np.random.random_sample(n)+means_x0[i])*quad_var
        x1[(i*n):((i+1)*n)]=(np.random.random_sample(n)+means_x1[i])*quad_var
        y[(i*n):((i+1)*n)]=np.full(n,val_y[i])
    fin_x=pd.DataFrame(data={'x0':x0,'x1':x1})
    fin_y=np.array(y)
    return fin_x.values,fin_y

def gen_donuts(inner_radius=1,outer_radius=2,n=100):
    #initialization
    b,a=outer_radius,inner_radius
    x0=np.zeros(n*2)
    x1=np.zeros(n*2)
    y=np.zeros(n*2)
    theta=2*np.pi*np.random.rand(n*2)
    
    #Outer doughnut in first n values
    x0[:n]=(np.random.random(n)+a)*(b-a)*(np.cos(theta[:n]))
    x1[:n]=(np.random.random(n)+a)*(b-a)*(np.sin(theta[:n]))
    y[:n]=np.ones(n)

    #inner circle in next n values
    x0[n:]=np.random.random(n)*a*(np.cos(theta[n:]))
    x1[n:]=np.random.random(n)*a*(np.sin(theta[n:]))
    y[n:]=np.zeros(n)

    fin_x=pd.DataFrame(data={'x0':x0,'x1':x1})
    fin_y=np.array(y)
    return fin_x.values,fin_y


if __name__=='__main__':
    data=gen_donuts(n=200)
    plt.scatter(x=data['x0'],y=data['x1'],c=data['y']*255)
    plt.show()