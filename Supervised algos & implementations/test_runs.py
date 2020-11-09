import numpy as np
import pandas as pd
from my_implementations1 import KNN
from synthetic_inputs import gen_xor, gen_donuts
from sklearn.model_selection import train_test_split
from datetime import datetime

def main():
    n=100
    x,y=gen_donuts(n=n)
    xtrain, xtest,ytrain, ytest = train_test_split(x,y,test_size=0.4)
    for i in range(1,5):
        t0=datetime.now()
        knn_model=KNN(i)
        knn_model.fit(xtrain,ytrain)
        print('training time: '+str(datetime.now()-t0))
        print("training accuracy: "+ str(knn_model.score(xtrain,ytrain)))
        
        t0=datetime.now()
        print("test accuracy: "+ str(knn_model.score(xtest,ytest)))
        print('test time: '+str(datetime.now()-t0))

main()
    