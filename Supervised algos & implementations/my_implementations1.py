# implementing KNN function here. The individual algorithm functions will be inste out seperately.
# Since Knn have no cluster centroids, it is difficult to test what version of K is suitable. 
# ##<Ram Note: add any supporting 3rd party link for future>

import numpy as np
from sortedcontainers import SortedList,SortedDict

from datetime import datetime

def get_data(limit=None):
    print("ingesting source data")
    df = pd.read_csv('D:/Datasets/digit-recognizer/train.csv')
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0 # data is from 0..255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y

class KNN(object):
    def __init__(self,k):
        self.k = k
    
    def fit(self,X,y):
        self.X=X
        self.y=y # lazy classifier and hence x and y are declarative

    def predict(self,x):
        y=np.zeros(len(x))
        for i,x in enumerate(x):
            sl=SortedList()
            for j, xt in enumerate(self.X):
                diff=x-xt
                d=diff.dot(diff)
                if len(sl)<self.k:
                    sl.add((d,self.y[j]))
                else:
                    if d<sl[-1][0]:
                        del sl[-1]
                        sl.add((d,self.y[j]))
            max_class=SortedDict()
            for ignore , y_pred in sl:
                if y_pred in max_class.keys():
                    max_class[y_pred]+=1
                else:
                    max_class[y_pred]=1
            y[i]=max_class.keys()[-1]
        return y
    
    def score(self, X, Y_0):
        Y=self.predict(X)
        return np.mean(Y == Y_0)

if __name__ == '__main__':
    X,Y =get_data(2000)
    ntrain=1000
    Xtrain,Ytrain=X[:ntrain],Y[:ntrain]
    Xtest,Ytest=X[ntrain:],Y[ntrain:]
    
    for k in range(2,11):
        tests=KNN(k)
        t0=datetime.now()
        tests.fit(Xtrain,Ytrain)
        print("-"*20+"\n k= "+str(k))
        print("fit time= "+str(datetime.now()-t0))
        
        t0=datetime.now()
        print("training accuracy:"+ str(tests.score(Xtrain,Ytrain)))
        print("time to train:" + str(datetime.now()-t0))

        t0=datetime.now()
        print("test accuracy:"+ str(tests.score(Xtest,Ytest)))
        print("time to test:" + str(datetime.now()-t0))





                