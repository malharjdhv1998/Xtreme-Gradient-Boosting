import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

lam=.3
gamma=.5

df=pd.read_csv("data_1000.csv")
df.dropna(inplace=True)

xq=np.array(df.drop(["Y"],1))


y=np.array(df["Y"]).reshape(-1,1)
xq=preprocessing.scale(xq)


yt=np.zeros(y.shape)
xq=np.c_[xq,y,yt]

np.random.shuffle(xq)

xt=xq[0:int(.8*len(xq))]  #train
xe=xq[int(.8*len(xq)):len(xq)] #test


def g1(yt,y):
    return 2*(yt-y)

def best_split(n):#Sorting each feature calls bestsplit1
    m=[]
    for i in range(len(n[0])-2):
       
        n = n[n[:,i].argsort()]
        d=best_split1(n,i)
        m.append([d,d["score"]])
   
    m.sort(key= lambda a:a[1])
    return m[-1][0]


def best_split1(n,q):#Finding best example within each feature
   
    H=2*len(n)
    l=[]
    for i in range(0,len(n)):
        l.append(g1(n[i,-1],n[i,-2]))
    l=np.array(l)
    G=np.sum(l,axis=0)
    s=[]
    gl=0
    hl=0
    hr=0
    gr=0
    for i in range(0,len(n)):
        gl=gl+l[i]
        hl=hl+2
        gr=G-gl
        hr=H-hl
        sc=(gl**2)/(hl+lam)+(gr**2)/(hr+lam)-(G**2)/(H+lam)-gamma
        s.append(sc)
    s=np.array(s)
    s=s.reshape(-1,1)
    m,j = np.unravel_index(s.argmax(), s.shape)
 
 
    ma=max(s)
    xl=n[0:m,:]
    xr=n[m:len(n),:]
    d={}
    d["sample_index"]=m
    d["group"]=[xl,xr]
    d["val"]=n[m,q]
    d["score"]=ma
    d["feature_index"]=q
    d["lens"]=[len(xl),len(xr)]
   
    
    return d

def term(x):
    if len(x)!=0:
    
        m=0
        for i in range(0,len(x)):
            m=m+g1(x[i,-1],x[i,-2])
        return (-m)/(2*len(x)+lam)

k={}

def split_branch(node, max_depth, min_num_sample, depth):
#    print("split_branch")
#    print("depth",depth)
    left_node = node['group'][0]
    right_node = node['group'][1]
    del(node['group'])
    
    if len(left_node)==0 or len(right_node)==0:
#        print("case 0")
        if len(left_node)==0:
            node['left']=term(right_node)
            node['right']=term(right_node)
        elif len(right_node)==0:
            node['right']=term(left_node)
            node["left"]=term(left_node)
        return
   
    
    if depth >= max_depth :
#        print("1")
        xl=left_node
        xr=right_node
        node['left'] = term(left_node)
        node['right'] = term(right_node)
#        print(node['left'],node['right'])
#       
        k[node['left']]=xl
        k[node['right']]=xr
        return 
    if len(left_node) <= min_num_sample:
#        print("2")
     
        node['left'] = term(left_node)
#        print(node['left'])
    else :
#        print("3")
        node['left'] = best_split(left_node)
      
#        print("left=",node['left'])
        split_branch(node['left'], max_depth, min_num_sample, depth+1)
    if len(right_node) <= min_num_sample:
#        print("4")
        node['right'] = term(right_node)
#        print(node['right'])
    else:
#        print("5")
#        print("right",right_node)
        node['right'] = best_split(right_node)
#        print(node['right'])
        split_branch(node['right'], max_depth, min_num_sample, depth+1)


def build_tree(x, max_depth, min_num_sample):
    root = best_split(x) 
    split_branch(root, max_depth, min_num_sample, 1)
    return root

def display_tree(node, depth=0):
    if isinstance(node,dict):
        print('{}[sample_index{} < {:.2f}]'.format(depth*'\t',(node['sample_index']+1), node['val']))
        display_tree(node['left'], depth+1)
        display_tree(node['right'], depth+1)
    else:
        print('{}[{}]'.format(depth*'\t', node))

def predict_sample(node,sample):

    if sample[node["feature_index"]] <= node['val']:
        if isinstance(node['left'],dict):
            return predict_sample(node['left'],sample)
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict_sample(node['right'],sample)
        else:
            return node['right']

def predict(X,tree):
    y_pred =[]
    for i in range(0,len(X)):
        y_pred.append(predict_sample(tree,X[i]))
    return y_pred

def error(ypred,y):
    ypred.reshape(-1,1)
    y.reshape(-1,1)
    return abs(np.mean(ypred-y))*100   

t={}
print(build_tree(xt,8,3))
tree=build_tree(xt,8,3)
t[0]=tree

q = np.array(predict(xt,tree))

xt[:,-1]=q
e1=np.zeros((xe.shape[0],1))
e=np.zeros((xt.shape[0],1))



for i in range(1,2): 
    print("i==========================",i)
    tree=build_tree(xt,10,3)
    t[i]=tree
    
    s2 = np.array(predict(xt,tree))
    
    q=s2+q
   
    xt[:,-1]=q
   

#    e.append(error(xt[:,-1],xt[:,-2]))
#    e1.append(error(np.array(predict(xe,tree)),xe[:,-2]))

for i in range(0,len(t)):
    tree=t[i]
    e1=e1+np.array(predict(xe,tree)).reshape(-1,1)
    e=e+np.array(predict(xt,tree)).reshape(-1,1)

er=error(e1,xe[:,-2])
print(er)
print(e1[0:3],xe[0:3,-2])
print(error(e,xt[:,-2]))
print(e[0:3],xt[0:3,-2])


    
    
    

#print(tree)
#print("\n")
#print(predict(xt[0:2],tree),xt[0:2])
#
#
#print(predict(xt[0:5],tree),xt[0:5,-2])
#print(error(np.array(predict(xt[0:5],tree)),xt[0:5,-2]))
plt.figure(1)
plt.plot(xt[:,-2].reshape(-1,1)-e)
plt.show()
plt.figure(2)
plt.plot(e1-xe[:,-2].reshape(-1,1))
plt.show()

#plt.figure(2)
#plt.plot(e1)
#plt.show()

        