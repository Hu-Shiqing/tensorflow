import struct  
#import numpy as np   
import matplotlib.pyplot as plt   
import tkinter as tk;
  
# ----------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------
def loadMnistData(filename, N):
    binfile = open(filename,'rb')
    buf = binfile.read()  
    index = 0  
    magic, numImages, numRows, numColums = struct.unpack_from('>IIII',buf,index)
    print('Loading Mnist data: ', magic,' ',numImages,' ',numRows,' ',numColums )
    index += struct.calcsize('>IIII')  

    for i in range(0, N):
        im = struct.unpack_from('>784B',buf,index)
        index += struct.calcsize('>784B' )
        yield im

def loadMnistLabel(filename, N):
    binfile = open(filename,'rb')
    buf = binfile.read()  
    index = 0  
    magic, numImages = struct.unpack_from('>II',buf,index)
    print('Loading Mnist label:', magic,' ',numImages)
    index += struct.calcsize('>II')  

    for i in range(0, N):
        im = struct.unpack_from('>1B',buf,index)
        index += struct.calcsize('>1B' )
        yield im[0]

def makeMatrix(lst, m, n=-1):
    matrix=list()
    if n == -1:
        n = m;
    for i in range(0, m):
        row=list()
        for j in range(0,n):
            row.append(lst[i*n+j])
        matrix.append(row)
    return matrix

def printMatrix(m):
    for i in m:
        row = ''
        for j in i:
            row = row + ' %3d'%(j)
        print(row)

'''
def plotMatrix(m):
    fig = plt.figure()  
    plt.imshow(m,cmap = 'binary')
    plt.show()
'''

def plotMatrix(A, scale = 5, margine = 20):
    root = tk.Tk()
    canvas = tk.Canvas(root, width = scale*len(A)+2*margine, height = scale*len(A[0])+2*margine, bg = 'white')
    for i in range(0, len(A)): # this is Y
        for j in range(0, len(A[0])): # this is X
            canvas.create_rectangle(margine+scale*j,     margine+scale*i,
                                    margine+scale*(j+1), margine+scale*(i+1), 
                                    fill='#%02x%02x%02x'%(255-A[i][j],255-A[i][j],255-A[i][j]), outline='')
    canvas.pack()
    root.mainloop()  

def plotMatrixList(AL, scale = 5, margine = 20):
        root = tk.Tk()
        for A in AL:
                canvas = tk.Canvas(root, width = scale*len(A)+2*margine, height = scale*len(A[0])+2*margine, bg = 'white')
                for i in range(0, len(A)): # this is Y
                      for j in range(0, len(A[0])): # this is X
                              canvas.create_rectangle(margine+scale*j,     margine+scale*i,
                                                           margine+scale*(j+1), margine+scale*(i+1), 
                                                          fill='#%02x%02x%02x'%(255-A[i][j],255-A[i][j],255-A[i][j]), outline='')
                canvas.pack()
        root.mainloop()  
# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
dataset = [m for m in loadMnistData('train-images.idx3-ubyte' , 1000)]
label   = [m for m in loadMnistLabel('train-labels.idx1-ubyte', 1000)]

plotMatrixList([makeMatrix(m, 28) for m in dataset[60:65]])

'''
for i in range(0,10):
      print('%d.count = %d'%(i, label.count(i)))

k = 9
print(label[k]);A=makeMatrix(dataset[k], 28)
#printMatrix(A)
plotMatrix(A)
'''

#import pydotplus
#from sklearn.datasets import load_iris
from sklearn import tree
#from sklearn.externals.six import StringIO

clf = tree.DecisionTreeClassifier()
clf = clf.fit(dataset,label)

'''
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("mnist.pdf")
'''

pred = clf.predict([m for m in dataset[60:65]])
for index in pred: 
    print(index)
#plotMatrixList([makeMatrix(dataset[61], 28),makeMatrix(dataset[60], 28)])
