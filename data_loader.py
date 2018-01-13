import struct

def loadMnistData(filename, N):
    binfile = open(filename,'rb')
    buf = binfile.read()  
    index = 0  
    magic, numImages, numRows, numColums = struct.unpack_from('>IIII',buf,index)
    print('Loading Mnist data: ', magic,' ',numImages,' ',numRows,' ',numColums )
    index += struct.calcsize('>IIII')  

    for i in range(0, N):
        im = list(struct.unpack_from('>784B',buf,index))
        im = [int(m)/256 for m in im] # normalize input data to range(0,1)
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
        label = [0 for x in range(0,10)]
        label[int(im[0])] = 1
        yield label

def load_training_data(n_training_data):
    train_dataset = [m for m in loadMnistData ('MNIST_data/train-images.idx3-ubyte', n_training_data)]
    train_label   = [m for m in loadMnistLabel('MNIST_data/train-labels.idx1-ubyte', n_training_data)]
    training_data = list(zip(train_dataset, train_label))
    return training_data

def load_test_data(n_test_data):
    test_dataset  = [m for m in loadMnistData ('MNIST_data/t10k-images.idx3-ubyte', n_test_data)]
    test_label    = [m for m in loadMnistLabel('MNIST_data/t10k-labels.idx1-ubyte', n_test_data)]
    return test_dataset, test_label


    
