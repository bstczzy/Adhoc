import numpy as np
import matplotlib.pyplot as plt

Dir = '/Users/zhenyuz/Documents/Projects/Davis Courses/loss_function_plots'

def exponential(x):
    return np.exp(x)

def quadratic(x):
    return (np.max(1-x,0))**2

def hinge(x):
    return max(1-x,0)

def sigmoid(x,k=1):
    return 1-np.tanh(x*k)

def plot_loss_function(loss,loss_name,x,plot_dir):
    y=[]
    for xi in x:
        y.append(loss(xi))
    fig = plt.figure(figsize=(16.0, 10.0))
    fig.add_subplot(111)
    plt.plot(x,y)

    plt.title("Loss Function %s" % (loss_name), fontsize=20)
    plt.xlabel("x", fontsize=16)
    plt.ylabel("y", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig('%s/loss_function_%s.png' %(plot_dir, loss_name), bbox_inches='tight')


xvec=[-i/50 for i in range(101)] + [i/50 for i in range(101)]
xvec.sort()

loss_functions=[exponential,quadratic,hinge,sigmoid]
loss_function_names=['exponential',
                      'truncated_quadratic',
                      'hinge',
                      'sigmoid']


for i in range(len(loss_functions)):
    plot_loss_function(loss_functions[i],loss_function_names[i],xvec,Dir)


