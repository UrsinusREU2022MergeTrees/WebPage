import numpy as np
import matplotlib.pyplot as plt

def backtrace(backpointers, node, involved):
    optimal = False
    for P in backpointers[node]:
        if backtrace(backpointers, (P[0], P[1]), involved):
            P[2] = True
            optimal = True
            involved.append([node[0], node[1]])
    if node[0] == 0 and node[1] == 0:
        return True #Reached the beginning
    return optimal

def DTW(X, Y, distfn):
    M = X.shape[0]
    N = Y.shape[0]
    CSM = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            CSM[i, j] = distfn(X[i, :], Y[j, :])
            
    backpointers = {}
    for i in range(1, M+1):
        for j in range(1, N+1):
            backpointers[(i, j)] = []
    backpointers[(0, 0)] = []

    D = np.zeros((M+1, N+1))
    D[1::, 0] = np.inf
    D[0, 1::] = np.inf
    for i in range(1, M+1):
        for j in range(1, N+1):
            d = CSM[i-1, j-1]
            dul = d + D[i-1, j-1]
            dl = d + D[i, j-1]
            du = d + D[i-1, j]
            D[i, j] = min(min(dul, dl), du)
            if dul == D[i, j]:
                backpointers[(i, j)].append([i-1, j-1, False])
            if dl == D[i, j]:
                backpointers[(i, j)].append([i, j-1, False])
            if du == D[i, j]:
                backpointers[(i, j)].append([i-1, j, False])
    involved = []
    backtrace(backpointers, (M, N), involved) #Recursive backtrace from the end
    return (D, CSM, backpointers, involved)


def drawLineColored(idx, x, C):
    for i in range(len(x)-1):
        plt.plot(idx[i:i+2], x[i:i+2], c=C[i, :])

def DTWExample():
    #Make dynamic time warping example
    np.random.seed(100)
    t1 = np.linspace(0, 1, 400)
    t1 = t1
    t2 = np.sqrt(t1)
    t1 = t1**2
    N = len(t1)
    
    X = np.cos(8*np.pi*t1) + t1
    Y = np.cos(8*np.pi*t2) + t2
    
    (D, CSM, backpointers, path) = DTW(X[:, None], Y[:, None], lambda x,y: np.abs(x - y).flatten())
    path = np.array(path)
    
    plotbgcolor = (0.15, 0.15, 0.15)
    plt.figure(figsize=(12, 5))

    for i in range(path.shape[0]):
        plt.clf()
        plt.subplot(121)
        plt.plot(np.arange(N), X, c='C0')
        plt.plot(np.arange(N), Y, c='C1')
        plt.xlim([0, N])
        plt.ylim([-1, 2])
        plt.title("Two Different Time Series")
        ax = plt.gca()
        #ax.set_facecolor(plotbgcolor)
        ax.scatter([path[i, 0]], X[path[i, 0]], c='C0')
        ax.scatter([path[i, 1]], Y[path[i, 1]], c='C1')
        ax.plot([path[i, 0], path[i, 1]], [X[path[i, 0]], Y[path[i, 1]]], c='k', linestyle='--')
        
        
        plt.subplot(122)
        plt.imshow(CSM, interpolation = 'nearest', cmap=plt.get_cmap('magma'), aspect = 'auto')
        plt.plot(path[0:i+1, 1], path[0:i+1, 0], 'c.')
        plt.xlim([-1, D.shape[1]])
        plt.ylim([D.shape[0], -1])
        plt.xlabel("Blue Curve")
        plt.ylabel("Orange Curve")
        plt.title("Cross-Similarity Matrix")
        
        #plt.show()
        plt.savefig("DTWExample{}.png".format(i), bbox_inches='tight')

if __name__ == '__main__':
    DTWExample()