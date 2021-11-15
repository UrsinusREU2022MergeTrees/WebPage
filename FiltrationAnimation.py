import numpy as np
from ripser import ripser
from persim import plot_diagrams as plot_dgms
import matplotlib.pyplot as plt
from scipy import sparse


def lower_star_filtration(x):
    N = len(x)
    # Add edges between adjacent points in the time series, with the "distance" 
    # along the edge equal to the max value of the points it connects
    I = np.arange(N-1)
    J = np.arange(1, N)
    V = np.maximum(x[0:-1], x[1::])
    # Add vertex birth times along the diagonal of the distance matrix
    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, x))
    #Create the sparse distance matrix
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    return ripser(D, maxdim=0, distance_matrix=True)['dgms'][0]

def make_filtration_animation():
    np.random.seed(1)
    NPeriods = 5
    NSamples = 100
    t = np.linspace(-0.5, NPeriods, NSamples)
    x = np.sin(2*np.pi*t) + t

    H0 = lower_star_filtration(x)

    births = H0[:, 0]
    deaths = H0[:, 1]
    N = 100
    cutoffs = np.linspace(np.min(x)-0.1, np.max(x)+0.1, N)
    cutoffs = np.concatenate((cutoffs, births, deaths))
    idxs = np.argsort(cutoffs)



    fig = plt.figure(figsize=(9.5, 3))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    pidx = 0
    for ii in range(len(cutoffs)):
        birth = False
        death = False
        title = "Regular Point"
        bdidx = -1
        if idxs[ii] >= N:
            if idxs[ii] < N+H0.shape[0]:
                birth = True
                title = "Birth!"
                bdidx = idxs[ii] - N
            else:
                death = True
                title = "Death!"
                bdidx = idxs[ii] - N - H0.shape[0]
        cutoff = cutoffs[idxs[ii]]
        ax1.clear()
        ax2.clear()

        ax1.plot(x)
        ax1.plot([0, len(x)], [cutoff, cutoff])
        ax1.set_ylim([np.min(x)-0.2, np.max(x)+0.2])
        #ax1.scatter(np.arange(len(x)), x)
        for i in range(len(x)):
            if x[i] <= cutoff:
                ax1.scatter([i]*2, [x[i]]*2, c='C1')
            if i < len(x)-1:
                if x[i] <= cutoff and x[i+1] <= cutoff:
                    ax1.plot([i, i+1], x[i:i+2], c='C1')
        plt.sca(ax2)
        plot_dgms(H0)
        ax2.plot([np.min(x)-0.1, cutoff], [cutoff, cutoff], linestyle='--', c='C1')
        ax2.plot([cutoff, cutoff], [cutoff, np.max(x)+0.6], linestyle='--', c='C1')
        ax1.set_title(title)
        if birth or death:
            ax2.scatter([H0[bdidx, 0]], [H0[bdidx, 1]], 100)
            for i in range(10):
                plt.savefig("Filtration{}.png".format(pidx), bbox_inches='tight')
                pidx += 1
        plt.savefig("Filtration{}.png".format(pidx), bbox_inches='tight')
        pidx += 1


def make_warp_animation():
    np.random.seed(1)
    NPeriods = 5
    NSamples = 400
    t = np.linspace(-0.5, NPeriods, NSamples)
    x = np.sin(2*np.pi*t) + t
    x += np.random.randn(NSamples)

    warps = np.logspace(np.log10(0.6), np.log10(4), 100)
    warps = np.concatenate((warps, warps[::-1]))

    fig = plt.figure(figsize=(6, 3))
    ax1 = plt.subplot(111)
    #ax2 = plt.subplot(122)


    for i, p in enumerate(warps):   
        ax1.clear()
        #ax2.clear()
        
        # Get slider values
        t = np.linspace(0, 1, NSamples)**p
        t = (t*(NPeriods+0.5))-0.5
        x = np.sin(2*np.pi*t) + t #+ 0.1*np.random.randn(NSamples)
        
        ax1.plot(x)
        ax1.set_ylim(-1.5, 6)
        #ax1.scatter(np.arange(len(x)), x)
        H0 = lower_star_filtration(x)
        #plt.sca(ax2)
        #plot_dgms(H0)
        plt.savefig("Warp{}.png".format(i), bbox_inches='tight', facecolor='white')

if __name__ == '__main__':
    #make_filtration_animation()
    make_warp_animation()