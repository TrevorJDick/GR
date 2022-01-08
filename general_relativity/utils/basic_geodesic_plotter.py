# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


def geodesic_plotter_3d(x1, x2, x3, axes_names=['X', 'Y', 'Z']):
    fig, axs = plt.subplots(1, 2, sharex=False, sharey=False)

    axs[0].scatter(x1, x2)
    axs[0].set_xlabel(axes_names[0])
    axs[0].set_ylabel(axes_names[1])
    axs[0].scatter(
        x1[0],
        x2[0],
        marker='_',
        color='black',
        s=100,
        label='start'
    )
    axs[0].scatter(
        x1[-1],
        x2[-1],
        marker='^',
        color='black',
        s=100,
        label='end'
    )
    
    axs[1].scatter(x1, x3)
    axs[1].set_xlabel(axes_names[0])
    axs[1].set_ylabel(axes_names[2])
    axs[1].scatter(
        x1[0],
        x3[0],
        marker='_',
        color='black',
        s=100,
        label='start'
    )
    axs[1].scatter(
        x1[-1],
        x3[-1],
        marker='^',
        color='black',
        s=100,
        label='end'
    )
    axs[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    del fig, axs


    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x1, x2, x3, s=80)

    ax.plot(x1, x3, 'r', zdir='y', zs=x2.max(), lw=1, alpha=0.75)
    ax.plot(x2, x3, 'g', zdir='x', zs=x1.min(), lw=1, alpha=0.75)
    ax.plot(x1, x2, 'k', zdir='z', zs=x3.min(), lw=1, alpha=0.75)
    ax.scatter(
        x1[0],
        x2[0],
        x3[0],
        marker='s',
        color='black',
        s=750,
        label='start'
    )
    ax.scatter(
        x1[-1],
        x2[-1],
        x3[-1],
        marker='^',
        color='black',
        s=750,
        label='end'
    )
    ax.legend(loc='upper right')

    ax.set_xlabel(axes_names[0])
    ax.set_ylabel(axes_names[1])
    ax.set_zlabel(axes_names[2])
    plt.show()
    del fig, ax
    return

