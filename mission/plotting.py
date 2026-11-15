"""Our formation and baseline plots; matplotlib is optional."""
def plot_formation(problem,x):
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(6,3,figsize=(11,16))
    names = ["XY","XZ","YZ"]
    for epoch,observation in enumerate(problem.observations(x)):
        for plane,data in enumerate(observation["planes"]):
            for row,key in [(2*epoch,"cells"),(2*epoch+1,"baselines")]:
                ax = axes[row][plane]
                if data[key]:
                    a,b = zip(*data[key])
                    ax.scatter(a,b,marker="s",s=8)
                if key=="cells":
                    ax.set_xlim(-0.5,problem.grid_size-0.5)
                    ax.set_ylim(-0.5,problem.grid_size-0.5)
                    ax.set_title(f"t={epoch}T, {names[plane]}, retained={observation['remaining']}")
                else:
                    edge = problem.grid_size-0.5
                    ax.set_xlim(-edge,edge)
                    ax.set_ylim(-edge,edge)
                    ax.set_title(f"Fill={data['fill']:.6f}")
                ax.set_aspect("equal")
    fig.tight_layout()
    return fig
