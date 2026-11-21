import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def objective(x):
    return 2 * np.sin(x) + 3* np.cos(2*x) +5 *np.sin(2/3*x)


n_iter = 10000
n = 10000
data_x = np.linspace(0, 4 * np.pi,n)
data_y = objective(data_x)

def simulated_annealing(objective, cooling_rate, n_iter):
    curr_x_iter = np.zeros(n_iter)
    curr_obj_iter = np.zeros(n_iter)
    best_x_iter = np.zeros(n_iter)
    best_obj_iter = np.zeros(n_iter)
    T_iter = np.zeros(n_iter)
    x_iter = np.zeros((n_iter,))
    obj_iter = np.zeros((n_iter,))

    T =3.0


    index = 7000
    curr_x = data_x[index]
    curr_obj = objective(curr_x)

    best_x = curr_x
    best_obj = curr_obj

    for i in range(n_iter):
        # down tempaliture
        T *= cooling_rate

        # choice new candidate 
        # index = np.random.choice(n)
        # new_x = data_x[index]
        new_x = np.clip(
            curr_x + np.random.uniform(-0.2, 0.2),
            0,
            4 * np.pi
        )
        new_obj = objective(new_x)

        # wether move to new point from current one
        if(new_obj >curr_obj) or (np.random.rand() < np.exp((new_obj -curr_obj) /T)):
            curr_x = new_x
            curr_obj = new_obj

        if new_obj > best_obj:
            best_x = new_x
            best_obj = new_obj

        curr_x_iter[i] = curr_x
        curr_obj_iter[i] = curr_obj
        best_x_iter[i] = best_x
        best_obj_iter[i] = best_obj
        T_iter[i] = T
        x_iter[i] = best_x
        obj_iter[i] = best_obj

    return  (
        curr_x_iter, curr_obj_iter,
        best_x_iter, best_obj_iter, T_iter
    )


curr_x, curr_obj, best_x, best_obj, temperatures = (
    simulated_annealing(objective, 0.999, n_iter)
)

# drwaing
fig,(ax1, ax2) = plt.subplots(1,2,figsize=(13,5))

# objective function and the current points
ax1.plot(data_x, data_y, color="gray",label= "Objective")
current_point, = ax1.plot([],[],"bo",markersize=9, label="Current")
best_point, = ax1.plot([],[],"r*",markersize=14,label="Best")


ax1.set_xlim(0,4*np.pi)
ax1.set_ylim(data_y.min()-1,data_y.max()+1)
ax1.set_xlabel("x")
ax1.set_ylabel("Objective")
ax1.legend()
ax1.grid(True)

ax2.plot(range(n_iter), curr_obj, color="blue",
         alpha=0.25, label="Current history")
ax2.plot(range(n_iter), best_obj, color="red",
         alpha=0.25, label="Best history")

current_history, = ax2.plot([], [], "b-", linewidth=2)
best_history, = ax2.plot([], [], "r-", linewidth=2)

ax2.set_xlim(0, n_iter)
ax2.set_ylim(data_y.min()-1, data_y.max()+1)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Objective value")
ax2.legend()
ax2.grid(True)

def update(i):
    current_point.set_data([curr_x[i]], [curr_obj[i]])
    best_point.set_data([best_x[i]], [best_obj[i]])

    current_history.set_data(
        np.arange(i+1), curr_obj[:i+1]
    )
    best_history.set_data(
        np.arange(i+1), best_obj[:i+1]
    )

    ax1.set_title(
        f"Iteration: {i+1} | T = {temperatures[i]:.5f}"
    )
    ax2.set_title("Optimization history")

    return (
        current_point, best_point,
        current_history, best_history
    )

ani = FuncAnimation(
    fig,
    update,
    frames=n_iter,
    interval=200,
    repeat=False
)

plt.tight_layout()
plt.show()

print(f"Best x: {best_x[-1]:.6f}")
print(f"Best objective: {best_obj[-1]:.6f}")


print("best x:{}, best objective: {}".format(x_iter[best_index],obj_iter[best_index]))