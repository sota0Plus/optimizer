"""Run: python pso_visualization.py (requires numpy and matplotlib)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Problem:
    @staticmethod
    def objective(x, y):
        return (418.9829*2 -(x*np.sin(np.sqrt(np.abs(x)))) - y * np.sin(np.sqrt(np.abs(y))))


class Particle:
    def __init__(self):
        self.velocity = np.zeros(2)
        self.point = np.random.uniform(-500, 500, 2)
        self.personal_best = self.point.copy()

    def update_velocity(self, g_best):
        inertia = 0.7
        personal_acceleration = 1.5
        global_acceleration = 1.5
        r1 = np.random.rand()
        r2 = np.random.rand()
        self.velocity = (
            inertia * self.velocity
            + personal_acceleration * r1 * (self.personal_best - self.point)
            + global_acceleration * r2 * (g_best - self.point)
        )

    def update_position(self):
        self.point += self.velocity

    def update_best(self):
        current_score = Problem.objective(*self.point)
        best_score = Problem.objective(*self.personal_best)
        if current_score > best_score:
            self.personal_best = self.point.copy()
        return self.personal_best


class Swarm:
    def __init__(self, particle_num):
        if particle_num < 1:
            raise ValueError("particle_num must be at least 1")
        self.particles = [Particle() for _ in range(particle_num)]
        self.g_best = self.particles[0].personal_best.copy()
        self.update_best()

        # Save step 0 as well. Copies keep old frames from changing later.
        self.position_history = []
        self.g_best_history = []
        self.score_history = []
        self.record_history()

    def update_best(self):
        for particle in self.particles:
            if Problem.objective(*particle.personal_best) > Problem.objective(*self.g_best):
                self.g_best = particle.personal_best.copy()

    def step(self):
        for particle in self.particles:
            particle.update_velocity(self.g_best)
            particle.update_position()
            particle.update_best()
        self.update_best()
        self.record_history()

    def record_history(self):
        self.position_history.append(
            np.array([particle.point.copy() for particle in self.particles])
        )
        self.g_best_history.append(self.g_best.copy())
        self.score_history.append(Problem.objective(*self.g_best))

    def run_pso(self, iterations):
        for _ in range(iterations):
            self.step()

    def print_best(self):
        print(f"best_position: {self.g_best}")
        print(f"best_score: {Problem.objective(*self.g_best):.12g}")

    def animate(self, interval=80):
        # Playback uses recorded data, so redraws never advance the PSO.
        positions = np.array(self.position_history)
        bests = np.array(self.g_best_history)
        scores = np.array(self.score_history)
        steps = np.arange(len(scores))

        fig = plt.figure(figsize=(13, 5.5), layout="constrained")
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        curve_ax = fig.add_subplot(1, 2, 2)

        # Fixed axes include every recorded position and this problem's optimum.
        all_points = np.vstack((positions.reshape(-1, 2), bests, [[2, 3]]))
        lower = all_points.min(axis=0)
        upper = all_points.max(axis=0)
        margin = np.maximum((upper - lower) * 0.08, 0.5)
        lower -= margin
        upper += margin
        x = np.linspace(lower[0], upper[0], 65)
        y = np.linspace(lower[1], upper[1], 65)
        xx, yy = np.meshgrid(x, y)
        zz = Problem.objective(xx, yy)
        ax.plot_surface(xx, yy, zz, cmap="viridis", alpha=0.22,
                        linewidth=0, antialiased=True)

        # Line3D markers allow animation through the public set_data_3d API.
        particles, = ax.plot([], [], [], linestyle="None", marker="o",
                              markersize=4, color="tab:blue", alpha=0.8,
                              label="Particles")
        best_dot, = ax.plot([], [], [], linestyle="None", marker="*",
                             markersize=14, color="crimson", label="gBest")
        best_path, = ax.plot([], [], [], color="crimson", lw=1.5,
                             alpha=0.8, label="gBest path")
        ax.scatter([2], [3], [0], s=80, c="black", marker="x",
                   label="Optimum (2, 3, 0)")
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
               zlim=(float(zz.min()), max(1.0, float(zz.max()))),
               xlabel="x", ylabel="y", zlabel="Objective f(x, y)")
        ax.set_box_aspect((1, 1, 0.75))
        ax.view_init(elev=28, azim=-60)
        ax.legend(loc="upper left", fontsize=8)

        curve_ax.set(title="Convergence (maximization)",
                     xlabel="Iteration", ylabel="gBest objective value",
                     xlim=(0, max(1, len(scores) - 1)))
        score_margin = max(float(np.ptp(scores)) * 0.08, 1e-6)
        curve_ax.set_ylim(scores.min() - score_margin,
                          max(0.0, scores.max()) + score_margin)
        curve_ax.axhline(0, color="gray", ls="--", lw=1, label="Maximum = 0")
        curve, = curve_ax.plot([], [], color="tab:blue", lw=2, label="gBest score")
        curve_ax.grid(alpha=0.25)
        curve_ax.legend()

        def update(frame):
            points = positions[frame]
            particles.set_data_3d(points[:, 0], points[:, 1],
                                  Problem.objective(points[:, 0], points[:, 1]))
            best_dot.set_data_3d(bests[frame:frame + 1, 0],
                                 bests[frame:frame + 1, 1], scores[frame:frame + 1])
            best_path.set_data_3d(bests[:frame + 1, 0], bests[:frame + 1, 1],
                                  scores[:frame + 1])
            curve.set_data(steps[:frame + 1], scores[:frame + 1])
            ax.set_title(f"Iteration {frame} | gBest = {scores[frame]:.3e}")
            return particles, best_dot, best_path, curve

        update(0)
        animation = FuncAnimation(
            fig, update, frames=len(scores), init_func=lambda: update(0),
            interval=interval, repeat=False, blit=False,
        )
        return fig, animation


if __name__ == "__main__":
    np.random.seed(42)  # Change/remove this to try a different initial swarm.
    iterations = 100
    particle_num = 100

    swarm = Swarm(particle_num)
    swarm.run_pso(iterations)
    swarm.print_best()

    # Keep a reference to animation until the plot window is closed.
    fig, animation = swarm.animate(interval=500)
    plt.show()
