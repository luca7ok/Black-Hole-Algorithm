from sklearn.cluster import KMeans
import numpy as np
import random
from typing import Callable


def kmeans_algorithm(
    dimension,
    max_iterations,
    star_num,
    lower_bound,
    upper_bound,
    evaluate,
    data=None,
    K=None,
):
    km = KMeans(n_clusters=K, n_init=1, max_iter=max_iterations, init="random")
    km.fit(np.array(data))
    centroids = km.cluster_centers_.flatten().tolist()
    best = evaluate(centroids)
    return centroids, best, [best] * max_iterations


def initialize_star(data, k) -> list[float]:
    star = []
    for _ in range(k):
        star.extend(data[random.randrange(len(data))])
    return star

def clip(x, lower_bound, upper_bound, j):
    low = lower_bound[j] if isinstance(lower_bound, list) else lower_bound
    high = upper_bound[j] if isinstance(upper_bound, list) else upper_bound
    return max(low, min(high, x))


def pso_algorithm(
    dimension: int,
    max_iterations: int,
    star_num: int,
    lower_bound,
    upper_bound,
    evaluate: Callable,
    data=None,
    K=None,
) -> tuple[list[float], float, list[float]]:
    w = 0.72
    c1 = 1.49
    c2 = 1.49

    v_max = [
        0.5
        * (
            (upper_bound[j] if isinstance(upper_bound, list) else upper_bound)
            - (lower_bound[j] if isinstance(lower_bound, list) else lower_bound)
        )
        for j in range(dimension)
    ]

    if data is not None and K is not None:
        particles = [initialize_star(data, K) for _ in range(star_num)]
    else:
        particles = [
            initialize_star(dimension, lower_bound, upper_bound)
            for _ in range(star_num)
        ]

    velocities = [
        [random.uniform(-v_max[j], v_max[j]) for j in range(dimension)]
        for _ in range(star_num)
    ]

    fitness = [evaluate(p) for p in particles]

    pbest = [p[:] for p in particles]
    pbest_fitness = fitness[:]

    gbest_index = min(range(star_num), key=lambda i: fitness[i])
    gbest = particles[gbest_index][:]
    gbest_fitness = fitness[gbest_index]

    convergence = []

    for _ in range(max_iterations):
        for i in range(star_num):
            for j in range(dimension):
                r1 = random.random()
                r2 = random.random()
                velocities[i][j] = (
                    w * velocities[i][j]
                    + c1 * r1 * (pbest[i][j] - particles[i][j])
                    + c2 * r2 * (gbest[j] - particles[i][j])
                )
                if velocities[i][j] > v_max[j]:
                    velocities[i][j] = v_max[j]
                elif velocities[i][j] < -v_max[j]:
                    velocities[i][j] = -v_max[j]

                particles[i][j] = clip(
                    particles[i][j] + velocities[i][j], lower_bound, upper_bound, j
                )

            fitness[i] = evaluate(particles[i])

            if fitness[i] < pbest_fitness[i]:
                pbest[i] = particles[i][:]
                pbest_fitness[i] = fitness[i]

                if fitness[i] < gbest_fitness:
                    gbest = particles[i][:]
                    gbest_fitness = fitness[i]

        convergence.append(gbest_fitness)

    return gbest, gbest_fitness, convergence


def bb_bc_algorithm(
    dimension: int,
    max_iterations: int,
    star_num: int,       
    lower_bound,
    upper_bound,
    evaluate: Callable,
    data: list[list[float]],
    K: int,
) -> tuple[list[float], float, list[float]]:
    alpha = 1.0  

    candidates = [initialize_star(data, K) for _ in range(star_num)]
    fitness = [evaluate(c) for c in candidates]

    best_index = min(range(star_num), key=lambda i: fitness[i])
    best = candidates[best_index][:]
    best_value = fitness[best_index]

    convergence = []

    for k in range(1, max_iterations + 1):

        weights = [1.0 / f if f > 0 else 1e10 for f in fitness]
        total_weight = sum(weights)
        center = [
            sum(candidates[i][j] * weights[i] for i in range(star_num)) / total_weight
            for j in range(dimension)
        ]

        for i in range(star_num):
            new_candidate = []
            for j in range(dimension):
                lo = lower_bound[j] if isinstance(lower_bound, list) else lower_bound
                hi = upper_bound[j] if isinstance(upper_bound, list) else upper_bound
                r = random.gauss(0, 1)
                new_val = center[j] + alpha * r * (hi - lo) / k
                new_candidate.append(clip(new_val, lower_bound, upper_bound, j))

            new_fitness = evaluate(new_candidate)
            candidates[i] = new_candidate
            fitness[i] = new_fitness

            if new_fitness < best_value:
                best = new_candidate[:]
                best_value = new_fitness

        worst_index = max(range(star_num), key=lambda i: fitness[i])
        candidates[worst_index] = best[:]
        fitness[worst_index] = best_value

        convergence.append(best_value)

    return best, best_value, convergence