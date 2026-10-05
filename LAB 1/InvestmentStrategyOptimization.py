import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# ============================================================
# 1. PROBLEM DEFINITION – Portfolio Optimization
# ============================================================
# We have 4 assets. You can replace these with real data later.
ASSETS = ["Stock A", "Stock B", "Bond C", "Commodity D"]

# Expected annual returns
EXPECTED_RETURNS = np.array([0.12, 0.18, 0.06, 0.09])

# Covariance matrix (annualized)
COVARIANCE = np.array([
    [0.040, 0.018, 0.005, 0.010],
    [0.018, 0.090, 0.008, 0.015],
    [0.005, 0.008, 0.010, 0.004],
    [0.010, 0.015, 0.004, 0.025]
])

RISK_FREE_RATE = 0.03   # 3% risk-free rate


def portfolio_performance(weights: np.ndarray) -> Tuple[float, float, float]:
    """Return (expected_return, volatility, sharpe_ratio)"""
    ret = np.dot(weights, EXPECTED_RETURNS)
    vol = np.sqrt(np.dot(weights.T, np.dot(COVARIANCE, weights)))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0
    return ret, vol, sharpe


def fitness(chromosome: np.ndarray) -> float:
    """Fitness = Sharpe ratio (higher is better)"""
    _, _, sharpe = portfolio_performance(chromosome)
    return sharpe


# ============================================================
# 2. GENETIC ALGORITHM PARAMETERS
# ============================================================
POPULATION_SIZE = 100
GENOME_LENGTH = len(ASSETS)          # number of assets
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8
NUM_GENERATIONS = 150
TOURNAMENT_SIZE = 5
ELITISM = True                       # keep the best individual each generation


# ============================================================
# 3. INITIAL POPULATION
# ============================================================
def create_individual() -> np.ndarray:
    """Create a random portfolio (weights sum to 1)"""
    weights = np.random.random(GENOME_LENGTH)
    return weights / np.sum(weights)


def create_population(size: int) -> List[np.ndarray]:
    return [create_individual() for _ in range(size)]


# ============================================================
# 4 & 5. EVALUATION + SELECTION
# ============================================================
def tournament_selection(population: List[np.ndarray],
                         fitnesses: List[float]) -> np.ndarray:
    """Select one parent using tournament selection"""
    indices = np.random.choice(len(population), TOURNAMENT_SIZE, replace=False)
    best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
    return population[best_idx].copy()


# ============================================================
# 6. CROSSOVER
# ============================================================
def blend_crossover(parent1: np.ndarray, parent2: np.ndarray,
                    alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Blend crossover (BLX-α) – good for continuous weights"""
    if np.random.rand() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1

    # Ensure weights stay non-negative and sum to 1
    child1 = np.clip(child1, 0, None)
    child2 = np.clip(child2, 0, None)
    child1 /= np.sum(child1)
    child2 /= np.sum(child2)
    return child1, child2


# ============================================================
# 7. MUTATION
# ============================================================
def mutate(chromosome: np.ndarray) -> np.ndarray:
    """Gaussian mutation + re-normalization"""
    if np.random.rand() < MUTATION_RATE:
        mutation = np.random.normal(0, 0.05, size=GENOME_LENGTH)
        chromosome = chromosome + mutation
        chromosome = np.clip(chromosome, 0, None)   # no shorting
        if np.sum(chromosome) == 0:
            chromosome = create_individual()
        else:
            chromosome /= np.sum(chromosome)
    return chromosome


# ============================================================
# 8. MAIN EVOLUTION LOOP
# ============================================================
def run_genetic_algorithm():
    population = create_population(POPULATION_SIZE)

    best_fitness_history = []
    avg_fitness_history = []
    best_individual = None
    best_fitness = -np.inf

    print("Starting Genetic Algorithm for Portfolio Optimization...\n")

    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness
        fitnesses = [fitness(ind) for ind in population]

        # Track best
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]
        gen_best_ind = population[gen_best_idx]

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_individual = gen_best_ind.copy()

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(np.mean(fitnesses))

        # Progress report every 20 generations
        if (generation + 1) % 20 == 0 or generation == 0:
            ret, vol, sharpe = portfolio_performance(best_individual)
            print(f"Gen {generation+1:3d} | Best Sharpe: {best_fitness:.4f} | "
                  f"Return: {ret*100:.2f}% | Vol: {vol*100:.2f}%")

        # Create next generation
        new_population = []

        # Elitism
        if ELITISM:
            new_population.append(best_individual.copy())

        # Fill the rest of the population
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = blend_crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    # ========================================================
    # 9. OUTPUT THE BEST SOLUTION
    # ========================================================
    print("\n" + "="*60)
    print("OPTIMAL INVESTMENT STRATEGY FOUND")
    print("="*60)

    ret, vol, sharpe = portfolio_performance(best_individual)

    print(f"\nBest Sharpe Ratio : {sharpe:.4f}")
    print(f"Expected Return   : {ret*100:.2f}%")
    print(f"Volatility (Risk) : {vol*100:.2f}%")
    print("\nOptimal Asset Allocation:")
    for asset, weight in zip(ASSETS, best_individual):
        print(f"  {asset:12s} : {weight*100:6.2f}%")

    # Plot convergence
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness_history, label="Best Sharpe", linewidth=2)
    plt.plot(avg_fitness_history, label="Average Sharpe", alpha=0.7)
    plt.xlabel("Generation")
    plt.ylabel("Sharpe Ratio")
    plt.title("Genetic Algorithm Convergence – Portfolio Optimization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return best_individual, best_fitness


# ============================================================
# Run the algorithm
# ============================================================
if __name__ == "__main__":
    best_weights, best_sharpe = run_genetic_algorithm()