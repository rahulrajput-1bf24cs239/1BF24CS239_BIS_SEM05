import numpy as np
import random
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# 1. Dataset (you can replace with your own)
# ============================================================
X, y = make_classification(n_samples=500, n_features=6, n_informative=4,
                           n_redundant=0, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

FEATURE_NAMES = [f"F{i}" for i in range(X.shape[1])]

# ============================================================
# 2. GEP Parameters
# ============================================================
POP_SIZE = 80
GENES = 2                  # Multigenic chromosome
HEAD_LENGTH = 6
TAIL_LENGTH = HEAD_LENGTH + 1
GENE_LENGTH = HEAD_LENGTH + TAIL_LENGTH
CHROMOSOME_LENGTH = GENES * GENE_LENGTH

MAX_GENERATIONS = 60
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 4

# Function set and Terminal set for rule discovery
FUNCTIONS = ['AND', 'OR', '>', '<', '>=', '<=']
TERMINALS = FEATURE_NAMES + ['0.0', '0.5', '1.0', '-0.5']  # features + constants

# ============================================================
# 3. Chromosome creation & Gene Expression
# ============================================================
def create_gene():
    gene = []
    for i in range(GENE_LENGTH):
        if i < HEAD_LENGTH:
            gene.append(random.choice(FUNCTIONS + TERMINALS))
        else:
            gene.append(random.choice(TERMINALS))
    return gene

def create_chromosome():
    return [create_gene() for _ in range(GENES)]

def express_gene(gene, sample):
    """Very simplified expression of one gene into a boolean condition"""
    try:
        # Simple sequential evaluation (educational version)
        stack = []
        for symbol in gene:
            if symbol in FEATURE_NAMES:
                idx = FEATURE_NAMES.index(symbol)
                stack.append(sample[idx])
            elif symbol in ['0.0', '0.5', '1.0', '-0.5']:
                stack.append(float(symbol))
            elif symbol in ['>', '<', '>=', '<='] and len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                if symbol == '>':
                    stack.append(a > b)
                elif symbol == '<':
                    stack.append(a < b)
                elif symbol == '>=':
                    stack.append(a >= b)
                elif symbol == '<=':
                    stack.append(a <= b)
            elif symbol in ['AND', 'OR'] and len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                if symbol == 'AND':
                    stack.append(bool(a) and bool(b))
                else:
                    stack.append(bool(a) or bool(b))
        return bool(stack[-1]) if stack else False
    except:
        return False

def express_chromosome(chrom, sample):
    """Combine multiple genes with OR (simple linking)"""
    results = [express_gene(gene, sample) for gene in chrom]
    return any(results)   # Linking function = OR

# ============================================================
# 4. Fitness Evaluation
# ============================================================
def fitness(chrom, X, y):
    predictions = [1 if express_chromosome(chrom, x) else 0 for x in X]
    return accuracy_score(y, predictions)

# ============================================================
# 5. Genetic Operators
# ============================================================
def tournament_selection(population, fitnesses):
    candidates = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
    return max(candidates, key=lambda x: x[1])[0]

def mutate(chrom):
    new_chrom = [gene[:] for gene in chrom]
    for g in range(GENES):
        for i in range(GENE_LENGTH):
            if random.random() < MUTATION_RATE:
                if i < HEAD_LENGTH:
                    new_chrom[g][i] = random.choice(FUNCTIONS + TERMINALS)
                else:
                    new_chrom[g][i] = random.choice(TERMINALS)
    return new_chrom

def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return [gene[:] for gene in parent1], [gene[:] for gene in parent2]
    
    # One-point crossover at gene level
    point = random.randint(1, GENES - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# ============================================================
# 6. Main GEP Loop
# ============================================================
def run_gep():
    population = [create_chromosome() for _ in range(POP_SIZE)]
    best_chrom = None
    best_fit = -1
    history = []

    print("Starting GEP for Knowledge Discovery...\n")

    for gen in range(MAX_GENERATIONS):
        fitnesses = [fitness(chrom, X_train, y_train) for chrom in population]
        
        # Track best
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fit:
            best_fit = fitnesses[gen_best_idx]
            best_chrom = [gene[:] for gene in population[gen_best_idx]]
        
        history.append(best_fit)
        
        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"Generation {gen+1:3d} | Best Fitness (Accuracy): {best_fit:.4f}")

        # Create next generation
        new_population = [best_chrom]  # Elitism

        while len(new_population) < POP_SIZE:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_population.append(c1)
            if len(new_population) < POP_SIZE:
                new_population.append(c2)

        population = new_population

    # Final evaluation
    train_acc = fitness(best_chrom, X_train, y_train)
    test_acc = fitness(best_chrom, X_test, y_test)

    print("\n" + "="*50)
    print("Best Discovered Rules/Patterns")
    print("="*50)
    for i, gene in enumerate(best_chrom):
        print(f"Gene {i+1}: {' '.join(gene)}")
    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")

    return best_chrom, history

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    best_solution, fitness_history = run_gep()