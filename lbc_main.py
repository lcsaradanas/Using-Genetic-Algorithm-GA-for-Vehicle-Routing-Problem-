import pandas as pd
import random

POP_SIZE = 100
N_GENERATIONS = 50
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
average_speed = 20

distance_matrix = {
    ('Pasay', 'QC'): 8, 
    ('QC', 'Makati'): 9, 
    ('Pasay', 'Makati'): 10,
    ('Mandaluyong', 'Manila'): 7, 
    ('Taguig', 'Makati'): 5, 
    ('Makati', 'Pasig'): 7,
    ('Pasig', 'Manila'): 9,
}

# Load LBC CSV
df = pd.read_csv('C:/Users/clari/Downloads/VRP/LBC Express/lbc_express.csv')
vehicles = []
for _, row in df.iterrows():
    vehicles.append({
        'id': row['ID'],
        'capacity': int(row['Max_Weight_kg']),
        'route': [r.strip() for r in row['Route'].split('->')],
        'parcel_num': int(row['No_of_Parcel']),
        'distance': int(row['Distance_km']),
        'travel_time': int(row['Travel_Time_min']),
        'over_capacity': int(row['Over_Capacity']),
        'late_delivery': int(row['Late_Delivery'])
    })

def fitness(individual, vehicles, distance_matrix, alpha=1, beta=1, gamma=10, delta=10):
    total_distance = 0
    total_time = 0 
    over_capacity = 0
    late_delivery = 0
    for v in vehicles:
        route = individual[v['id']]
        for i in range(len(route) - 1):
            total_distance += distance_matrix.get((route[i], route[i+1]), 0)
        over_capacity += v['over_capacity']
        late_delivery += v['late_delivery']
    # Estimate time based on total_distance
    total_time += total_distance / average_speed * 60
    # Genetic Algorithm Equation
    return alpha * total_distance + beta * total_time + gamma * over_capacity + delta * late_delivery


def initialize_population(vehicles):
    population = []
    for _ in range(POP_SIZE):
        individual = {}
        for v in vehicles:
            shuffled_route = v['route'][:]
            random.shuffle(shuffled_route)
            individual[v['id']] = shuffled_route
        population.append(individual)
    return population

def select_population(population, vehicles, distance_matrix):
    population.sort(key=lambda ind: fitness(ind, vehicles, distance_matrix))
    return population[:POP_SIZE // 2]

def crossover(ind1, ind2):
    child = {}
    for v_id in ind1:
        child[v_id] = ind1[v_id][:] if random.random() < CROSSOVER_RATE else ind2[v_id][:]
    return child

def mutate(individual):
    for v_id in individual:
        if random.random() < MUTATION_RATE:
            random.shuffle(individual[v_id])
    return individual

def genetic_algorithm(vehicles, distance_matrix):
    population = initialize_population(vehicles)
    for gen in range(N_GENERATIONS):
        selected = select_population(population, vehicles, distance_matrix)
        children = []
        for i in range(len(selected)//2):
            child = crossover(selected[2*i], selected[2*i+1])
            child = mutate(child)
            children.append(child)
        population = selected + children
    best_ind = min(population, key=lambda ind: fitness(ind, vehicles, distance_matrix))
    return best_ind, fitness(best_ind, vehicles, distance_matrix)

if __name__ == "__main__":
    best_routes, best_fitness = genetic_algorithm(vehicles, distance_matrix)
    print("Best Route Assignment:", best_routes)
    print("Fitness Score:", best_fitness)
