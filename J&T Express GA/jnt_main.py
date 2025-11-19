import pandas as pd
import random

POP_SIZE = 100
N_GENERATIONS = 50
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
ALPHA = 1
GAMMA = 10

distance_matrix = {
    ('QC', 'Pasig'): 6, ('Pasig', 'QC'): 6,
    ('Pasig', 'Mandaluyong'): 5, ('Mandaluyong', 'Pasig'): 5,
    ('Mandaluyong', 'QC'): 7, ('QC', 'Mandaluyong'): 7,
    ('Makati', 'Taguig'): 4, ('Taguig', 'Makati'): 4,
    ('Manila', 'San Juan'): 6, ('San Juan', 'Manila'): 6,
    ('San Juan', 'QC'): 5, ('QC', 'San Juan'): 5,
    ('Manila', 'QC'): 8, ('QC', 'Manila'): 8,
    ('Manila', 'Pasig'): 7, ('Pasig', 'Manila'): 7,
    ('Pasig', 'Makati'): 10, ('Makati', 'Pasig'): 10,
}

csv_file = 'C:/Users/clari/Downloads/VRP/J&T Express/jnt_express.csv'
df = pd.read_csv(csv_file)
vehicles = []
for _, row in df.iterrows():
    vehicles.append({
        'id': row['ID'],
        'capacity': int(row['Capacity']),
        'route': [r.strip() for r in row['Route'].split('->')],
        'parcel_num': int(row['No_of_Parcel'])
    })

def fitness(individual, vehicles, distance_matrix, alpha=ALPHA, gamma=GAMMA):
    total_distance = 0
    over_capacity = 0
    for v in vehicles:
        route = individual[v['id']]
        for i in range(len(route) - 1):
            total_distance += distance_matrix.get((route[i], route[i+1]), 0)
        if v['parcel_num'] > v['capacity']:
            over_capacity += (v['parcel_num'] - v['capacity'])
    return alpha * total_distance + gamma * over_capacity

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

def per_vehicle_scores(best_routes, vehicles, distance_matrix, alpha=ALPHA, gamma=GAMMA, export_csv=False):
    print("\nPer-vehicle fitness breakdown:")
    results = []
    for v in vehicles:
        route = best_routes[v['id']]
        distance = 0
        for i in range(len(route) - 1):
            distance += distance_matrix.get((route[i], route[i+1]), 0)
        over_cap = max(0, v['parcel_num'] - v['capacity'])
        indiv_score = alpha * distance + gamma * over_cap
        results.append({
            'ID': v['id'],
            'Route': ' -> '.join(route),
            'Distance': distance,
            'OverCapacity': over_cap,
            'Fitness': indiv_score
        })
        print(f"{v['id']} | Route: {route} | Distance: {distance} km | OverCapacity: {over_cap} | Fitness: {indiv_score}")
    if export_csv:
        pd.DataFrame(results).to_csv('per_vehicle_fitness.csv', index=False)
        print("\nCSV file 'per_vehicle_fitness.csv' exported!")
    return results

if __name__ == "__main__":
    best_routes, best_fitness = genetic_algorithm(vehicles, distance_matrix)
    print("\nBest Global Assignment:", best_routes)
    print("Overall Fitness Score:", best_fitness)
    per_vehicle_scores(best_routes, vehicles, distance_matrix, alpha=ALPHA, gamma=GAMMA, export_csv=True)