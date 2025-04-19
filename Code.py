import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict


def parse_vpr_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    nodes = []
    demands = []
    
    # Parse all nodes including depot
    for line in lines:
        parts = list(map(int, line.split()))
        x = parts[1]
        y = parts[2]
        demand = parts[3]
        nodes.append((x, y))
        demands.append(demand)
    
    # Separate depot (first node) and customers
    depot = nodes[0]
    customers = nodes[1:]  # All nodes after the first are customers
    customer_demands = demands[1:]  # Exclude depot demand
    
    # Create distance matrix (depot + customers)
    all_points = [depot] + customers
    n_points = len(all_points)
    distance_matrix = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(n_points):
            dx = all_points[i][0] - all_points[j][0]
            dy = all_points[i][1] - all_points[j][1]
            distance_matrix[i][j] = np.sqrt(dx**2 + dy**2)
    
    return customer_demands, distance_matrix

# Genetic Algorithm Components
def generate_individual(required_visits, allowable_days):
    individual = []
    for visits, days in zip(required_visits, allowable_days):
        # Choose distinct days from allowable days
        selected_days = np.random.choice(days, visits, replace=False)
        individual.append(sorted(selected_days.tolist()))
    return individual

def calculate_route_distance(route, distance_matrix):
    if not route:
        return 0.0
    total = distance_matrix[0][route[0]]
    for i in range(len(route)-1):
        total += distance_matrix[route[i]][route[i+1]]
    total += distance_matrix[route[-1]][0]
    return total

def evaluate_individual(individual, distance_matrix, n_days):
    day_customers = defaultdict(list)
    total_distance = 0
    max_daily = 0
    
    # Customer indices now match distance_matrix[1..n]
    for cust_idx, days in enumerate(individual):
        for day in days:
            day_customers[day].append(cust_idx + 1)  # +1 to skip depot
    
    for day in range(1, n_days+1):
        customers = day_customers.get(day, [])
        if not customers:
            continue
            
        # Nearest neighbor heuristic
        unvisited = set(customers)
        route = []
        current = 0  # depot
        
        while unvisited:
            next_node = min(unvisited, key=lambda x: distance_matrix[current][x])
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        
        day_distance = calculate_route_distance(route, distance_matrix)
        total_distance += day_distance
        max_daily = max(max_daily, day_distance)
    
    return total_distance, max_daily

def dominates(a, b):
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

def non_dominated_sort(population, evaluations):
    fronts = [[]]
    domination_counts = [0] * len(population)
    dominated_by = [[] for _ in range(len(population))]
    
    for i, a in enumerate(evaluations):
        for j, b in enumerate(evaluations):
            if i == j:
                continue
            if dominates(a, b):
                dominated_by[i].append(j)
            elif dominates(b, a):
                domination_counts[i] += 1
                
        if domination_counts[i] == 0:
            fronts[0].append(i)
    
    current_front = 0
    # Modified loop condition to prevent index errors
    while current_front < len(fronts) and fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_by[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        
        if next_front:
            fronts.append(next_front)
        current_front += 1
    
    return fronts

def crowding_distance(evaluations):
    n = len(evaluations)
    distances = [0.0] * n
    
    if n <= 2:
        return [float('inf')]*n if n > 0 else []
    
    for m in range(2):
        # Sort indices based on objective m
        sorted_indices = sorted(range(n), key=lambda i: evaluations[i][m])
        
        # Set boundary points to infinity
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        # Skip if all values are equal
        if evaluations[sorted_indices[-1]][m] == evaluations[sorted_indices[0]][m]:
            continue
            
        # Normalization factor
        norm = evaluations[sorted_indices[-1]][m] - evaluations[sorted_indices[0]][m]
        
        # Update middle points
        for i in range(1, n-1):
            distances[sorted_indices[i]] += (
                evaluations[sorted_indices[i+1]][m] - 
                evaluations[sorted_indices[i-1]][m]
            ) / norm
    
    return distances

# Genetic Operators
def crossover(parent1, parent2, n_days):
    child = []
    for p1, p2 in zip(parent1, parent2):
        if random.random() < 0.5:
            new_days = p1.copy()
        else:
            new_days = p2.copy()
        
        if len(set(new_days)) < len(new_days):
            unique = list(set(new_days))
            missing = len(new_days) - len(unique)
            available = list(set(range(1, n_days+1)) - set(unique))
            unique += np.random.choice(available, missing, replace=False).tolist()
            new_days = sorted(unique)
        
        child.append(new_days)
    return child

def mutate(individual, mutation_rate, n_days):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            days = individual[i].copy()
            idx = random.randint(0, len(days)-1)
            available = list(set(range(1, n_days+1)) - set(days))
            if available:
                days[idx] = random.choice(available)
                individual[i] = sorted(days)
    return individual

# NSGA-II Implementation
def nsga2(params):
    demands, distance_matrix = parse_vpr_file(params['filename'])
    n_days = params['n_days']
    required_visits = params['required_visits']
    allowable_days = params['allowable_days']
    
    population = [generate_individual(required_visits, allowable_days) 
                for _ in range(params['population_size'])]
    
    archive = []
    
    for gen in range(params['generations']):
        evaluations = [evaluate_individual(ind, distance_matrix, n_days) 
                      for ind in population]
        
        # Update archive
        combined = population + archive
        combined_evals = evaluations + [evaluate_individual(ind, distance_matrix, n_days) 
                                       for ind in archive]
        fronts = non_dominated_sort(combined, combined_evals)
        
        new_archive = []
        for front in fronts:
            front_inds = [combined[i] for i in front]
            front_evals = [combined_evals[i] for i in front]
            
            if len(new_archive) + len(front_inds) <= params['archive_size']:
                new_archive.extend(front_inds)
            else:
                distances = crowding_distance(front_evals)
                sorted_front = sorted(zip(front_inds, distances), 
                                    key=lambda x: -x[1])
                needed = params['archive_size'] - len(new_archive)
                new_archive.extend([x[0] for x in sorted_front[:needed]])
                break
        
        archive = new_archive
        
        # Selection and reproduction
        parents = random.choices(population, k=params['population_size'])
        offspring = []
        
        for _ in range(params['population_size'] // 2):
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2, n_days)
            child = mutate(child, params['mutation_rate'], n_days)
            offspring.append(child)
        
        population = parents + offspring
    
    return archive

# Visualization
def plot_pareto_front(archive, distance_matrix, n_days):
    evaluations = [evaluate_individual(ind, distance_matrix, n_days) for ind in archive]
    x = [e[0] for e in evaluations]
    y = [e[1] for e in evaluations]
    
    plt.scatter(x, y)
    plt.xlabel('Total Distance')
    plt.ylabel('Max Daily Distance')
    plt.title('Pareto Front')
    plt.show()

# Parameters and Execution
# Parameters and Execution
params = {
    'filename': 'vrp10.txt',
    'population_size': 100,
    'generations': 50,
    'mutation_rate': 0.1,
    'archive_size': 100,
    'n_days': 10,
    'required_visits': [1] * 99,  # 99 customers (nodes 2-100)
    'allowable_days': [list(range(1, 6)) for _ in range(99)]
}

archive = nsga2(params)
demands, distance_matrix = parse_vpr_file(params['filename'])  # Only 2 values now
n_days = params['n_days']  # Get from parameters

plot_pareto_front(archive, distance_matrix, n_days)