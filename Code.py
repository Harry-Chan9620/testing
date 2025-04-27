import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import pandas as pd
#For Consistent reproducible results
random.seed(5)
np.random.seed(5)
save_file = False
def vpr_file(filename): #reads data from file and extracts nodes [idx,x,y,demand] for distance matrix
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    nodes, demands = [], []
    for line in lines:
        parts = list(map(int, line.split()))
        x, y, demand = parts[1], parts[2], parts[3]
        nodes.append((x, y))
        demands.append(demand)
    
    #Distance matrix
    all_points = nodes
    n_points = len(all_points)
    distance_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            dx = all_points[i][0] - all_points[j][0]
            dy = all_points[i][1] - all_points[j][1]
            distance_matrix[i][j] = np.sqrt(dx**2 + dy**2)
    
    return demands[1:], distance_matrix  # Exclude depot demand, remove [1:] to exclude depot resulting in 100 customers
# get_pareto_front to include customer_demands and daily_capacity.
def get_pareto_front(population, distance_matrix, n_days, customer_demands, daily_capacity, allowable_days):
    evaluations = [evaluate_individual(ind, distance_matrix, n_days, customer_demands, daily_capacity, allowable_days) 
                  for ind in population]
    fronts = non_dominated_sort(population, evaluations)
    return [population[i] for i in fronts[0]] if fronts else []

def plot_pareto_front(population, distance_matrix, n_days, customer_demands, daily_capacity, allowable_days):
    evaluations = [evaluate_individual(ind, distance_matrix, n_days, customer_demands, daily_capacity, allowable_days)
                  for ind in population]
    pareto = get_pareto_front(population, distance_matrix, n_days, customer_demands, daily_capacity, allowable_days)
    pareto_evals = [evaluate_individual(ind, distance_matrix, n_days, customer_demands, daily_capacity, allowable_days)
                   for ind in pareto]
    plt.scatter([e[0] for e in evaluations], [e[1] for e in evaluations], alpha=0.3, label='All Solutions')
    plt.scatter([e[0] for e in pareto_evals], [e[1] for e in pareto_evals], c='red', label='Pareto Front')
    plt.xlabel('Total Distance')
    plt.ylabel('Max Single-Day Distance')
    plt.title('Pareto Front with Solution Filtering')
    plt.legend()
    plt.show()

# NSGA-II Components
def generate_individual(required_visits, allowable_days):
    individual = []
    for visits, days in zip(required_visits, allowable_days):
        selected_days = np.random.choice(days, visits, replace=False)
        individual.append(sorted(selected_days.tolist()))
    return individual

def calculate_route_distance(route, distance_matrix):
#calulates the distance for starting and ending at depot which index 0 is used.
    if not route:
        return 0.0
    total = distance_matrix[0][route[0]]
    for i in range(len(route)-1):
        total += distance_matrix[route[i]][route[i+1]]
    total += distance_matrix[route[-1]][0]
    return total

def evaluate_individual(individual, distance_matrix, n_days, customer_demands, daily_capacity, allowable_days):
    day_customers = defaultdict(list)
    day_demands = defaultdict(int)
    valid = True
    
    # First pass: check capacity constraints
    valid = True
    for cust_idx, days in enumerate(individual):
        # Check 1: Are all assigned days within allowable_days?
        if not set(days).issubset(set(allowable_days[cust_idx])):
            valid = False
            break  # Exit immediately if invalid

        # Check 2: Does the assignment violate daily capacity?
        for day in days:
            day_demands[day] += customer_demands[cust_idx]
            day_customers[day].append(cust_idx + 1)
            if day_demands[day] > daily_capacity:
                valid = False
                break  # Exit day loop if capacity exceeded
        if not valid:
            break  # Exit customer loop if any violation
    
    # Immediately reject invalid solutions
    if not valid:
        return (float('inf'), float('inf'))  #PENALTY if solution is not valid where demand is greater than daily capacity
    
    # Second pass: calculate distances
    total_distance = 0
    max_single_day_distance = 0
    for day, customers in day_customers.items():
        route = []
        current = 0
        unvisited = set(customers)
        while unvisited:
            next_node = min(unvisited, key=lambda x: distance_matrix[current][x])
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        day_distance = calculate_route_distance(route, distance_matrix)
        total_distance += day_distance
        max_single_day_distance = max(max_single_day_distance, day_distance)
    
    return total_distance, max_single_day_distance

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
        return [float('inf')] * n
    
    for m in range(2):
        sorted_indices = sorted(range(n), key=lambda i: evaluations[i][m])
        min_val = evaluations[sorted_indices[0]][m]
        max_val = evaluations[sorted_indices[-1]][m]
        if max_val == min_val:
            continue
        
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        for i in range(1, n-1):
            distances[sorted_indices[i]] += (
                evaluations[sorted_indices[i+1]][m] - evaluations[sorted_indices[i-1]][m]
            ) / (max_val - min_val)
    
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
            available = list(set(range(1, n_days+1)) - set(unique))
            np.random.shuffle(available)
            new_days = sorted(unique + available[:len(new_days)-len(unique)])
        child.append(new_days)
    return child

def mutate(individual, mutation_rate, n_days, allowable_days):
    for i in range(len(individual)):
        if not individual[i]:
            continue
        if random.random() < mutation_rate:
            days = individual[i].copy()
            idx = random.randint(0, len(days)-1)
            # Use allowable_days[i], not all days
            available = list(set(allowable_days[i]) - set(days))
            if available:
                days[idx] = random.choice(available)
                individual[i] = sorted(days)
    return individual

# NSGA-II Algorithm
def nsga2(params, customer_demands, distance_matrix):
    n_days = params['n_days']
    required_visits = params['required_visits']
    allowable_days = params['allowable_days']
    daily_capacity = params['daily_capacity']
    data_collection = []
    
    
    population = [generate_individual(required_visits, allowable_days) 
                  for _ in range(params['population_size'])]
    
    # Setup plots
    plt.figure(figsize=(10, 6))
    axis = plt.gca()
    plt.xlabel('Total Distance')
    plt.ylabel('Max Single-Day Distance')
    plt.title('Pareto Front Evolution')
    
    all_fronts = []
    all_evaluations = []  # Track all evaluations across generations

    def update(gen):
        axis.clear()
        axis.set_xlabel('Total Distance')
        axis.set_ylabel('Max Single-Day Distance')
        axis.set_title(f'Generation {gen+1}/{params["generations"]}')
        
        # Plot all evaluations (including dominated) with filtering for valid solutions
        if all_evaluations:
            # Filter out invalid solutions (inf, inf)
            valid_evaluations = [e for e in all_evaluations if e[0] != float('inf') and e[1] != float('inf')]
            axis.scatter(
                [e[0] for e in valid_evaluations], 
                [e[1] for e in valid_evaluations], 
                c='gray', alpha=0.3, label='All Solutions (Including Dominated)'
            )
        
        # Plot historical Pareto fronts with transparency
        for i, front in enumerate(all_fronts):
            axis.scatter(
                [e[0] for e in front], 
                [e[1] for e in front],
                c='blue', alpha=(i+1)/len(all_fronts)*0.5, 
                edgecolors='none', s=15, label='Historical Fronts' if i == 0 else ""
            )
        
        # Plot current Pareto front
        current_front = get_pareto_front(population, distance_matrix, n_days, customer_demands, daily_capacity, allowable_days)
        current_evals = [evaluate_individual(ind, distance_matrix, n_days, customer_demands, daily_capacity, allowable_days)
                        for ind in current_front]
        axis.scatter(
            [data[0] for data in current_evals], 
            [data[1] for data in current_evals],
            c='red', s=30, label='Current Pareto Front'
        )
        
        # Ensures labels are unique
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axis.legend(by_label.values(), by_label.keys())
        plt.draw()


    
    for gen in range(params['generations']):
        evaluations = [evaluate_individual(ind, distance_matrix, n_days, customer_demands, daily_capacity, allowable_days)
                      for ind in population]
                # Track current generation's data
        for ind, eval_result in zip(population, evaluations):
            total_dist, max_dist = eval_result
            valid = (total_dist != float('inf') and max_dist != float('inf'))
            data_collection.append({
                'generation': gen,
                'total_distance': total_dist,
                'max_daily_distance': max_dist,
                'valid': valid,
                'solution': ind  # Optional: store the actual solution
            })
        
        offspring = []
        for _ in range(params['population_size']):
            candidates = random.sample(range(len(population)), 2)
            a, b = candidates[0], candidates[1]
            winner = population[a] if (evaluations[a][0] + evaluations[a][1] < evaluations[b][0] + evaluations[b][1]) else population[b]
            offspring.append(winner)
        
        # Generate children
        children = []
        for i in range(0, len(offspring), 2):
            p1, p2 = offspring[i], offspring[i+1] if i+1 < len(offspring) else offspring[i]
            children.append(mutate(crossover(p1, p2, n_days), params['mutation_rate'], n_days, allowable_days))
        
        # Combine and select next population
        combined = population + children
        combined_evals = [evaluate_individual(ind, distance_matrix, n_days, customer_demands, daily_capacity, allowable_days)
                          for ind in combined]
        all_evaluations.extend(combined_evals)  
        
        fronts = non_dominated_sort(combined, combined_evals)
        next_population = []
        remaining = params['population_size']
        
        for front in fronts:
            front_inds = [combined[i] for i in front]
            front_evals = [combined_evals[i] for i in front]
            
            if len(next_population) + len(front_inds) <= remaining:
                next_population.extend(front_inds)
                remaining -= len(front_inds)
            else:
                distances = crowding_distance(front_evals)
                sorted_front = sorted(zip(front_inds, distances), key=lambda x: -x[1])
                next_population.extend([ind for ind, _ in sorted_front[:remaining]])
                break
        
        population = next_population[:params['population_size']]
        current_front = get_pareto_front(population, distance_matrix, n_days, customer_demands, daily_capacity, allowable_days)
        current_evals = [evaluate_individual(ind, distance_matrix, n_days, customer_demands, daily_capacity, allowable_days)
                for ind in current_front]

        all_fronts.append(current_evals)
        
        # Update plot every N generations
        if gen % 5 == 0:  # Plot every 5 generations
            update(gen)
            plt.pause(0.5)  # Pause to see updates
    
    # Final plot
    update(params['generations']-1)
    plt.show()
    if save_file == True:
        df = pd.DataFrame(data_collection)
        df.to_csv('moga_results.csv', index=False)
        
    return population        

def print_customer_demands(customer_demands):
    """Print customer IDs and their demands"""
    print("\nCustomer Demands:")
    for cust_id, demand in enumerate(customer_demands, start=1):  # Customers start at 1
        print(f"Customer {cust_id}: Demand = {demand}")
def print_schedule(individual, n_days):
    """Print customer assignments per day, including empty days"""
    day_assignments = defaultdict(list)
    for cust_idx, days in enumerate(individual):
        for day in days:
            day_assignments[day].append(cust_idx + 1)  # Customers start at 1
    
    print(f"\nSchedule (Total Days: {n_days}):")
    for day in range(1, n_days + 1):
        customers = sorted(day_assignments.get(day, []))
        print(f"Day {day}: Customers {customers}")
#--------------------Solution stats-------------------
def print_solution_metrics(individual, distance_matrix, customer_demands, daily_capacity, n_days, allowable_days):
    total_dist, max_dist = evaluate_individual(individual, distance_matrix, n_days, customer_demands, daily_capacity, allowable_days)
    valid = total_dist != float('inf')
    
    print(f"\n=== Solution {'VALID' if valid else 'INVALID'} ===")
    print(f"Total Distance: {total_dist:.2f}")
    print(f"Max Single-Day Distance: {max_dist:.2f}")
    
    if valid:
        day_assignments = defaultdict(list)
        for cust_idx, days in enumerate(individual):
            for day in days:
                day_assignments[day].append(cust_idx + 1)
        
        for day in sorted(day_assignments):
            demand = sum(customer_demands[cust-1] for cust in day_assignments[day])
            print(f"Day {day}: {len(day_assignments[day])} customers | Demand: {demand}/{daily_capacity}")
# Execution
params = {
    'filename': 'vrp10.txt',
    'population_size': 100,
    'generations': 50,
    'mutation_rate': 0.1,
    'n_days': 10,
    'daily_capacity': 200,
    'required_visits': [1] * 99,
    'allowable_days': [list(range(1, 11)) for _ in range(99)]
}

# Parse data once and reuse
customer_demands, distance_matrix = vpr_file(params['filename'])
print_customer_demands(customer_demands)

# Run NSGA-II with parameters
final_population = nsga2(params, customer_demands, distance_matrix)
#examples
print("\nExample Schedules:")
for idx in [0, len(final_population)//2, -1]:
    print(f"\n=== Solution {idx+1} ===")
    print_schedule(final_population[idx], params['n_days'])
    print_solution_metrics(final_population[idx], distance_matrix, customer_demands, params['daily_capacity'], params['n_days'], params['allowable_days'])
# Plot Pareto front with all parameters
plot_pareto_front(final_population, 
                 distance_matrix, 
                 params['n_days'],
                 customer_demands,
                 params['daily_capacity'],
                 params['allowable_days'])