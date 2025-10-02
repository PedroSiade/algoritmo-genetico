import random
import numpy as np
from itertools import permutations

def calculate_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def fitness(route, city_locations):
    total_distance = 0
    visited_cities = 1
    daily_distance = 0
    
    for i in range(len(route) - 1):
        distance = calculate_distance(city_locations[route[i]], city_locations[route[i+1]])
        if daily_distance + distance > 60:
            break
        daily_distance += distance
        visited_cities += 1
    
    return visited_cities

def generate_population(city_ids, population_size):
    return [random.sample(city_ids, len(city_ids)) for _ in range(population_size)]

def select_parents(population, city_locations):
    population_fitness = [(route, fitness(route, city_locations)) for route in population]
    population_fitness.sort(key=lambda x: x[1], reverse=True)
    return [route for route, _ in population_fitness[:len(population)//2]]

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    remaining = [city for city in parent2 if city not in child]
    child = [city if city is not None else remaining.pop(0) for city in child]
    return child

def mutate(route, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def genetic_algorithm(city_locations, generations=100, population_size=50, mutation_rate=0.1):
    city_ids = list(city_locations.keys())
    population = generate_population(city_ids, population_size)
    
    for _ in range(generations):
        selected_parents = select_parents(population, city_locations)
        new_population = selected_parents[:]
        
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
    
    best_route = max(population, key=lambda route: fitness(route, city_locations))
    return best_route
