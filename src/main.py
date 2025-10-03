"""
Algoritmo Genético para Roteirização de Cidades
Maximiza cidades visitadas em dias fixos respeitando limite de distância diária.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from plot_cities import load_cities, plot_paths

# Constantes globais
DAYS = 5  # Número de dias disponíveis para visitar cidades
MAX_DIST_DAY = 40.0  # Distância máxima que pode ser percorrida em um único dia

def calculate_distance(city1: list[float], city2: list[float]) -> float:
    """Calcula distância euclidiana entre duas cidades."""
    dx = city1[0] - city2[0]
    dy = city1[1] - city2[1]
    return np.sqrt(dx**2 + dy**2)

def create_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """Cria matriz de distâncias entre todas as cidades."""
    num_cities = len(cities)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i, num_cities):
            pos_i = cities[i, 1:]
            pos_j = cities[j, 1:]
            distance = calculate_distance(pos_i, pos_j)
            dist_matrix[i, j] = dist_matrix[j, i] = distance
    return dist_matrix

def group_cities(route: list, dist_matrix: np.ndarray) -> list[list]:
    """Agrupa rota em dias respeitando limite de distância diária."""
    if not route:
        return []

    route_iter = iter(route)
    current_city = next(route_iter)
    daily_routes = []
    day = 0

    while day < DAYS:
        daily_route = [current_city]
        daily_distance = dist_matrix[0, int(current_city)]  # Base → primeira cidade
        
        try:
            while True:
                next_city = next(route_iter)
                distance_to_next = dist_matrix[int(current_city), int(next_city)]
                return_distance = dist_matrix[0, int(next_city)]  # Próxima cidade → base
                
                # Decisão: cabe no limite diário? (percurso + retorno)
                if daily_distance + distance_to_next + return_distance <= MAX_DIST_DAY:
                    daily_distance += distance_to_next
                    daily_route.append(next_city)
                    current_city = next_city
                else:
                    # Dia completo - inicia novo dia com a cidade que não coube
                    daily_routes.append(daily_route)
                    day += 1
                    current_city = next_city
                    break
        except StopIteration:
            # Rota esgotada - adiciona último dia
            daily_routes.append(daily_route)
            break
    
    return daily_routes

def fitness_function(route: list, dist_matrix: np.ndarray, num_cities: int) -> float:
    """Calcula fitness da rota baseado em cidades únicas visitadas."""
    if not route:
        return 0

    route_iter = iter(route)
    current_city = next(route_iter)
    unique_cities = {current_city}  # Rastreia cidades únicas para evitar repetições
    total_distance = 0.0
    day = 0

    while day < DAYS:
        daily_visits = 1
        daily_distance = dist_matrix[0, int(current_city)]  # Distância da base até primeira cidade

        try:
            while True:
                next_city = next(route_iter)
                distance_to_next = dist_matrix[int(current_city), int(next_city)]
                return_distance = dist_matrix[0, int(next_city)]  # Distância de retorno à base

                # Decisão crítica: verifica se cabe no limite diário (ida + volta)
                if daily_distance + distance_to_next + return_distance <= MAX_DIST_DAY:
                    daily_distance += distance_to_next
                    daily_visits += 1
                    current_city = next_city
                    unique_cities.add(current_city)  # Adiciona cidade única
                else:
                    # Finaliza dia atual e inicia próximo
                    total_distance += daily_distance + dist_matrix[0, int(current_city)]
                    day += 1
                    current_city = next_city
                    break
        except StopIteration:
            # Rota esgotada - finaliza último dia
            total_distance += daily_distance + dist_matrix[0, int(current_city)]
            break
    
    # Sistema de pontuação: recompensa cidades únicas, penaliza distância
    city_score = len(unique_cities) * 100  # 100 pontos por cidade única
    max_distance = DAYS * MAX_DIST_DAY
    distance_penalty = (total_distance / max_distance) * 10  # Penalidade proporcional
    return city_score - distance_penalty

def nearest_neighbor_heuristic(dist_matrix: np.ndarray, start_city: int, num_cities: int) -> list:
    """Cria rota usando heurística do vizinho mais próximo."""
    unvisited = set(range(num_cities))
    current_city = start_city
    route = [current_city]
    unvisited.remove(current_city)

    while unvisited:
        nearest = min(unvisited, key=lambda city: dist_matrix[current_city, city])
        route.append(nearest)
        unvisited.remove(nearest)
        current_city = nearest
        
    return route

def initial_population(dist_matrix: np.ndarray, pop_size: int, heuristic_count: int = 10) -> list[list]:
    """Cria população inicial com heurística e indivíduos aleatórios."""
    num_cities = dist_matrix.shape[0]
    city_ids = list(range(num_cities))
    population = []

    # Indivíduos com heurística
    start_cities = random.sample(city_ids, min(heuristic_count, len(city_ids)))
    for start_city in start_cities:
        population.append(nearest_neighbor_heuristic(dist_matrix, start_city, num_cities))

    # Indivíduos aleatórios
    remaining = pop_size - len(population)
    for _ in range(remaining):
        population.append(random.sample(city_ids, len(city_ids)))
        
    return population

def select_parents(fitness_list: list, tournament_size: int = 3, selected_size: int = 10) -> list[list]:
    """Seleciona pais usando seleção por torneio."""
    selected = []
    
    for _ in range(selected_size):
        tournament = random.sample(fitness_list, tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    
    return selected

def crossover(parent1: list, parent2: list) -> list:
    """Cruzamento de ordem (OX) entre dois pais."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))  # Segmento aleatório do pai1
    
    child = [None] * size
    child[start:end] = parent1[start:end]  # Copia segmento do pai1
    
    # Preenche restante com cidades do pai2 (preserva ordem relativa)
    remaining = [city for city in parent2 if city not in child]
    child = [city if city is not None else remaining.pop(0) for city in child]
    
    return child

def adaptive_mutation_rate(gen: int, total_gens: int, initial_rate: float = 0.09, final_rate: float = 0.01) -> float:
    """Calcula taxa de mutação adaptativa que diminui ao longo das gerações."""
    if initial_rate <= final_rate:
        return initial_rate

    progress = gen / (total_gens - 1)  # Progresso: 0.0 (início) → 1.0 (fim)
    # Decaimento exponencial: alta exploração inicial → baixa exploração final
    current_rate = initial_rate * (final_rate / initial_rate) ** progress
    
    return current_rate

def mutation(route: list, mutation_rate: float) -> list:
    """Aplica mutação por inversão de segmento."""
    if random.random() < mutation_rate:
        start, end = sorted(random.sample(range(len(route)), 2))
        segment = route[start:end]
        route[start:end] = segment[::-1]
    
    return route

def genetic_algorithm(dist_matrix: np.ndarray, pop_size: int, generations: int, tournament_size: int,
                      elitism_rate: int, initial_mutation_rate: float, final_mutation_rate: float, heuristic_count: int):
    """Executa algoritmo genético completo."""
    num_cities = dist_matrix.shape[0]
    population = initial_population(dist_matrix, pop_size, heuristic_count)
    fitness_list = [(route, fitness_function(route, dist_matrix, num_cities)) for route in population]
    
    mean_fitnesses = []
    best_fitnesses = []

    best_route, best_fitness = sorted(fitness_list, key=lambda x: x[1], reverse=True)[0]
    mean_fitness = sum(fit[1] for fit in fitness_list) / len(fitness_list)

    best_fitnesses.append(best_fitness)
    mean_fitnesses.append(mean_fitness)

    for gen in range(generations):
        # Taxa de mutação adaptativa: diminui ao longo das gerações
        current_mutation_rate = adaptive_mutation_rate(gen, generations, initial_mutation_rate, final_mutation_rate)
        
        # Seleção por torneio + elitismo
        selected_parents = select_parents(fitness_list, tournament_size, elitism_rate)
        new_population = selected_parents[:]  # Elitismo: melhores passam direto

        # Geração de novos indivíduos via crossover + mutação
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child = crossover(parent1, parent2)
            child = mutation(child, current_mutation_rate)  # Mutação adaptativa
            new_population.append(child)
        
        # Avaliação da nova população
        new_fitness_list = [(route, fitness_function(route, dist_matrix, num_cities)) for route in new_population]
        
        new_best_route, new_best_fitness = sorted(new_fitness_list, key=lambda x: x[1], reverse=True)[0]
        new_mean_fitness = sum(fit[1] for fit in new_fitness_list) / len(new_fitness_list)

        best_fitnesses.append(new_best_fitness)
        mean_fitnesses.append(new_mean_fitness)

        # Atualização da melhor solução encontrada
        if new_best_fitness > best_fitness:
            best_fitness = new_best_fitness
            best_route = new_best_route
        
        population = new_population
        fitness_list = new_fitness_list

    return best_route, mean_fitnesses, best_fitnesses


def main():
    """Função principal: executa AG e exibe resultados."""
    cities_df = load_cities('../assets/cities.csv')
    cities = cities_df.to_numpy()
    dist_matrix = create_distance_matrix(cities)
    
    # Parâmetros do algoritmo
    POP_SIZE = 200
    GENERATIONS = 100
    TOURNAMENT_SIZE = 3
    ELITISM_RATE = 4
    HEURISTIC_PERCENTAGE = 0.15
    INITIAL_MUTATION_RATE = 0.6
    FINAL_MUTATION_RATE = 0.01
    HEURISTIC_COUNT = int(POP_SIZE * HEURISTIC_PERCENTAGE)
    
    print(f"Semeando população inicial com {HEURISTIC_COUNT} indivíduos da heurística 'Vizinho Mais Próximo'.")
    print(f"Iniciando AG para {len(cities)} cidades em {DAYS} dias...")
    print(f"Distância máxima por dia: {MAX_DIST_DAY} unidades")
    print(f"Mutação adaptativa: {INITIAL_MUTATION_RATE} -> {FINAL_MUTATION_RATE}\n")

    best_route, mean_fitnesses, best_fitnesses = genetic_algorithm(
        dist_matrix, POP_SIZE, GENERATIONS, TOURNAMENT_SIZE, ELITISM_RATE,
        INITIAL_MUTATION_RATE, FINAL_MUTATION_RATE, HEURISTIC_COUNT
    )
    
    print("Algoritmo genético concluído. Organizando resultados...")
    
    best_route_grouped = group_cities(best_route, dist_matrix)
    flattened_route = [city for day in best_route_grouped for city in day]
    unique_cities_count = len(np.unique(flattened_route))

    print("\n=== ESTATÍSTICAS DAS GERAÇÕES ===")
    gen_table = PrettyTable()
    gen_table.field_names = ["Geração", "Melhor fitness", "Fitness médio"]
    gen_table.add_rows([[i+1, mean_fitnesses[i], best_fitnesses[i]] for i in range(len(mean_fitnesses))])
    print(gen_table)

    plt.plot(best_fitnesses)
    plt.plot(mean_fitnesses)
    plt.legend(['Melhor fitness', 'Fitness médio'])
    
    print("\n=== ESTATÍSTICAS DA SOLUÇÃO ===")
    print(f"Total de cidades únicas visitadas: {unique_cities_count} de {len(cities)}")
    print(f"Porcentagem de cobertura: {unique_cities_count/len(cities)*100:.2f}%")
    
    print("\n=== DETALHAMENTO POR DIA ===")
    day_table = PrettyTable()
    day_table.field_names = ["Dia", "Quantidade total", "Quantidade única", "Cidades"]
    day_table.add_rows([[i+1, len(day), len(day)-1 if i > 0 else len(day), np.array(day)] for i, day in enumerate(best_route_grouped)])
    print(day_table)
    
    plot_paths(cities_df, best_route_grouped, return_to_base=True, base_coords=(0.0, 0.0))
    print("\nProcesso concluído.")

if __name__ == "__main__":
    main()