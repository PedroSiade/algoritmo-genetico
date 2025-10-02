import random
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from plot_cities import load_cities, plot_paths

DAYS = 5
MAX_DIST_DAY = 40.0

def calculate_distance(city_1: list[float], city_2: list[float]):
    dx = city_1[0] - city_2[0]
    dy = city_1[1] - city_2[1]
    return np.sqrt(dx**2 + dy**2)

def group_cities(route: list, cities: np.ndarray) -> list[list]:
    """
    Esta função simula a visita às cidades seguindo a ordem da rota e agrupa-as por dias
    de viagem. A cada passo, verifica se adicionar uma nova cidade não ultrapassa o limite
    diário de distância, considerando também a distância de retorno à base.
    
    NOVA FUNCIONALIDADE: Tenta até 3 cidades diferentes antes de finalizar o dia.
    """
    base_position = [0.0, 0.0]  # Posição da base (0,0)
    total_visits = []
    current_day_index = 0
    daily_visits = [route[0]]

    i = 0
    while i < len(route) and current_day_index < DAYS:
        city_id_1 = route[i]
        position_1 = cities[int(city_id_1), 1:]
        
        # Calcula distância da base para a primeira cidade do dia
        first_city_pos = cities[int(daily_visits[0]), 1:]
        base_to_first = calculate_distance(base_position, first_city_pos)
        
        # Calcula distância entre cidades já visitadas no dia
        day_distance = 0
        for j in range(len(daily_visits) - 1):
            pos_j = cities[int(daily_visits[j]), 1:]
            pos_j1 = cities[int(daily_visits[j+1]), 1:]
            day_distance += calculate_distance(pos_j, pos_j1)
        
        # NOVA ESTRATÉGIA: Tenta até 3 cidades diferentes antes de finalizar o dia
        city_added = False
        max_attempts = min(3, len(route) - i - 1)
        
        for attempt in range(max_attempts):
            if i + 1 + attempt >= len(route):
                break
                
            city_id_2 = route[i + 1 + attempt]
            position_2 = cities[int(city_id_2), 1:]
            distance = calculate_distance(position_1, position_2)
            
            # Adiciona a nova distância e o retorno à base
            return_distance = calculate_distance(position_2, base_position)
            total_day_distance = base_to_first + day_distance + distance + return_distance
            
            # Verifica se adicionar esta cidade não ultrapassa o limite diário
            if total_day_distance <= MAX_DIST_DAY:
                daily_visits.append(city_id_2)
                i += 1 + attempt
                city_added = True
                break
        
        # Se nenhuma cidade pôde ser adicionada, finaliza o dia
        if not city_added:
            total_visits.append(daily_visits)
            current_day_index += 1
            daily_visits = [city_id_1]
            if current_day_index == DAYS: break
            i += 1
    
    if len(daily_visits) > 1 and current_day_index < DAYS:
        total_visits.append(daily_visits)
    
    return total_visits

def fitness_function(route: list, cities: np.ndarray) -> float:
    """
    Avalia a qualidade (fitness) de uma rota contando quantas cidades podem ser visitadas
    respeitando o limite de distância diária e incluindo retorno obrigatório à base.
    
    NOVA FUNCIONALIDADE: Tenta até 3 cidades diferentes antes de finalizar o dia.
    Também dá bônus para cidades novas para incentivar exploração de novas regiões.
    """
    base_position = [0.0, 0.0]  # Posição da base (0,0)
    visited_cities = np.zeros(DAYS)
    current_day_index = 0
    daily_visits = 1
    daily_route = [route[0]]
    
    # Rastreia cidades únicas visitadas para dar bônus
    unique_cities_visited = set([route[0]])
    new_city_bonus = 0

    i = 0
    while i < len(route) and current_day_index < DAYS:
        city_id_1 = route[i]
        position_1 = cities[int(city_id_1), 1:]
        
        # Calcula distância da base para a primeira cidade do dia
        first_city_pos = cities[int(daily_route[0]), 1:]
        base_to_first = calculate_distance(base_position, first_city_pos)
        
        # Calcula distância entre cidades já visitadas no dia
        day_distance = 0
        for j in range(len(daily_route) - 1):
            pos_j = cities[int(daily_route[j]), 1:]
            pos_j1 = cities[int(daily_route[j+1]), 1:]
            day_distance += calculate_distance(pos_j, pos_j1)
        
        # NOVA ESTRATÉGIA: Tenta até 3 cidades diferentes antes de finalizar o dia
        city_added = False
        max_attempts = min(3, len(route) - i - 1)
        
        for attempt in range(max_attempts):
            if i + 1 + attempt >= len(route):
                break
                
            city_id_2 = route[i + 1 + attempt]
            position_2 = cities[int(city_id_2), 1:]
            distance = calculate_distance(position_1, position_2)
            
            # Adiciona a nova distância e o retorno à base
            return_distance = calculate_distance(position_2, base_position)
            total_day_distance = base_to_first + day_distance + distance + return_distance
            
            # Verifica se é possível visitar esta cidade no mesmo dia
            if total_day_distance <= MAX_DIST_DAY:
                daily_visits += 1
                daily_route.append(city_id_2)
                
                # Verifica se é uma cidade nova e adiciona bônus
                if city_id_2 not in unique_cities_visited:
                    unique_cities_visited.add(city_id_2)
                    new_city_bonus += 1  # Bônus alterado de 0.5 para 1
                
                i += 1 + attempt
                city_added = True
                break
        
        # Se nenhuma cidade pôde ser adicionada, finaliza o dia
        if not city_added:
            visited_cities[current_day_index] = daily_visits
            current_day_index += 1
            daily_visits = 1
            daily_route = [city_id_1]
            if current_day_index == DAYS: break
            i += 1
    
    if daily_visits > 1 and current_day_index < DAYS:
        visited_cities[current_day_index] = daily_visits
    
    total_visited_cities = np.sum(visited_cities)
    cities_proportion = total_visited_cities / len(route)
    
    # Fitness inclui bônus para cidades novas
    base_fitness = total_visited_cities + (cities_proportion * 10)
    return base_fitness + new_city_bonus

def initial_population(cities: np.ndarray, population_size: int) -> np.ndarray:
    city_ids = list(cities[:,0])
    
    return [random.sample(city_ids, len(city_ids)) for _ in range(population_size)]

def select_parents(fitnessess: np.ndarray,
                   tournament_size: int = 3,
                   selected_size: int = 10) -> list[list]:
    selected = []
    
    for _ in range(selected_size):
        tournament = random.sample(fitnessess, tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    
    return selected

def crossover(parent_1: list, parent_2: list) -> list:
    size = len(parent_1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent_1[start:end]
    remaining = [city for city in parent_2 if city not in child]
    child = [city if city is not None else remaining.pop(0) for city in child]
    
    return child

def adaptive_mutation_rate(generation: int, total_generations: int, initial_rate: float = 0.3, final_rate: float = 0.3) -> float:
    """
    Calcula a taxa de mutação adaptativa que diminui ao longo das gerações
    
    Esta função implementa uma mutação adaptativa que começa com uma taxa alta
    para explorar o espaço de busca e gradualmente diminui para refinar soluções
    nas gerações finais.
    
    Parâmetros:
    - generation: geração atual (0 a total_generations-1)
    - total_generations: número total de gerações
    - initial_rate: taxa de mutação inicial (padrão: 0.3)
    - final_rate: taxa de mutação final (padrão: 0.01)
    
    Retorno:
    - taxa de mutação para a geração atual
    """
    # Progressão linear da taxa de mutação
    progress = generation / (total_generations - 1)  # 0.0 a 1.0
    current_rate = initial_rate - (initial_rate - final_rate) * progress
    
    # Garante que a taxa não seja negativa
    return max(current_rate, final_rate)

def mutation(route: list, mutation_rate: float) -> list:
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    
    return route



def optimize_route_2opt_constrained(route, cities):
    """
    Otimização 2-opt que respeita restrições de distância diária e retorno à base.
    Usa fitness function como critério de otimização em vez de distância total.
    """
    improved = True
    best_route = route.copy()
    best_fitness = fitness_function(best_route, cities)
  
    while improved:
        improved = False
      
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
        
                new_route = two_opt_swap(best_route, i, j)
                new_fitness = fitness_function(new_route, cities)
                
                # Só aceita se melhorar o fitness (que já considera todas as restrições)
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_route = new_route
                    improved = True
    
    return best_route

def two_opt_swap(route, i, j):
    new_route = route.copy()
    new_route[i:j+1] = reversed(route[i:j+1])
    
    return new_route

def calculate_total_distance(route, cities):
    total = 0
    
    for i in range(len(route) - 1):
        city1 = cities[int(route[i]), 1:]
        city2 = cities[int(route[i+1]), 1:]
        total += calculate_distance(city1, city2)
    
    return total

def simple_clustering(cities, num_clusters = DAYS):
    coords = cities[:, 1:]
    city_ids = cities[:, 0].astype(int)
    n_cities = len(cities)
  
    centroids_idx = np.random.choice(n_cities, num_clusters, replace=False)
    centroids = coords[centroids_idx]
    
    clusters = np.zeros(n_cities, dtype=int)
    
    max_iterations = 20
    for _ in range(max_iterations):
        for i in range(n_cities):
            distances = [calculate_distance(coords[i], centroid) for centroid in centroids]
            clusters[i] = np.argmin(distances)
        
        new_centroids = np.zeros((num_clusters, 2))
        for j in range(num_clusters):
            if np.sum(clusters == j) > 0:
                new_centroids[j] = np.mean(coords[clusters == j], axis=0)
            else:
                new_centroids[j] = centroids[j]
        
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids
    
    clustered_cities = []
  
    for cluster_id in range(num_clusters):
        cluster_city_ids = city_ids[clusters == cluster_id].tolist()
        clustered_cities.extend(cluster_city_ids)
    
    return clustered_cities

def genetic_algorithm(cities: np.ndarray,
                      population_size: int,
                      generations: int,
                      mutation_rate: float,
                      tournament_size: int,
                      elitism_rate: int):
    clustered_cities = simple_clustering(cities)
    routes = initial_population(cities, population_size - 1)
    fitnessess: list[tuple[np.ndarray, float]] = [(route, fitness_function(route, cities)) for route in routes]

    routes.append(clustered_cities)
    
    mean_fitnesses = []
    best_fitnesses = []

    best_route, best_fitness = sorted(fitnessess, key=lambda x: x[1], reverse=True)[0]
    mean_fitness = sum(fit[1] for fit in fitnessess)/len(fitnessess)

    best_fitnesses.append(best_fitness)
    mean_fitnesses.append(mean_fitness)
  
    for i in range(generations):
        # Calcula taxa de mutação adaptativa para esta geração
        current_mutation_rate = adaptive_mutation_rate(i, generations, mutation_rate, 0.01)
        
        selected_parents = select_parents(fitnessess, tournament_size, elitism_rate)
        new_population = selected_parents[:]
      
        while len(new_population) < population_size:
            parent_1, parent_2 = random.sample(selected_parents, 2)
            child = crossover(parent_1, parent_2)
            child = mutation(child, current_mutation_rate)
            new_population.append(child)
      
        # Otimização 2-opt com restrições aplicada a cada 10 gerações
        if i % 10 == 0 and best_route is not None:
            optimized_route = optimize_route_2opt_constrained(best_route, cities)
            new_population.append(optimized_route)
            if len(new_population) > population_size:
                idx_to_remove = random.randint(0, len(new_population) - 2)
                new_population.pop(idx_to_remove)
      
        new_fitnessess = [(route, fitness_function(route, cities)) for route in new_population]
        
        new_best_route, new_best_fitness = sorted(fitnessess, key=lambda x: x[1], reverse=True)[0]
        new_mean_fitness = sum(fit[1] for fit in fitnessess)/len(fitnessess)

        best_fitnesses.append(new_best_fitness)
        mean_fitnesses.append(new_mean_fitness)

        if new_best_fitness > best_fitness:
            best_fitness = new_best_fitness
            best_route = new_best_route
        
        routes = new_population
        fitnessess = new_fitnessess
      
    # Aplicação final da otimização 2-opt com restrições
    if best_route is not None:
        best_route = optimize_route_2opt_constrained(best_route, cities)
    
    return best_route, mean_fitnesses, best_fitnesses

def main():
    cities_df = load_cities('assets/cities.csv')
    cities = cities_df.to_numpy()
    
    print(f"Iniciando algoritmo genético para roteirização de {len(cities)} cidades em {DAYS} dias...")
    print(f"Distância máxima por dia: {MAX_DIST_DAY} unidades")
    print(f"Mutacao adaptativa: 0.3 -> 0.01 ao longo das gerações\n")

    best_route, mean_fitnesses, best_fitnesses = genetic_algorithm(cities=cities,
                                                                   population_size=300,
                                                                   generations=600,
                                                                   mutation_rate=0.3,
                                                                   tournament_size=3,
                                                                   elitism_rate=2)    
    
    print("\nAlgoritmo genético concluído. Organizando resultados...")
    
    best_route_grouped = group_cities(best_route, cities)
    flattened_best_route_grouped = [city for day in best_route_grouped for city in day]

    total_cities_raw = len(flattened_best_route_grouped)
    unique_cities_count = len(np.unique(flattened_best_route_grouped))

    print("\n=== ESTATÍSTICAS DAS GERAÇÕES ===")
    generation_table = PrettyTable()
    generation_table.field_names = ["Geração", "Melhor fitness", "Fitness médio"]
    generation_table.add_rows([[i+1, mean_fitnesses[i], best_fitnesses[i]] for i in range(len(mean_fitnesses))])
    print(generation_table)

    plt.plot(best_fitnesses)
    plt.plot(mean_fitnesses)
    plt.legend(['Melhor fitness', 'Fitness médio'])
    
    print("\n=== ESTATÍSTICAS DA SOLUÇÃO ===")
    print(f"Total de cidades visitadas (com repetições): {total_cities_raw}")
    print(f"(Essas repetições ocorrem, pois são contabilizadas as mesmas cidades visitadas em dias diferentes: cidade final e cidade inicial de um dia)")
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