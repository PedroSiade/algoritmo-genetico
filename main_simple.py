"""
Algoritmo Genético para Roteirização de Cidades

Este programa implementa um algoritmo genético para resolver o problema de roteirização de cidades,
onde o objetivo é maximizar o número de cidades visitadas em um número fixo de dias,
respeitando a distância máxima que pode ser percorrida por dia.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from plot_cities import load_cities, plot_paths

# Constantes globais
DAYS = 5  # Número de dias disponíveis para visitar cidades
MAX_DIST_DAY = 40.0  # Distância máxima que pode ser percorrida em um único dia

def calculate_distance(city_1: list[float], city_2: list[float]) -> float:
    """
    Calcula a distância euclidiana entre duas cidades

    Parâmetros:
    - city_1: coordenadas (x,y) da primeira cidade
    - city_2: coordenadas (x,y) da segunda cidade
    
    Retorno:
    - distância euclidiana entre as cidades
    """
    dx = city_1[0] - city_2[0]  # Diferença entre coordenadas x
    dy = city_1[1] - city_2[1]  # Diferença entre coordenadas y
    return np.sqrt(dx**2 + dy**2)  # Aplicação do teorema de Pitágoras

def create_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """
    Cria e retorna uma matriz de distâncias entre todas as cidades.
    A matriz é pré-calculada para otimizar os cálculos de distância.
    """
    num_cities = len(cities)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i, num_cities):
            # city_id é o índice na matriz
            pos_i = cities[i, 1:]
            pos_j = cities[j, 1:]
            distance = calculate_distance(pos_i, pos_j)
            dist_matrix[i, j] = dist_matrix[j, i] = distance
    return dist_matrix

def group_cities(route: list, dist_matrix: np.ndarray) -> list[list]:
    """
    Agrupa a rota em dias, respeitando a distância máxima diária com retorno à base.
    Parâmetros:
    - route: lista com IDs das cidades na ordem de visita
    - dist_matrix: matriz pré-calculada com as distâncias entre as cidades.
    
    Retorno:
    - lista de listas, onde cada lista interna representa as cidades visitadas em um dia
    """
    base_position = [0.0, 0.0]  # Posição da base (0,0)
    total_visits = []  # Lista final com agrupamentos de cidades por dia
    current_day_index = 0  # Dia atual
    
    if not route:
        return []

    route_iterator = iter(route)
    current_city = next(route_iterator)

    while current_day_index < DAYS:
        daily_route = [current_city]
        # Distância do dia começa com a ida da base até a primeira cidade
        daily_distance = dist_matrix[0, int(current_city)]
        
        try:
            while True:
                next_city = next(route_iterator)
                distance_to_next = dist_matrix[int(current_city), int(next_city)]
                return_to_base_dist = dist_matrix[0, int(next_city)]
                
                # Verifica se a viagem (dia atual + próximo trecho + retorno à base) cabe no limite
                if daily_distance + distance_to_next + return_to_base_dist <= MAX_DIST_DAY:
                    daily_distance += distance_to_next
                    daily_route.append(next_city)
                    current_city = next_city
                else:
                    # Dia cheio, finaliza o dia atual e começa o próximo com a 'next_city'
                    total_visits.append(daily_route)
                    current_day_index += 1
                    current_city = next_city
                    break # Sai do loop do dia
        except StopIteration:
            # Acabaram as cidades na rota
            total_visits.append(daily_route)
            break # Sai do loop de dias
    
    return total_visits

def fitness_function(route: list, dist_matrix: np.ndarray, num_total_cities: int) -> float:
    """
    Calcula o fitness de uma rota.
    Rotas que visitam mais cidades dentro dos limites diários recebem maior pontuação.
    Um bônus é dado para incentivar a visita a cidades únicas.
    Parâmetros:
    - route: lista com IDs das cidades na ordem de visita
    - dist_matrix: matriz pré-calculada com as distâncias entre as cidades.
    - num_total_cities: número total de cidades no problema.
    
    Retorno:
    - valor numérico que representa a qualidade da rota
    """
    base_position = [0.0, 0.0]  # Posição da base (0,0)
    visited_cities = np.zeros(DAYS)  # Array para armazenar o número de cidades visitadas em cada dia
    current_day_index = 0  # Índice do dia atual
    total_distance_traveled = 0.0
    unique_cities_visited = set()

    route_iterator = iter(route)
    
    try:
        current_city = next(route_iterator)
    except StopIteration:
        return 0 # Rota vazia tem fitness 0

    while current_day_index < DAYS:
        daily_visits = 1
        unique_cities_visited.add(current_city)
        daily_distance = dist_matrix[0, int(current_city)]

        try:
            while True:
                next_city = next(route_iterator)
                distance_to_next = dist_matrix[int(current_city), int(next_city)]
                # Distância de retorno da *próxima* cidade para a base
                return_dist = dist_matrix[0, int(next_city)]

                if daily_distance + distance_to_next + return_dist <= MAX_DIST_DAY:
                    daily_distance += distance_to_next
                    daily_visits += 1
                    current_city = next_city
                    unique_cities_visited.add(current_city)
                else:
                    visited_cities[current_day_index] = daily_visits
                    # Adiciona a distância do dia, incluindo o retorno à base da *última* cidade do dia
                    total_distance_traveled += daily_distance + dist_matrix[0, int(current_city)]
                    current_day_index += 1
                    current_city = next_city
                    break
        except StopIteration:
            visited_cities[current_day_index] = daily_visits
            total_distance_traveled += daily_distance + dist_matrix[0, int(current_city)]
            break
    
    # Fitness: Recompensa fortemente por cidades únicas e penaliza levemente pela distância total.
    score_from_cities = len(unique_cities_visited) * 100

    # Penalidade normalizada: penaliza a distância de forma proporcional ao máximo possível
    max_possible_dist = DAYS * MAX_DIST_DAY
    distance_penalty = (total_distance_traveled / max_possible_dist) * 10 # Fator de penalidade
    fitness = score_from_cities - distance_penalty
    return fitness

def nearest_neighbor_heuristic(dist_matrix: np.ndarray, start_city_id: int, num_cities: int) -> list:
    """
    Cria uma rota usando a heurística do vizinho mais próximo.
    A partir de uma cidade, visita-se repetidamente a cidade mais próxima ainda não visitada.

    Parâmetros:
    - dist_matrix: matriz pré-calculada com as distâncias entre as cidades.
    - start_city_id: ID da cidade inicial.
    - num_cities: número total de cidades.
    
    Retorno:
    - Uma rota (lista de IDs de cidades) gerada pela heurística.
    """
    city_ids = set(range(num_cities))
    
    current_city_id = start_city_id
    route = [current_city_id]
    unvisited_cities = city_ids - {current_city_id}

    while unvisited_cities:
        # Encontra a cidade não visitada mais próxima
        nearest_city_id = min(
            unvisited_cities, key=lambda city_id: dist_matrix[int(current_city_id), int(city_id)]
        )
        # Adiciona a cidade mais próxima à rota e a remove das não visitadas
        route.append(nearest_city_id)
        unvisited_cities.remove(nearest_city_id)
        current_city_id = nearest_city_id
        
    return route

def initial_population(dist_matrix: np.ndarray, population_size: int, heuristic_count: int = 10) -> list[list]:
    """
    Cria a população inicial.
    Uma parte é gerada com a heurística do "vizinho mais próximo" para acelerar a convergência.
    O restante é aleatório para garantir diversidade.
    
    Parâmetros:
    - dist_matrix: matriz pré-calculada com as distâncias entre as cidades.
    - population_size: tamanho da população a ser gerada
    - heuristic_count: número de indivíduos a serem gerados com a heurística
    
    Retorno:
    - lista de rotas (população inicial)
    """
    num_cities = dist_matrix.shape[0]
    city_ids = list(range(num_cities))
    population = []

    # Gera indivíduos com a heurística do vizinho mais próximo
    start_cities = random.sample(city_ids, min(heuristic_count, len(city_ids)))
    for start_city in start_cities:
        population.append(nearest_neighbor_heuristic(dist_matrix, start_city, num_cities))

    # Preenche o restante da população com indivíduos aleatórios
    remaining_size = population_size - len(population)
    for _ in range(remaining_size):
        population.append(random.sample(city_ids, len(city_ids)))
        
    return population

def select_parents(fitnessess: np.ndarray,
                   tournament_size: int = 3,
                   selected_size: int = 10) -> list[list]:
    """
    Seleciona os pais para a próxima geração usando seleção por torneio.
    Indivíduos aleatórios competem, e o de maior fitness é escolhido.
    
    Parâmetros:
    - fitnessess: lista de tuplas com rotas e suas respectivas fitness
    - tournament_size: número de rotas a serem selecionadas para o torneio
    - selected_size: número de pais a serem selecionados
    
    Retorno:
    - lista com as rotas selecionadas para reprodução
    """
    selected = []  # Lista para armazenar os pais selecionados
    
    # Seleciona 'selected_size' pais através de torneios
    for _ in range(selected_size):
        tournament = random.sample(fitnessess, tournament_size)  # Seleciona rotas aleatórias para o torneio
        winner = max(tournament, key=lambda x: x[1])[0]  # Escolhe a rota com maior fitness como vencedora
        selected.append(winner)  # Adiciona o vencedor à lista de selecionados
    
    return selected

def crossover(parent_1: list, parent_2: list) -> list:
    """
    Realiza o cruzamento de ordem (OX) entre dois pais para gerar um filho.
    Este método preserva a ordem relativa das cidades, sendo ideal para problemas de rota.
    1. Um segmento do pai 1 é copiado para o filho.
    2. O restante do filho é preenchido com cidades do pai 2, na ordem em que aparecem.
    
    Parâmetros:
    - parent_1, parent_2: rotas dos pais
    
    Retorno:
    - nova rota (filho)
    """
    size = len(parent_1)  # Tamanho da rota (número de cidades)
    
    # Seleciona dois pontos aleatórios para definir o segmento a ser copiado do primeiro pai
    start, end = sorted(random.sample(range(size), 2))
    
    # Inicializa o filho como uma lista de valores nulos
    child = [None] * size
    
    # Copia o segmento do primeiro pai para o filho
    child[start:end] = parent_1[start:end]
    
    # Pega as cidades do segundo pai que ainda não estão no filho
    remaining = [city for city in parent_2 if city not in child]
    
    # Preenche as posições vazias do filho com as cidades restantes
    child = [city if city is not None else remaining.pop(0) for city in child]
    
    return child

def adaptive_mutation_rate(generation: int, total_generations: int, initial_rate: float = 0.09, final_rate: float = 0.01) -> float:
    """
    Calcula a taxa de mutação adaptativa que diminui ao longo das gerações
    Começa com uma taxa alta para exploração e diminui para refinar as melhores soluções.

    Parâmetros:
    - generation: geração atual (0 a total_generations-1)
    - total_generations: número total de gerações
    - initial_rate: taxa de mutação inicial (padrão: 0.3)
    - final_rate: taxa de mutação final (padrão: 0.01)
    
    Retorno:
    - taxa de mutação para a geração atual
    """
    if initial_rate <= final_rate:
        return initial_rate

    # Calcula o progresso da evolução (de 0.0 a 1.0)
    progress = generation / (total_generations - 1)

    # Aplica um decaimento exponencial, que é mais eficaz para GAs
    # Mantém a taxa alta no início (exploração) e a reduz rapidamente no final (refinamento)
    current_rate = initial_rate * (final_rate / initial_rate) ** progress
    
    return current_rate

def mutation(route: list, mutation_rate: float) -> list:
    """
    Aplica mutação a uma rota invertendo um segmento (Inversion Mutation).
    Este método é geralmente mais eficaz para o TSP do que a simples troca.

    Parâmetros:
    - route: rota a ser potencialmente mutada
    - mutation_rate: probabilidade de ocorrer mutação (0.0 a 1.0)
    
    Retorno:
    - rota (possivelmente) mutada
    """
    # Determina aleatoriamente se ocorrerá mutação baseado na taxa fornecida
    if random.random() < mutation_rate:
        # Seleciona dois pontos de corte para definir o segmento a ser invertido
        start, end = sorted(random.sample(range(len(route)), 2))
        
        # Inverte o segmento da rota entre os pontos de corte
        segment = route[start:end]
        route[start:end] = segment[::-1]
    
    return route

def genetic_algorithm(dist_matrix: np.ndarray,
                      population_size: int,
                      generations: int,                      
                      tournament_size: int,
                      elitism_rate: int,
                      initial_mutation_rate: float,
                      final_mutation_rate: float,
                      heuristic_count: int):
    """
    Executa o algoritmo genético completo.
    A cada geração, seleciona os melhores indivíduos, aplica crossover e mutação para criar uma nova população.
    
    Parâmetros:
    - dist_matrix: matriz pré-calculada com as distâncias entre as cidades.
    - population_size: tamanho da população
    - generations: número de gerações a executar
    - tournament_size: número de rotas a serem selecionadas para o torneio
    - elitism_rate: número de pais a serem selecionados
    - initial_mutation_rate: taxa de mutação inicial
    - final_mutation_rate: taxa de mutação final
    - heuristic_count: número de indivíduos a serem gerados com a heurística
    
    Retorno:
    - melhor rota encontrada, lista de fitness média e lista de melhores fitness
    """
    # Fase 1: Inicialização - Cria a população inicial
    num_cities = dist_matrix.shape[0]
    population = initial_population(dist_matrix, population_size, heuristic_count)
    fitnessess: list[tuple[np.ndarray, float]] = [(route, fitness_function(route, dist_matrix, num_cities)) for route in population]
    
    mean_fitnesses = []  # Lista para armazenar a média de fitness ao longo das gerações
    best_fitnesses = []  # Lista para armazenar o melhor fitness ao longo das gerações

    # Inicializa as melhores rotas e fitness
    best_route, best_fitness = sorted(fitnessess, key=lambda x: x[1], reverse=True)[0]
    mean_fitness = sum(fit[1] for fit in fitnessess) / len(fitnessess)

    best_fitnesses.append(best_fitness)  # Adiciona o melhor fitness à lista
    mean_fitnesses.append(mean_fitness)  # Adiciona a média de fitness à lista

    # Fase 2: Evolução - Executa o algoritmo pelo número especificado de gerações
    for generation in range(generations):
        # Calcula taxa de mutação adaptativa para esta geração
        current_mutation_rate = adaptive_mutation_rate(generation, generations, initial_mutation_rate, final_mutation_rate)
        
        selected_parents = select_parents(fitnessess, tournament_size, elitism_rate)  # Seleciona pais para reprodução
        new_population = selected_parents[:]  # Inicia nova população com os pais selecionados

        # Completa a população com novos indivíduos gerados por cruzamento e mutação
        while len(new_population) < population_size:
            parent_1, parent_2 = random.sample(selected_parents, 2)  # Seleciona aleatoriamente dois pais diferentes
            child = crossover(parent_1, parent_2)  # Gera um filho por cruzamento
            child = mutation(child, current_mutation_rate)  # Aplica mutação adaptativa ao filho
            new_population.append(child)  # Adiciona o filho à nova população
        
        # Avalia a nova população
        new_fitnessess = [(route, fitness_function(route, dist_matrix, num_cities)) for route in new_population]
        
        # Atualiza as melhores rotas e fitness
        new_best_route, new_best_fitness = sorted(new_fitnessess, key=lambda x: x[1], reverse=True)[0]
        new_mean_fitness = sum(fit[1] for fit in new_fitnessess) / len(new_fitnessess)

        best_fitnesses.append(new_best_fitness)  # Adiciona o novo melhor fitness à lista
        mean_fitnesses.append(new_mean_fitness)  # Adiciona a nova média de fitness à lista

        # Atualiza a melhor rota se necessário
        if new_best_fitness > best_fitness:
            best_fitness = new_best_fitness
            best_route = new_best_route
        
        population = new_population  # Atualiza a população para a próxima geração
        fitnessess = new_fitnessess  # Atualiza a lista de fitness

    return best_route, mean_fitnesses, best_fitnesses  # Retorna a melhor rota e as listas de fitness


def main():
    """
    Função principal: carrega os dados, executa o AG e exibe os resultados.
    """
    # Carrega dados das cidades a partir do arquivo CSV
    cities_df = load_cities('assets/cities.csv')
    cities = cities_df.to_numpy()

    # Otimização: Pré-calcula a matriz de distâncias
    dist_matrix = create_distance_matrix(cities)
    
    # --- PARÂMETROS DO ALGORITMO GENÉTICO ---
    POPULATION_SIZE = 200       # Indivíduos na população
    GENERATIONS = 200           # Número de gerações
    TOURNAMENT_SIZE = 3         # Tamanho do torneio para seleção
    ELITISM_RATE = 4            # Indivíduos de elite que passam diretamente para a próxima geração
    HEURISTIC_PERCENTAGE = 0.15  # Porcentagem da população inicial gerada com a heurística
    INITIAL_MUTATION_RATE = 0.6 # Taxa de mutação inicial (mais alta para exploração)
    FINAL_MUTATION_RATE = 0.05  # Taxa de mutação final
    HEURISTIC_COUNT = int(POPULATION_SIZE * HEURISTIC_PERCENTAGE) # Número de indivíduos gerados com a heurística
    # -----------------------------------------
    
    print(f"Semeando a população inicial com {HEURISTIC_COUNT} indivíduos da heurística 'Vizinho Mais Próximo'.")
    print(f"Iniciando algoritmo genético para roteirização de {len(cities)} cidades em {DAYS} dias...")
    print(f"Distância máxima por dia: {MAX_DIST_DAY} unidades")
    print(f"Mutação adaptativa: {INITIAL_MUTATION_RATE} -> {FINAL_MUTATION_RATE} ao longo das gerações\n")

    # Executa o algoritmo genético com parâmetros definidos
    best_route, mean_fitnesses, best_fitnesses = genetic_algorithm(dist_matrix=dist_matrix,
                                                                   population_size=POPULATION_SIZE,
                                                                   generations=GENERATIONS,
                                                                   tournament_size=TOURNAMENT_SIZE,
                                                                   elitism_rate=ELITISM_RATE,
                                                                   initial_mutation_rate=INITIAL_MUTATION_RATE,
                                                                   final_mutation_rate=FINAL_MUTATION_RATE,
                                                                   heuristic_count=HEURISTIC_COUNT)
    
    print("\nAlgoritmo genético concluído. Organizando resultados...")
    
    # Agrupa as cidades da melhor rota por dias de viagem
    best_route_grouped = group_cities(best_route, dist_matrix)
    flattened_best_route_grouped = [city for day in best_route_grouped for city in day]  # Achata a lista de cidades visitadas

    total_cities_raw = len(flattened_best_route_grouped)  # Total de cidades visitadas (com repetições)
    unique_cities_count = len(np.unique(flattened_best_route_grouped))  # Total de cidades únicas visitadas

    print("\n=== ESTATÍSTICAS DAS GERAÇÕES ===")
    generation_table = PrettyTable()  # Cria uma tabela para exibir estatísticas das gerações
    generation_table.field_names = ["Geração", "Melhor fitness", "Fitness médio"]
    generation_table.add_rows([[i+1, mean_fitnesses[i], best_fitnesses[i]] for i in range(len(mean_fitnesses))])  # Adiciona dados à tabela
    print(generation_table)

    # Plota as estatísticas de fitness
    plt.plot(best_fitnesses)
    plt.plot(mean_fitnesses)
    plt.legend(['Melhor fitness', 'Fitness médio'])
    
    print("\n=== ESTATÍSTICAS DA SOLUÇÃO ===")
    print(f"Total de cidades visitadas (com repetições): {total_cities_raw}")
    print(f"(Essas repetições ocorrem, pois são contabilizadas as mesmas cidades visitadas em dias diferentes: cidade final e cidade inicial de um dia)")
    print(f"Total de cidades únicas visitadas: {unique_cities_count} de {len(cities)}")
    print(f"Porcentagem de cobertura: {unique_cities_count/len(cities)*100:.2f}%")
    
    print("\n=== DETALHAMENTO POR DIA ===")
    day_table = PrettyTable()  # Cria uma tabela para exibir detalhes por dia
    day_table.field_names = ["Dia", "Quantidade total", "Quantidade única", "Cidades"]
    day_table.add_rows([[i+1, len(day), len(day)-1 if i > 0 else len(day), np.array(day)] for i, day in enumerate(best_route_grouped)])  # Adiciona dados à tabela
    print(day_table)
    
    
    # Visualiza a rota encontrada
    plot_paths(cities_df, best_route_grouped, return_to_base=True, base_coords=(0.0, 0.0))
    
    print("\nProcesso concluído.")

# Ponto de entrada do programa
if __name__ == "__main__":
    main()