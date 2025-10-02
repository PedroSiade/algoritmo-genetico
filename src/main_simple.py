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
    
    Retorno:
    - distância euclidiana entre as cidades
    """
    distanceX = city_1[0] - city_2[0]  
    distanceY = city_1[1] - city_2[1] 
    return np.sqrt(distanceX**2 + distanceY**2)  

def group_cities(route: list, cities: np.ndarray) -> list[list]:
    """
    Esta função simula a visita às cidades seguindo a ordem da rota e agrupa-as por dias
    de viagem. A cada passo, verifica se adicionar uma nova cidade não ultrapassa o limite
    diário de distância, considerando também a distância de retorno à base.
    
    NOVA FUNCIONALIDADE: Tenta até 3 cidades diferentes antes de finalizar o dia.
    """
    base_position = [0.0, 0.0]  # Posição da base (0,0)
    total_visits = []  # Lista final com agrupamentos de cidades por dia
    current_day_index = 0  # Dia atual
    daily_visits = [route[0]]  # Inicializa com a primeira cidade no primeiro dia

    i = 0  # Índice da cidade atual na rota
    # Percorre a rota enquanto houver cidades e dias disponíveis
    while i < len(route) and current_day_index < DAYS:
        city_id_1 = route[i]  # ID da cidade atual
        position_1 = cities[int(city_id_1), 1:]  # Coordenadas (x,y) da cidade atual
        
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
        max_attempts = min(3, len(route) - i - 1)  # Máximo 3 tentativas ou até o fim da rota
        
        for attempt in range(max_attempts):
            if i + 1 + attempt >= len(route):
                break
                
            city_id_2 = route[i + 1 + attempt]  # ID da cidade candidata
            position_2 = cities[int(city_id_2), 1:]  # Coordenadas da cidade candidata
            distance = calculate_distance(position_1, position_2)  # Distância até a cidade candidata
            
            # Adiciona a nova distância e o retorno à base
            return_distance = calculate_distance(position_2, base_position)
            total_day_distance = base_to_first + day_distance + distance + return_distance
            
            # Verifica se adicionar esta cidade não ultrapassa o limite diário
            if total_day_distance <= MAX_DIST_DAY:
                daily_visits.append(city_id_2)  # Adiciona cidade ao dia atual
                i += 1 + attempt  # Avança para a próxima cidade (pula as tentativas)
                city_added = True
                break
        
        # Se nenhuma cidade pôde ser adicionada, finaliza o dia
        if not city_added:
            total_visits.append(daily_visits)  # Adiciona as visitas do dia à lista total
            current_day_index += 1  # Avança para o próximo dia
            daily_visits = [city_id_1]  # Inicializa o próximo dia com a cidade atual
            if current_day_index == DAYS: break  # Encerra se acabaram os dias
            i += 1  # Avança para a próxima cidade na rota
    
    # Adiciona o último grupo de visitas se ainda houver dias disponíveis
    if len(daily_visits) > 1 and current_day_index < DAYS:
        total_visits.append(daily_visits)
    
    return total_visits

def fitness_function(route: list, cities: np.ndarray) -> float:
    """
    Avalia a qualidade (fitness) de uma rota contando quantas cidades podem ser visitadas
    respeitando o limite de distância diária e incluindo retorno obrigatório à base.
    
    Parâmetros:
    - route: lista com IDs das cidades na ordem de visita
    - cities: matriz com dados das cidades
    
    Retorno:
    - valor numérico que representa a qualidade da rota
    """
    base_position = [0.0, 0.0]  # Posição da base (0,0)
    visited_cities = np.zeros(DAYS)  # Array para armazenar o número de cidades visitadas em cada dia
    current_day_index = 0  # Índice do dia atual
    daily_visits = 1  # Contador de visitas (começa com 1 para a primeira cidade)
    daily_route = [route[0]]  # Lista das cidades visitadas no dia atual
    
    # NOVO: Rastreia cidades únicas visitadas para dar bônus
    unique_cities_visited = set([route[0]])  # Conjunto de cidades únicas visitadas
    new_city_bonus = 0  # Bônus acumulado por cidades novas

    i = 0  # Índice para percorrer a rota
    # Simula a viagem através das cidades na rota
    while i < len(route) and current_day_index < DAYS:
        city_id_1 = route[i]  # ID da cidade atual
        position_1 = cities[int(city_id_1), 1:]  # Coordenadas da cidade atual
        
        # Calcula distância da base para a primeira cidade do dia
        first_city_pos = cities[int(daily_route[0]), 1:]
        base_to_first = calculate_distance(base_position, first_city_pos)
        
        # Calcula distância entre cidades já visitadas no dia
        day_distance = 0
        for j in range(len(daily_route) - 1):
            pos_j = cities[int(daily_route[j]), 1:]
            pos_j1 = cities[int(daily_route[j+1]), 1:]
            day_distance += calculate_distance(pos_j, pos_j1)
        
        city_added = False
        max_attempts = min(3, len(route) - i - 1)  # Máximo 3 tentativas ou até o fim da rota
        
        for attempt in range(max_attempts):
            if i + 1 + attempt >= len(route):
                break
                
            city_id_2 = route[i + 1 + attempt]  # ID da cidade candidata
            position_2 = cities[int(city_id_2), 1:]  # Coordenadas da cidade candidata
            distance = calculate_distance(position_1, position_2)  # Distância até a cidade candidata
            
            # Adiciona a nova distância e o retorno à base
            return_distance = calculate_distance(position_2, base_position)
            total_day_distance = base_to_first + day_distance + distance + return_distance
            
            # Verifica se é possível visitar esta cidade no mesmo dia
            if total_day_distance <= MAX_DIST_DAY:
                daily_visits += 1  # Incrementa o contador de visitas do dia
                daily_route.append(city_id_2)  # Adiciona cidade à rota do dia
                
                # NOVO: Verifica se é uma cidade nova e adiciona bônus
                if city_id_2 not in unique_cities_visited:
                    unique_cities_visited.add(city_id_2)
                    new_city_bonus += 1  # Bônus leve para cidades novas
                
                i += 1 + attempt  # Avança para a próxima cidade (pula as tentativas)
                city_added = True
                break
        
        # Se nenhuma cidade pôde ser adicionada, finaliza o dia
        if not city_added:
            visited_cities[current_day_index] = daily_visits  # Registra número de visitas do dia
            current_day_index += 1  # Avança para o próximo dia
            daily_visits = 1  # Reinicia contador (começa com 1 para a primeira cidade do novo dia)
            daily_route = [city_id_1]  # Reinicia rota do dia com a cidade atual
            if current_day_index == DAYS: break  # Encerra se atingiu o número máximo de dias
            i += 1  # Avança para a próxima cidade na rota
    
    # Adiciona a pontuação do último dia (que pode estar incompleto)
    if daily_visits > 1 and current_day_index < DAYS:
        visited_cities[current_day_index] = daily_visits
    
    # Calcula o fitness total: soma das cidades visitadas + bônus por proporção + bônus por cidades novas
    total_visited_cities = np.sum(visited_cities)  # Total de cidades visitadas
    cities_proportion = total_visited_cities / len(route)  # Proporção de cidades visitadas
    
    # NOVO: Fitness inclui bônus para cidades novas
    base_fitness = total_visited_cities + (cities_proportion * 10)  # Fitness base
    return base_fitness + new_city_bonus  # Adiciona bônus por cidades novas

def initial_population(cities: np.ndarray, population_size: int) -> np.ndarray:
    """
    Cria a população inicial de rotas aleatórias
    
    Esta função gera diversas permutações aleatórias dos IDs das cidades, garantindo
    que cada indivíduo seja uma rota válida e diferente.
    
    Parâmetros:
    - cities: matriz com dados das cidades
    - population_size: tamanho da população a ser gerada
    
    Retorno:
    - lista de rotas aleatórias (população inicial)
    """
    city_ids = list(cities[:, 0])  # Extrai os IDs das cidades da matriz
    
    # Gera 'population_size' rotas diferentes embaralhando aleatoriamente os IDs das cidades
    return [random.sample(city_ids, len(city_ids)) for _ in range(population_size)]

def select_parents(fitnessess: np.ndarray,
                   tournament_size: int = 3,
                   selected_size: int = 10) -> list[list]:
    """
    Seleciona pais para reprodução usando seleção por torneio
    
    Seleciona-se indivíduos aleatoriamente e
    escolhe o melhor entre eles. Este processo é repetido várias vezes para
    formar o conjunto de pais que serão usados para criar a próxima geração.
    
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
    Realiza o cruzamento entre dois pais para gerar um filho
    usando o método de cruzamento de ordem (OX)
    
    O crossover de ordem é especialmente adequado para problemas de permutação como o TSP,
    pois garante que cada cidade apareça exatamente uma vez na rota resultante.
    O algoritmo:
    1. Seleciona um segmento aleatório do primeiro pai
    2. Copia esse segmento para o filho na mesma posição
    3. Preenche as posições restantes com cidades do segundo pai, na ordem que aparecem
       nele, pulando as que já foram incluídas do primeiro pai
    
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
    
    # Passo 1: Copia o segmento do primeiro pai para o filho
    child[start:end] = parent_1[start:end]
    
    # Passo 2: Identifica as cidades que não foram copiadas do primeiro pai
    remaining = [city for city in parent_2 if city not in child]  # Cidades a serem extraídas do segundo pai
    
    # Passo 3: Preenche as posições vazias com as cidades restantes, na ordem do segundo pai
    child = [city if city is not None else remaining.pop(0) for city in child]
    
    return child

def adaptive_mutation_rate(generation: int, total_generations: int, initial_rate: float = 0.3, final_rate: float = 0.01) -> float:
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
    """
    Aplica mutação à rota trocando duas cidades de posição
    
    A mutação é importante para manter a diversidade genética e permitir que o algoritmo
    explore novas regiões do espaço de busca. Neste caso, a mutação é implementada como
    uma simples troca de posição entre duas cidades aleatórias.
    
    Parâmetros:
    - route: rota a ser potencialmente mutada
    - mutation_rate: probabilidade de ocorrer mutação (0.0 a 1.0)
    
    Retorno:
    - rota possivelmente mutada
    """
    # Determina aleatoriamente se ocorrerá mutação baseado na taxa fornecida
    if random.random() < mutation_rate:
        # Seleciona aleatoriamente duas posições diferentes na rota
        idx1, idx2 = random.sample(range(len(route)), 2)
        
        # Troca as cidades nas posições selecionadas
        route[idx1], route[idx2] = route[idx2], route[idx1]
    
    return route

def genetic_algorithm(cities: np.ndarray,
                      population_size: int,
                      generations: int,
                      mutation_rate: float,
                      tournament_size: int,
                      elitism_rate: int):
    """
    Implementa o algoritmo genético para encontrar a melhor rota
    
    1. Cria uma população inicial
    2. Avalia o fitness de cada indivíduo
    3. Seleciona os melhores indivíduos para reprodução
    4. Gera novos indivíduos através de cruzamento e mutação
    5. Repete o processo por várias gerações
    
    Parâmetros:
    - cities: matriz com dados das cidades
    - population_size: tamanho da população
    - generations: número de gerações a executar
    - mutation_rate: taxa de mutação
    - tournament_size: número de rotas a serem selecionadas para o torneio
    - elitism_rate: número de pais a serem selecionados
    
    Retorno:
    - melhor rota encontrada, lista de fitness média e lista de melhores fitness
    """
    # Fase 1: Inicialização - Cria a população inicial
    population = initial_population(cities, population_size)  # Gera a população inicial
    fitnessess: list[tuple[np.ndarray, float]] = [(route, fitness_function(route, cities)) for route in population]  # Avalia a fitness de cada rota
    
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
        current_mutation_rate = adaptive_mutation_rate(generation, generations, mutation_rate, 0.1)
        
        selected_parents = select_parents(fitnessess, tournament_size, elitism_rate)  # Seleciona pais para reprodução
        new_population = selected_parents[:]  # Inicia nova população com os pais selecionados

        # Completa a população com novos indivíduos gerados por cruzamento e mutação
        while len(new_population) < population_size:
            parent_1, parent_2 = random.sample(selected_parents, 2)  # Seleciona aleatoriamente dois pais diferentes
            child = crossover(parent_1, parent_2)  # Gera um filho por cruzamento
            child = mutation(child, current_mutation_rate)  # Aplica mutação adaptativa ao filho
            new_population.append(child)  # Adiciona o filho à nova população
        
        # Avalia a nova população
        new_fitnessess = [(route, fitness_function(route, cities)) for route in new_population]
        
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
    Função principal: carrega dados, executa o algoritmo genético e plota os resultados.
    
    1. Carrega os dados das cidades
    2. Executa o algoritmo genético para encontrar a melhor rota
    3. Organiza a rota em dias de viagem
    4. Exibe estatísticas sobre a solução
    5. Visualiza graficamente a rota encontrada
    """
    # Carrega dados das cidades a partir do arquivo CSV
    cities_df = load_cities('assets/cities.csv')
    cities = cities_df.to_numpy()  # Converte para matriz numpy
    
    print(f"Iniciando algoritmo genético para roteirização de {len(cities)} cidades em {DAYS} dias...")
    print(f"Distância máxima por dia: {MAX_DIST_DAY} unidades")
    print(f"Mutacao adaptativa: 0.3 -> 0.01 ao longo das gerações\n")

    # Executa o algoritmo genético com parâmetros definidos
    best_route, mean_fitnesses, best_fitnesses = genetic_algorithm(cities=cities,
                                                                   population_size=300,  # 300 indivíduos na população
                                                                   generations=600,      # 200 gerações
                                                                   mutation_rate=0.3,    # 30% de chance de mutação
                                                                   tournament_size=3,    # Tamanho do torneio
                                                                   elitism_rate=2)      # Número de pais selecionados
    
    print("\nAlgoritmo genético concluído. Organizando resultados...")
    
    # Agrupa as cidades da melhor rota por dias de viagem
    best_route_grouped = group_cities(best_route, cities)
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