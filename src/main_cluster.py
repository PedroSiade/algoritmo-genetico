import random
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from plot_cities import load_cities, plot_paths

totalDias = 5
distMax = 40.0

def calcularDistancia(cidadeUm, cidadeDois):
    distanciaX = cidadeUm[0] - cidadeDois[0]  
    distanciaY = cidadeUm[1] - cidadeDois[1] 

    return np.sqrt(distanciaX**2 + distanciaY**2)

def group_cities(route: list, cities: np.ndarray) -> list[list]:
    coordBase = [0.0, 0.0]  # Posição da base (0,0)
    totalVisitas = []
    idxDiaAtual = 0
    visitasDia = [route[0]]

    i = 0
    while i < len(route) and idxDiaAtual < totalDias:
        cidadeUm = route[i]
        coordUm = cities[int(cidadeUm), 1:]
        
        # Calcula distância da base para a primeira cidade do dia
        cidadeFirst = cities[int(visitasDia[0]), 1:]
        distFirstBase = calcularDistancia(coordBase, cidadeFirst)
        
        # Calcula distância entre cidades já visitadas no dia
        distanciaDia = 0
        for idx in range(len(visitasDia) - 1):
            posicaoAtual = cities[int(visitasDia[idx]), 1:]
            posicaoProxima = cities[int(visitasDia[idx + 1]), 1:]
            distanciaDia += calcularDistancia(posicaoAtual, posicaoProxima)
        
        # Tenta até 3 cidades diferentes antes de finalizar o dia
        cidadeAdd = False
        tentativas = min(3, len(route) - i - 1)
        
        for tentativa in range(tentativas):
            if i + 1 + tentativa >= len(route):
                break
                
            cidadeDois = route[i + 1 + tentativa]
            coordDois = cities[int(cidadeDois), 1:]
            distancia = calcularDistancia(coordUm, coordDois)
            
            # Adiciona a nova distância e o retorno à base
            distanciaBase = calcularDistancia(coordDois, coordBase)
            distanciaDiaTotal = distFirstBase + distanciaDia + distancia + distanciaBase
            
            # Verifica se adicionar esta cidade não ultrapassa o limite diário
            if distanciaDiaTotal <= distMax:
                visitasDia.append(cidadeDois)
                i += 1 + tentativa
                cidadeAdd = True
                break
        
        # Se nenhuma cidade pôde ser adicionada, finaliza o dia
        if not cidadeAdd:
            totalVisitas.append(visitasDia)
            idxDiaAtual += 1
            visitasDia = [cidadeUm]
            if idxDiaAtual == totalDias: break
            i += 1
    
    if len(visitasDia) > 1 and idxDiaAtual < totalDias:
        totalVisitas.append(visitasDia)
    
    return totalVisitas

def fitnessFunction(rota, tabelaCidades):
    coordBase = [0.0, 0.0]  # Posição da Base (0,0)

    qtdCidadesVisitadas = np.zeros(totalDias)  # Número de cidades visitadas em cada dia
    idxDiaAtual = 0      # Índice do dia atual
    visitasDia = 1       # Contador de visitas (começa com 1 para a primeira cidade)
    rotaDia = [rota[0]]  # Cidades visitadas no dia atual
    
    # Rastreia cidades únicas visitadas para dar bônus
    listaCidadesVisitadas = set([rota[0]])  # Conjunto de cidades únicas visitadas
    bonusCidadesNovas = 0                   # Bônus acumulado por cidades novas

    i = 0  # Índice para percorrer a rota

    while i < len(rota) and idxDiaAtual < totalDias:
        cidadeUm = rota[i]  # ID da cidade atual
        coordUm = tabelaCidades[int(cidadeUm), 1:]
        
        # Calcula distância da base para a primeira cidade do dia
        cidadeFirst = tabelaCidades[int(rotaDia[0]), 1:]
        distFirstBase = calcularDistancia(coordBase, cidadeFirst)
        
        # Calcula distância entre cidades já visitadas no dia
        distanciaDia = 0
        for idx in range(len(rotaDia) - 1):
            posicaoAtual = tabelaCidades[int(rotaDia[idx]), 1:]
            posicaoProxima = tabelaCidades[int(rotaDia[idx + 1]), 1:]
            distanciaDia += calcularDistancia(posicaoAtual, posicaoProxima)
        
        cidadeAdd = False
        tentativas = min(3, len(rota) - i - 1)  # Máximo de 3 tentativas ou até o fim da rota
        
        for tentativa in range(tentativas):
            if i + 1 + tentativa >= len(rota):
                break
                
            cidadeDois = rota[i + 1 + tentativa]  # ID da cidade candidata
            coordDois = tabelaCidades[int(cidadeDois), 1:]  # Coordenadas da cidade candidata
            distancia = calcularDistancia(coordUm, coordDois)  # Distância até a cidade candidata
            
            # Adiciona a nova distância e o retorno à base
            distanciaBase = calcularDistancia(coordDois, coordBase)
            distanciaDiaTotal = distFirstBase + distanciaDia + distancia + distanciaBase
            
            # Verifica se é possível visitar esta cidade no mesmo dia
            if distanciaDiaTotal <= distMax:
                visitasDia += 1 
                rotaDia.append(cidadeDois)  # Adiciona cidade à rota do dia
                
                # Verifica se é uma cidade nova e adiciona bônus
                if cidadeDois not in listaCidadesVisitadas:
                    listaCidadesVisitadas.add(cidadeDois)
                    bonusCidadesNovas += 1  # Bônus para cidades novas
                
                i += 1 + tentativa  # Avança para a próxima cidade (pula as tentativas)
                cidadeAdd = True
                break
        
        # Se nenhuma cidade pôde ser adicionada, finaliza o dia
        if not cidadeAdd:
            qtdCidadesVisitadas[idxDiaAtual] = visitasDia  # Registra número de visitas do dia
            idxDiaAtual += 1                               # Avança para o próximo dia
            visitasDia = 1                                 # Reinicia contador
            rotaDia = [cidadeUm]                           # Reinicia rota do dia com a cidade atual

            if idxDiaAtual == totalDias:
                break  

            i += 1 
    
    # Adiciona a pontuação do último dia (que pode estar incompleto)
    if visitasDia > 1 and idxDiaAtual < totalDias:
        qtdCidadesVisitadas[idxDiaAtual] = visitasDia

    # Fitness Total: soma das cidades visitadas + bônus por proporção + bônus por cidades novas
    qtdCidadesVisitadasTotal = np.sum(qtdCidadesVisitadas)     # Total de cidades visitadas
    bonusProporcao = qtdCidadesVisitadasTotal / len(rota)      # Proporção de cidades visitadas
    
    fitness = qtdCidadesVisitadasTotal + (bonusProporcao * 10) # Fitness base

    return fitness + bonusCidadesNovas                         # Adiciona bônus por cidades novas

def criarPop(tabelaCidades, tamanhoPop):
    cidadesID = list(tabelaCidades[:, 0])  # Extrai os IDs das cidades da matriz
    
    return [random.sample(cidadesID, len(cidadesID)) for _ in range(tamanhoPop)]

def escolherPais(fitnessess, tamanhoTorneio, selected_size):

    selecionados = []  # Lista para armazenar os pais selecionados
    
    for _ in range(selected_size):
        torneio = random.sample(fitnessess, tamanhoTorneio)  # Seleciona rotas aleatórias para o torneio
        winner = max(torneio, key=lambda x: x[1])[0]         # Escolhe a rota com maior fitness como vencedora
        selecionados.append(winner)
    
    return selecionados

def crossover(paiUm, paiDois):
    size = len(paiUm)  # Tamanho da rota (número de cidades)
    
    # Seleciona dois pontos aleatórios para definir o segmento a ser copiado do primeiro pai
    start, end = sorted(random.sample(range(size), 2))
    
    # Inicializa o filho como uma lista de valores nulos
    filho = [None] * size
    
    # Passo 1: Copia o segmento do primeiro pai para o filho
    filho[start:end] = paiUm[start:end]
    
    # Passo 2: Identifica as cidades que não foram copiadas do primeiro pai
    resto = [cidade for cidade in paiDois if cidade not in filho]
    
    # Passo 3: Preenche as posições vazias com as cidades restantes, na ordem do segundo pai
    filho = [cidade if cidade is not None else resto.pop(0) for cidade in filho]
    
    return filho

def taxaMutacaoAdaptativa(geracao, geracoes, taxaInicial, taxaFinal):
    # Progressão Linear da taxa de mutação
    progressao = geracao / (geracoes - 1)  # 0.0 a 1.0
    taxaAtual = taxaInicial - (taxaInicial - taxaFinal) * progressao
    
    # Garante que a taxa não seja negativa
    return max(taxaAtual, taxaFinal)

def mutacao(rota, taxaMutacao):
    # Determina aleatoriamente se ocorrerá mutação baseado na taxa fornecida
    if random.random() < taxaMutacao:
        # Seleciona aleatoriamente duas posições diferentes na rota
        idx1, idx2 = random.sample(range(len(rota)), 2)
        
        # Troca as cidades nas posições selecionadas
        rota[idx1], rota[idx2] = rota[idx2], rota[idx1]
    
    return rota



def optimize_route_2opt_constrained(route, cities):
    """
    Otimização 2-opt que respeita restrições de distância diária e retorno à base.
    Usa fitness function como critério de otimização em vez de distância total.
    """
    melhorou = True
    melhorRota = route.copy()
    melhorFitness = fitnessFunction(melhorRota, cities)
  
    while melhorou:
        melhorou = False
      
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
        
                novaRota = two_opt_swap(melhorRota, i, j)
                novoFitness = fitnessFunction(novaRota, cities)
                
                # Só aceita se melhorar o fitness (que já considera todas as restrições)
                if novoFitness > melhorFitness:
                    melhorFitness = novoFitness
                    melhorRota = novaRota
                    melhorou = True
    
    return melhorRota

def two_opt_swap(route, i, j):
    novaRota = route.copy()
    novaRota[i:j+1] = reversed(route[i:j+1])
    
    return novaRota

def calcularDistanciaTotal(route, cities):
    total = 0
    
    for i in range(len(route) - 1):
        cidade1 = cities[int(route[i]), 1:]
        cidade2 = cities[int(route[i+1]), 1:]
        total += calcularDistancia(cidade1, cidade2)
    
    return total

def simple_clustering(cities, num_clusters = totalDias):
    coords = cities[:, 1:]
    city_ids = cities[:, 0].astype(int)
    n_cities = len(cities)
  
    centroids_idx = np.random.choice(n_cities, num_clusters, replace=False)
    centroids = coords[centroids_idx]
    
    clusters = np.zeros(n_cities, dtype=int)
    
    max_iterations = 20
    for _ in range(max_iterations):
        for i in range(n_cities):
            distances = [calcularDistancia(coords[i], centroid) for centroid in centroids]
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

def algoritmoGenetico(cities,
                      tamanhoPop,
                      geracoes,
                      taxaMutacao,
                      tamanhoTorneio,
                      tamanhoElitismo):
    clustered_cities = simple_clustering(cities)
    pop = criarPop(cities, tamanhoPop - 1)
    fitnessess: list[tuple[np.ndarray, float]] = [(rota, fitnessFunction(rota, cities)) for rota in pop]

    pop.append(clustered_cities)
    
    melhoresFitnesses = []
    mediasFitnesses = []

    melhorRota, melhorRotaFitness = max(fitnessess, key=lambda x: x[1])
    mediaRotaFitness = sum(fit[1] for fit in fitnessess)/len(fitnessess)

    melhoresFitnesses.append(melhorRotaFitness)
    mediasFitnesses.append(mediaRotaFitness)
  
    for geracao in range(geracoes):
        # Calcula taxa de mutação adaptativa para esta geração
        taxaMutacaoAtual = taxaMutacaoAdaptativa(geracao, geracoes, taxaMutacao, 0.01)
        
        paisEscolhidos = escolherPais(fitnessess, tamanhoTorneio, tamanhoElitismo)
        novaPop = paisEscolhidos[:]
      
        while len(novaPop) < tamanhoPop:
            paiUm, paiDois = random.sample(paisEscolhidos, 2)
            filho = crossover(paiUm, paiDois)
            filho = mutacao(filho, taxaMutacaoAtual)
            novaPop.append(filho)
      
        # Otimização 2-opt com restrições aplicada a cada 10 gerações
        if geracao % 10 == 0 and melhorRota is not None:
            rotaOtimizada = optimize_route_2opt_constrained(melhorRota, cities)
            novaPop.append(rotaOtimizada)
            if len(novaPop) > tamanhoPop:
                idxToRemove = random.randint(0, len(novaPop) - 2)
                novaPop.pop(idxToRemove)
      
        novosFitnessess = [(rota, fitnessFunction(rota, cities)) for rota in novaPop]
        
        novaMelhorRota, novoMelhorFitness = max(novosFitnessess, key=lambda x: x[1])
        novaMediaFitness = sum(fit[1] for fit in novosFitnessess)/len(novosFitnessess)

        melhoresFitnesses.append(novoMelhorFitness)
        mediasFitnesses.append(novaMediaFitness)

        if novoMelhorFitness > melhorRotaFitness:
            melhorRotaFitness = novoMelhorFitness
            melhorRota = novaMelhorRota
        
        pop = novaPop
        fitnessess = novosFitnessess
      
    # Aplicação final da otimização 2-opt com restrições
    if melhorRota is not None:
        melhorRota = optimize_route_2opt_constrained(melhorRota, cities)
    
    return melhorRota, mediasFitnesses, melhoresFitnesses

def main():
    # Carrega dados das cidades a partir do arquivo CSV
    cities_df = load_cities('assets/cities.csv')
    cities = cities_df.to_numpy()
    
    print(f"Iniciando algoritmo genético para roteirização de {len(cities)} cidades em {totalDias} dias...")
    print(f"Distância máxima por dia: {distMax} unidades")
    print(f"Mutação adaptativa: 0.3 -> 0.01 ao longo das gerações\n")

    # Executa o algoritmo genético com parâmetros definidos
    melhorRota, mediaFitnesses, melhorFitnesses = algoritmoGenetico(cities=cities,
                                                                   tamanhoPop=300,
                                                                   geracoes=600,
                                                                   taxaMutacao=0.3,
                                                                   tamanhoTorneio=3,
                                                                   tamanhoElitismo=2)    
    
    print("\nAlgoritmo genético concluído. Organizando resultados...")
    
    # Agrupa as cidades da melhor rota por dias de viagem
    melhorRotaAgrupada = group_cities(melhorRota, cities)
    melhorRotaAchatada = [city for day in melhorRotaAgrupada for city in day]  # Achata a lista de cidades visitadas

    totalCidades = len(melhorRotaAchatada)                  # Total de cidades visitadas (com repetições)
    totalCidadesUnicas = len(np.unique(melhorRotaAchatada)) # Total de cidades únicas visitadas

    print("\n=== ESTATÍSTICAS DAS GERAÇÕES ===")
    tabelaGeracoes = PrettyTable()  # Cria uma tabela para exibir estatísticas das gerações
    tabelaGeracoes.field_names = ["Geração", "Melhor fitness", "Fitness médio"]
    tabelaGeracoes.add_rows([[i+1, mediaFitnesses[i], melhorFitnesses[i]] for i in range(len(mediaFitnesses))])
    print(tabelaGeracoes)

    # Plota as estatísticas de fitness
    plt.plot(melhorFitnesses)
    plt.plot(mediaFitnesses)
    plt.legend(['Melhor fitness', 'Fitness médio'])
    
    print("\n=== ESTATÍSTICAS DA SOLUÇÃO ===")
    print(f"Total de cidades visitadas (com repetições): {totalCidades}")
    print(f"(Essas repetições ocorrem, pois são contabilizadas as mesmas cidades visitadas em dias diferentes: cidade final e cidade inicial de um dia)")
    print(f"Total de cidades únicas visitadas: {totalCidadesUnicas} de {len(cities)}")
    print(f"Porcentagem de cobertura: {totalCidadesUnicas/len(cities)*100:.2f}%")
    
    print("\n=== DETALHAMENTO POR DIA ===")
    tabelaDias = PrettyTable()  # Cria uma tabela para exibir detalhes por dia
    tabelaDias.field_names = ["Dia", "Quantidade total", "Quantidade única", "Cidades"]
    tabelaDias.add_rows([[i+1, len(dia), len(dia)-1 if i > 0 else len(dia), np.array(dia)] for i, dia in enumerate(melhorRotaAgrupada)])
    print(tabelaDias)
    
    # Visualiza a rota encontrada
    plot_paths(cities_df, melhorRotaAgrupada, return_to_base=True, base_coords=(0.0, 0.0))
    
    print("\nProcesso concluído.")

if __name__ == "__main__":
    main()