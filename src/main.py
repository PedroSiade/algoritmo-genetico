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
totalDias = 5  # Número de dias disponíveis para visitar cidades
distMax = 40.0  # Distância máxima que pode ser percorrida em um único dia

def calcularDistancia(cidadeUm, cidadeDois):
    distanciaX = cidadeUm[0] - cidadeDois[0]  
    distanciaY = cidadeUm[1] - cidadeDois[1] 

    return np.sqrt(distanciaX**2 + distanciaY**2)  

def agruparCidades(rota, cities):
    coordBase = [0.0, 0.0]  # Posição da base (0,0)
    totalVisitas = []  # Lista final com agrupamentos de cidades por dia
    idxDiaAtual = 0  # Dia atual
    visitasDia = [rota[0]]  # Inicializa com a primeira cidade no primeiro dia

    i = 0  # Índice da cidade atual na rota
    # Percorre a rota enquanto houver cidades e dias disponíveis
    while i < len(rota) and idxDiaAtual < totalDias:
        cidadeUm = rota[i]  # ID da cidade atual
        coordUm = cities[int(cidadeUm), 1:]  # Coordenadas (x,y) da cidade atual
        
        # Calcula distância da base para a primeira cidade do dia
        cidadeFirst = cities[int(visitasDia[0]), 1:]
        distFirstBase = calcularDistancia(coordBase, cidadeFirst)        
        # Calcula distância entre cidades já visitadas no dia
        distanciaDia = 0
        for idx in range(len(visitasDia) - 1):
            posicaoAtual = cities[int(visitasDia[idx]), 1:]
            posicaoProxima = cities[int(visitasDia[idx + 1]), 1:]
            distanciaDia += calcularDistancia(posicaoAtual, posicaoProxima)
        
        # Verifica se a próxima cidade pode ser adicionada ao dia
        if i + 1 < len(rota):
            cidadeDois = rota[i + 1]  # ID da próxima cidade
            coordDois = cities[int(cidadeDois), 1:]  # Coordenadas da próxima cidade
            distancia = calcularDistancia(coordUm, coordDois)  # Distância até a próxima cidade
            
            # Adiciona a nova distância e o retorno à base
            distanciaBase = calcularDistancia(coordDois, coordBase)
            distanciaDiaTotal = distFirstBase + distanciaDia + distancia + distanciaBase
            
            # Verifica se adicionar esta cidade não ultrapassa o limite diário
            if distanciaDiaTotal <= distMax:
                visitasDia.append(cidadeDois)  # Adiciona cidade ao dia atual
                i += 1  # Avança para a próxima cidade
            else:
                # Se a próxima cidade não cabe, finaliza o dia
                totalVisitas.append(visitasDia)  # Adiciona as visitas do dia à lista total
                idxDiaAtual += 1  # Avança para o próximo dia
                visitasDia = [cidadeUm]  # Inicializa o próximo dia com a cidade atual
                if idxDiaAtual == totalDias: break  # Encerra se acabaram os dias
                i += 1  # Avança para a próxima cidade na rota
        else:
            # Se não há mais cidades na rota, finaliza o dia
            totalVisitas.append(visitasDia)  # Adiciona as visitas do dia à lista total
            idxDiaAtual += 1  # Avança para o próximo dia
            visitasDia = [cidadeUm]  # Inicializa o próximo dia com a cidade atual
            if idxDiaAtual == totalDias: break  # Encerra se acabaram os dias
            i += 1  # Avança para a próxima cidade na rota
    
    # Adiciona o último grupo de visitas se ainda houver dias disponíveis
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
        
        # Verifica se a próxima cidade pode ser adicionada ao dia
        if i + 1 < len(rota):
            cidadeDois = rota[i + 1]  # ID da próxima cidade
            coordDois = tabelaCidades[int(cidadeDois), 1:]  # Coordenadas da próxima cidade
            distancia = calcularDistancia(coordUm, coordDois)  # Distância até a próxima cidade
            
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
                
                i += 1  # Avança para a próxima cidade
            else:
                # Se a próxima cidade não cabe, finaliza o dia
                qtdCidadesVisitadas[idxDiaAtual] = visitasDia  # Registra número de visitas do dia
                idxDiaAtual += 1                               # Avança para o próximo dia
                visitasDia = 1                                 # Reinicia contador
                rotaDia = [cidadeUm]                           # Reinicia rota do dia com a cidade atual

                if idxDiaAtual == totalDias:
                    break  

                i += 1
        else:
            # Se não há mais cidades na rota, finaliza o dia
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

def escolherPais(fitnessess, tamanhoTorneio, tamanhoElitismo):

    selecionados = []  # Lista para armazenar os pais selecionados
    
    for _ in range(tamanhoElitismo):
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
    progressao = geracao / (geracoes - 1)
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

def algoritmoGenetico(cities,
                      tamanhoPop,
                      geracoes,
                      taxaMutacao,
                      tamanhoTorneio,
                      tamanhoElitismo):

    # Fase 1: Inicialização - Cria a população inicial
    pop = criarPop(cities, tamanhoPop)  # Gera a população inicial
    fitnessess: list[tuple[np.ndarray, float]] = [(rota, fitnessFunction(rota, cities)) for rota in pop]  # Avalia a fitness de cada rota
    
    melhoresFitnesses = [] # Lista para armazenar o melhor fitness ao longo das gerações
    mediasFitnesses = []   # Lista para armazenar a média de fitness ao longo das gerações

    # Inicializa as melhores rotas e fitness
    melhorRota, melhorRotaFitness = max(fitnessess, key=lambda x: x[1])
    mediaRotaFitness = sum(fit[1] for fit in fitnessess) / len(fitnessess)

    melhoresFitnesses.append(melhorRotaFitness) # Adiciona o melhor fitness à lista
    mediasFitnesses.append(mediaRotaFitness)    # Adiciona a média de fitness à lista

    # Fase 2: Evolução - Executa o algoritmo pelo número especificado de gerações
    for geracao in range(geracoes):
        # Calcula taxa de mutação adaptativa para esta geração
        taxaMutacaoAtual = taxaMutacaoAdaptativa(geracao, geracoes, taxaMutacao, 0.1)
        
        paisEscolhidos = escolherPais(fitnessess, tamanhoTorneio, tamanhoElitismo)
        novaPop = paisEscolhidos[:]  # Inicia nova população com os pais selecionados

        # Completa a população com novos indivíduos gerados por cruzamento e mutação
        while len(novaPop) < tamanhoPop:
            paiUm, paiDois = random.sample(paisEscolhidos, 2)  # Seleciona aleatoriamente dois pais diferentes
            filho = crossover(paiUm, paiDois)                  # Gera um filho por cruzamento
            filho = mutacao(filho, taxaMutacaoAtual)           # Aplica mutação adaptativa ao filho
            novaPop.append(filho)                              # Adiciona o filho à nova população
        
        # Avalia a nova população
        novosFitnessess = [(rota, fitnessFunction(rota, cities)) for rota in novaPop]
        
        # Atualiza as melhores rotas e fitness
        novaMelhorRota, novoMelhorFitness = max(novosFitnessess, key=lambda x: x[1])
        novaMediaFitness = sum(fit[1] for fit in novosFitnessess) / len(novosFitnessess)

        melhoresFitnesses.append(novoMelhorFitness) # Adiciona o novo melhor fitness à lista
        mediasFitnesses.append(novaMediaFitness)    # Adiciona a nova média de fitness à lista

        # Atualiza a melhor rota se necessário
        if novoMelhorFitness > melhorRotaFitness:
            melhorRotaFitness = novoMelhorFitness
            melhorRota = novaMelhorRota
        
        pop = novaPop                 # Atualiza a população para a próxima geração
        fitnessess = novosFitnessess  # Atualiza a lista de fitness

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
                                                                   tamanhoPop=500,     # 500 indivíduos na população
                                                                   geracoes=600,       # 600 gerações
                                                                   taxaMutacao=0.3,    # 30% de chance de mutação
                                                                   tamanhoTorneio=3,   # Tamanho do Torneio
                                                                   tamanhoElitismo=2)  # Número de pais selecionados
    
    print("\nAlgoritmo genético concluído. Organizando resultados...")
    
    # Agrupa as cidades da melhor rota por dias de viagem
    melhorRotaAgrupada = agruparCidades(melhorRota, cities)
    melhorRotaAchatada = [city for day in melhorRotaAgrupada for city in day]  # Achata a lista de cidades visitadas

    totalCidades = len(melhorRotaAchatada)   # Total de cidades visitadas

    print("\n=== ESTATÍSTICAS DAS GERAÇÕES ===")
    tabelaGeracoes = PrettyTable()  # Cria uma tabela para exibir estatísticas das gerações
    tabelaGeracoes.field_names = ["Geração", "Melhor fitness", "Fitness médio"]
    tabelaGeracoes.add_rows([[i+1, mediaFitnesses[i], melhorFitnesses[i]] for i in range(len(mediaFitnesses))])  # Adiciona dados à tabela
    print(tabelaGeracoes)

    # Plota as estatísticas de fitness
    plt.plot(melhorFitnesses)
    plt.plot(mediaFitnesses)
    plt.legend(['Melhor fitness', 'Fitness médio'])
    
    print("\n=== ESTATÍSTICAS DA SOLUÇÃO ===")
    print(f"Total de cidades visitadas: {totalCidades}")
    print(f"Porcentagem de cobertura: {totalCidades/len(cities)*100:.2f}%")
    
    print("\n=== DETALHAMENTO POR DIA ===")
    tabelaDias = PrettyTable()  # Cria uma tabela para exibir detalhes por dia
    tabelaDias.field_names = ["Dia", "Quantidade total", "Cidades"]
    tabelaDias.add_rows([[i+1, len(dia), np.array(dia)] for i, dia in enumerate(melhorRotaAgrupada)])
    print(tabelaDias)
    
    # Visualiza a rota encontrada
    plot_paths(cities_df, melhorRotaAgrupada, return_to_base=True, base_coords=(0.0, 0.0))
    
    print("\nProcesso concluído.")

if __name__ == "__main__":
    main()