## Trabalho - Algoritmo Genético

### Descrição do Projeto

Este projeto implementa um **Algoritmo Genético** para resolver o problema de **roteirização de cidades** com restrições de tempo e distância.

### Problema a ser Resolvido

O objetivo é **maximizar o número de cidades únicas visitadas** em **5 dias**, respeitando as seguintes restrições:

- **Limite de distância diária**: 40 unidades por dia
- **Retorno obrigatório à base**: Cada dia deve terminar na base (coordenadas 0,0)
- **Cidades únicas**: Evitar repetir cidades já visitadas
- **Otimização de rota**: Encontrar a melhor sequência de visitas

### Estratégias Implementadas

- **Heurística do Vizinho Mais Próximo**: População inicial inteligente
- **Mutação Adaptativa**: Taxa de mutação que diminui ao longo das gerações
- **Cruzamento de Ordem (OX)**: Preserva ordem relativa das cidades
- **Seleção por Torneio**: Escolha dos melhores indivíduos
- **Otimização 2-opt**: Refinamento local das melhores soluções

### Etapas para Executar

1. Baixe o projeto
2. Instale python e conda no seu ambiente de desenvolvimento
3. Rode o comando `conda env create -f environment.yml`
4. Rode o comando `conda activate trabalhoAG`
5. Rode o comando `python ./src/main.py` para executar o algoritmo genético
