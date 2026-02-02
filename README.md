# 🚇 Otimização de Rotas Multimodais no Porto

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Grade](https://img.shields.io/badge/Nota_Final-18%2F20-brightgreen)
![Status](https://img.shields.io/badge/Status-Concluído-success)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> **Projeto de Computação para a Engenharia (CIN)** | Universidade do Minho

Este projeto visa calcular, visualizar e otimizar rotas multimodais (**Metro, Autocarro e Pedonal**) na cidade do Porto. Utilizando algoritmos de teoria de grafos, a solução foca-se não apenas na rapidez, mas também na sustentabilidade ambiental.

---

## 🎯 Objetivos e Algoritmos

O núcleo do projeto é um motor de busca de caminhos que implementa algoritmos de caminho mínimo (como **Dijkstra** e **A***) sobre um grafo multimodal complexo.

O sistema resolve um problema de otimização bi-critério:
1.  **⏳ Minimização do Tempo:** Cálculo da rota mais rápida considerando tempos de espera e transbordos.
2.  **🌱 Minimização da Pegada Ecológica:** Cálculo baseado em dados reais de emissões de CO₂.

### Dados de Sustentabilidade Utilizados
Para o cálculo de custos ambientais, foram utilizados os seguintes coeficientes baseados nos operadores locais:
* **STCP (Autocarro):** 109.9 gCO₂/P.km
* **Metro do Porto:** 40 gCO₂/P.km
* **Mobilidade Suave:** 0 gCO₂ (Caminhada)

---

## 📂 Estrutura do Repositório

### 💻 Código Fonte
* `code.ipynb`: **Interface Principal**. Notebook interativo onde o utilizador define origem/destino e visualiza os mapas e as fronteiras de Pareto.
* `func.py`: **Core Logic**. Contém a construção do grafo multimodal, implementação dos algoritmos de otimização e funções de custo.
* `dev.ipynb`: Ambiente de desenvolvimento, validação de algoritmos e testes unitários.

### 📊 Dados e Recursos (`/data`)
* `bus/`: Dados GTFS da rede STCP.
* `transit/`: Dados da rede do Metro do Porto.
* `walk/`: Rede viária extraída do OpenStreetMap (via OSMnx).

### 📄 Documentação
* `report.pdf`: Relatório técnico detalhado com a análise teórica e resultados.
* `manual.md`: Manual de instruções para execução e testes.
* `presentation.pdf`: Slides de apresentação do projeto.

---

## 👥 Autores

* **[Guilherme Pinto]** - [PG60225@alunos.uminho.pt]
* **[Pedro Reis]** - [PG59908@alunos.uminho.pt]
* **[Luís Silva]** - [PG60390@alunos.uminho.pt]
* **[João Azevedo]** - [PG61693@alunos.uminho.pt]

---

## ⚙️ Instalação e Ambiente

Este projeto utiliza `conda` para gestão de dependências e bibliotecas geoespaciais (`osmnx`, `geopandas`, etc.).

1. **Clonar o repositório:**
   
   ```bash
   git clone https://github.com/Luismpso/CIN.git
   ```

2. **Criar o ambiente virtual:**
   
   ```bash
   conda env create -f env.yml
   ```

3.  **Ativar o ambiente:**
    ```bash
    conda activate geo_opt_env
    ```

## 📚 Referências e Dados

* **Dados de Mobilidade:** [Porto Digital - Infraestruturas e Mobilidade](https://opendata.porto.digital/dataset/?q=Infraestruturas+e+Mobilidade&res_format=GTFS)
* **Sustentabilidade:** [Metro do Porto](https://www.metrodoporto.pt/pages/358) e [STCP](https://www.stcp.pt/pt/institucional/sustentabilidade/politica-energetica/)
* **Mapas de Fundo:** OpenStreetMap


