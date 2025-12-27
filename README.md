# Otimização de Rotas Multimodais (Porto)

Este projeto visa calcular e visualizar rotas multimodais (Metro, Autocarro e Pedonal) na cidade do Porto, utilizando algoritmos de grafos e otimização.

## 👥 Autores

* **[Guilherme Pinto]** - [PG60225@alunos.uminho.pt]
* **[Luís Silva]** - [PG60390@alunos.uminho.pt]
* **[João Azevedo]** - [PG61693@alunos.uminho.pt]
* **[Pedro Reis]** - [PG59908@alunos.uminho.pt]

---

## 🎯 Objetivos

O algoritmo desenvolvido permite encontrar soluções que não só minimizam o tempo, mas também consideram a pegada ecológica.

### Objetivos de Otimização:

1. **Minimizar o Tempo de Viagem**.
2. **Minimizar Emissões de $CO_2$**:
   * Utiliza dados reais de sustentabilidade:
   * **STCP:** 109.9 g$CO_2$/P.km
   * **Metro:** 40 g$CO_2$/P.km
3. **Transbordos e Caminhada:** Consideração de limites para o número de trocas e distância a pé.

---

## 📂 Estrutura do Repositório

### Código Fonte

* **`main.ipynb`**: **Interface Principal**. Notebook onde o utilizador define a origem/destino e visualiza os mapas e gráficos de Pareto.
* **`func.py`**: Módulo contendo a lógica de domínio: construção do grafo multimodal, implementação dos algoritmos de caminho mínimo (Dijkstra/A*) e cálculo de custos ($CO_2$, Tempo).
* **`dev.ipynb`**: Ambiente de desenvolvimento e validação dos algoritmos e exploração inicial dos dados.

### Dados

* **`bus/`**: Dados da rede STCP (GTFS).
* **`transit/`**: Dados da rede do Metro do Porto.
* **`walk/`**: Dados da rede viária (OpenStreetMap via OSMnx).

### Documentação

* **`env.yml`**: Ficheiro de configuração do ambiente (Conda) para replicabilidade.
* **`manual.md`**: Instruções detalhadas de execução e criação de cenários de teste.
* **`report.pdf`**: Relatório técnico.
* **`presentation.pdf`**: Suporte visual para a apresentação do projeto.

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


