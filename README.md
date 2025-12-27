# Planeamento Multi-Objetivo de Rotas Multimodais - Grande Porto

Este projeto implementa um sistema de encaminhamento multimodal (Metro, Autocarro e Pedonal) na área do Grande Porto. O sistema utiliza algoritmos evolutivos (MOEA/D) e de procura (A*) para oferecer rotas otimizadas considerando três critérios fundamentais: **Tempo**, **Sustentabilidade ($CO_2$)** e **Saúde**.

---

## 👥 Autores

Trabalho realizado no âmbito da Unidade Curricular de Computação Inteligente (Universidade do Minho).

| Nome | Número |
| :--- | :--- |
| **João Azevedo** | PG61693 |
| **Luís Silva** | PG60390 |
| **Guilherme Pinto** | PG60225 |
| **Pedro Reis** | PG59908 |

---

## 📂 Estrutura do Repositório

O projeto está organizado de forma a separar os dados brutos da lógica de implementação e da interface de utilização.

```text
├── 🚌 bus/                 # Contém os dados GTFS da rede de autocarros (STCP)
├── 🚇 transit/             # Contém os dados GTFS da rede de Metro do Porto
├── 🚶 walk/                # Contém os dados da rede pedonal (extraídos via OSM)
│
├── 📜 functions.py         # Módulo com todas as funções auxiliares e lógica do algoritmo
├── 📓 desenvolvimento.ipynb # Notebook de "sandbox" usado para testes e desenvolvimento
└── 📓 main.ipynb           # Notebook PRINCIPAL: Onde o programa é executado
