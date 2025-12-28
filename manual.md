# 📖 Manual de Utilizador

Este guia descreve como configurar e executar o sistema de otimização de rotas multimodais no Porto.

---

## 🚀 Execução Rápida

Para obter uma rota imediatamente, siga estes passos:

1.  **Ativar o Ambiente:**
    Certifique-se de que o ambiente Conda está ativo no seu terminal:
    ```bash
    conda activate geo_opt_env
    ```

2.  **Abrir o Notebook:**
    Inicie o Jupyter Notebook na raiz do projeto:
    ```bash
    jupyter notebook main.ipynb
    ```

3.  **Executar:**
    No menu superior do Jupyter, selecione **Kernel** > **Restart & Run All**.
    *Isto garante que todas as bibliotecas são carregadas e o grafo é construído corretamente.*

---

## 📍 Configuração de Cenários de Teste

No início do notebook `code.ipynb`, encontrará a secção de input identificada como **"Configuração da Viagem"**. Pode alterar as variáveis abaixo para testar diferentes complexidades.

### 1. Definir Data e Hora
A hora influencia a disponibilidade dos transportes (horários GTFS).

```python
start_datetime = datetime(2025, 1, 15, 8, 30, 0) # datetime(ano, mês, dia, horas, minutos, segundos)
```
### 2. Definir Origem e Destino
As coordenadas devem ser inseridas no formato (Latitude, Longitude).
    Nota: Pode obter coordenadas clicando no Google Maps ou no geojson.io.

```python
START_COORDS = (41.1584, -8.6291)  # Cordenadas do ponto de início: (Latitude, Longitude)
END_COORDS   = (41.1404, -8.6118)  # Cordenadas do ponto de fim: (Latitude, Longitude)
```

## 📊 Interpretação dos Resultados
Após a execução (pode demorar cerca de 1-2 minutos na primeira vez para carregar o grafo), o notebook apresentará três saídas principais:

### 1. Frente de Pareto (Gráfico de Dispersão)
Um gráfico com dois eixos conflituantes:
- Eixo X: Tempo Total (Segundos)
- Eixo Y: Emissões de CO2 (Gramas)

O algoritmo destaca automaticamente 3 soluções de interesse:
- Melhor CO2: A rota mais ecológica.
- Melhor Tempo: A rota mais rápida.
- Equilibrio: O ponto de compromisso ideal ("Knee point").

### 2. Métricas da Rota
No output textual, serão apresentados os valores exatos calculados com base nos coeficientes de sustentabilidade:

- STCP: 109.9 gCO2/km
- Metro: 40.0 gCO2/km
- Caminhada: 0 gCO2/km

### 3. Visualização no Mapa
O trajeto final é desenhado sobre o mapa da cidade com as seguintes cores:

- 🔴 Linha Vermelha: Metro
- 🔵 Linha Azul: Autocarro (STCP)
- 🟢 Linha Verde: Caminhada (Walk)