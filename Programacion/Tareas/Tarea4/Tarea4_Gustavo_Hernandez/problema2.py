# Rapido y seguro..

# Otra vez utilizamos al poderoso dijkstra.

import heapq
def dijkstra(grafo, inicio):
    # Inicializar distancias a todos los nodos como infinito y el nodo de inicio como 0
    distancias = {nodo: float('inf') for nodo in grafo}
    distancias[inicio] = 0

    # Inicializar una cola de prioridad (heap) con el nodo de inicio
    cola_prioridad = [(0, inicio)]

    while cola_prioridad:
        # Obtener el nodo con la distancia más corta de la cola de prioridad
        distancia_actual, nodo_actual = heapq.heappop(cola_prioridad)

        # Si encontramos una distancia más corta a través de este nodo, actualizamos la distancia
        if distancia_actual > distancias[nodo_actual]:
            continue

        # Explorar los vecinos del nodo actual
        for vecino, peso in grafo[nodo_actual].items():
            distancia = distancia_actual + peso

            # Si encontramos una distancia más corta hacia el vecino, actualizamos la distancia
            if distancia < distancias[vecino]:
                distancias[vecino] = distancia
                heapq.heappush(cola_prioridad, (distancia, vecino))

    return distancias

# Input
N = int(input())
M = int(input())
grafo = dict()
for i in range(1,N+1):
    grafo[i] = dict()
for _ in range(M):
    linea = input().split(" ")
    linea = [int(x) for x in linea]
    # Guardamos tanto la distancia como la seguridad de la carretera
    grafo[linea[0]][linea[1]] = (linea[2],linea[3])
    grafo[linea[1]][linea[0]] = (linea[2],linea[3])

entrada_salida_seguridad = input().split(" ")
entrada = int(entrada_salida_seguridad[0])
salida = int(entrada_salida_seguridad[1])
seguridad = int(entrada_salida_seguridad[2])

# Podemos "limpiar" el grafo para que solo contenga las carreteras que tienen una seguridad 
# mayor a C.
grafo_limpio = dict()
for nodo, adyacentes in grafo.items():
    grafo_limpio[nodo] = dict()
    for vecino, peso_y_seguridad in adyacentes.items():
        # Verificamos que su seguridad sea mayor a C.
        # En dado caso, se llena el grafo_limpio con esta ruta.
        # Si no, simplemente no lo agregamos
        if peso_y_seguridad[1] >= seguridad: grafo_limpio[nodo][vecino] = peso_y_seguridad[0]

distancias = dijkstra(grafo_limpio, entrada)

# Output
resultado = distancias[salida]
if resultado == float('inf'): resultado = -1 # Si no existe camino seguro
print(resultado)