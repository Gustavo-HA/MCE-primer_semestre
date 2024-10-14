# Vamos a repartir pero bien.

# Necesitamos el algoritmo de Dijkstra, asumiendo que siempre hay una ruta válida.

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
for i in range(M):
    costo = input().split(" ")
    costo = [int(x) for x in costo]
    grafo[costo[0]][costo[1]] = costo[2]
    grafo[costo[1]][costo[0]] = costo[2] 
entrada_y_salida = input().split()
entrada = int(entrada_y_salida[0])
salida = int(entrada_y_salida[1])


#Output
distancias = dijkstra(grafo, entrada)
resultado = distancias[salida]
print(resultado)

