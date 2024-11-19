# El sistema de transporte dimensional
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


def problema3(N, coeficientes, consulta):
    inicio = consulta[0]
    fin = consulta[1]
    F = consulta[2]
    
    # Hacemos el grafo calculando costos con los 
    # coeficientes.
    grafo = dict()
    for i in range(1,N+1):
        grafo[i] = dict()
    
    for ciudad in coeficientes.keys():
        for vecino in coeficientes[ciudad]:
            A = coeficientes[ciudad][vecino][0]
            B = coeficientes[ciudad][vecino][1]
            grafo[ciudad][vecino] = A + B*F
            grafo[vecino][ciudad] = A + B*F
    
    distancias = dijkstra(grafo, inicio)
    respuesta = distancias[fin]
    print(respuesta)
    


#Input
N, M, Q = map(int, input().split(" "))



coeficientes = dict()
for i in range(1,N+1):
    coeficientes[i] = dict()

for _ in range(M):
    # Se captura u, v, A, B
    u, v, A, B = map(int, input().split(" "))
    coeficientes[u][v] = (A, B)
    coeficientes[v][u] = (A, B)

consultas = list()
for _ in range(Q):
    # Se captura las ciudades C, D, y fuerza dimensional F.
    C, D, F = map(int, input().split(" "))
    consultas.append((C,D,F))


# Output

for consulta in consultas:
    problema3(N, coeficientes, consulta)
