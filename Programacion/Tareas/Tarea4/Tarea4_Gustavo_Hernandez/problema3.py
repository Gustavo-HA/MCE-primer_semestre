# La ruta del caballo

# Es un ejercicio de encontrar el camino más corto,
# con la diferencia de que los movimientos los haremos como si fueramos un
# caballo de ajedrez, y de que guardaremos el numero de pasos en lugar de el camino.

from collections import deque

def es_valido(x, y, renglones, columnas, matriz, visitados):
    return 0 <= x < renglones and 0 <= y < columnas and matriz[x][y] != '#' and (x,y) not in visitados

def caballo_camino_corto(matriz, inicio: tuple, final: tuple):
    renglones = len(matriz)
    columnas = len(matriz[0])
    
    # Como se mueve el caballo
    movimientos = [(-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2)]

    cola = deque([(inicio, 0)])
    visitados = set([inicio])
    while cola:

        (x, y), pasos = cola.popleft()

        if (x, y) == final:
            return pasos
        
        for dx, dy in movimientos:
            nx, ny = x + dx, y + dy
            if es_valido(nx, ny, renglones, columnas, matriz, visitados):
                visitados.add((nx, ny))
                cola.append(((nx, ny), pasos + 1))

    return -1



#Inputs
tamano = input().split(" ")
N = int(tamano[0])
M = int(tamano[1])

inicial = input().split(" ")
inicial = (int(inicial[0])-1, int(inicial[1])-1)

final = input().split(" ")
final = (int(final[0])-1, int(final[1])-1)

n_obstaculos = int(input())
obstaculos = list()
for _ in range(n_obstaculos):
    obstaculo = input().split(" ")
    obstaculo = (int(obstaculo[0])-1,int(obstaculo[1])-1)
    obstaculos.append(obstaculo)

matriz = [["." for _ in range(N)] for _ in range(M)]
for obstaculo in obstaculos:
    x = obstaculo[0]
    y = obstaculo[1]
    matriz[x][y] = "#"

# Outputs

resultado = caballo_camino_corto(matriz, inicial, final)
print(resultado)