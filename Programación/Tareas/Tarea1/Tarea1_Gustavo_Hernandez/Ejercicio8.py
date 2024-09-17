# Recorrido

def recorrido():
    # Inputs
    N = int(input())
    matriz = list()
    for i in range(N):
        matriz.append([int(x) for x in input().split(" ")])

    # Haremos una matriz de costos hasta llegar a la casilla i,j.
    # Para el primer renglón solo podemos llegar por la izquierda.
    # Y para la primera columna solo podemos llegar por arriba.
    matriz_costos = matriz
    for i in range(1,N):
        matriz_costos[i][0] += matriz_costos[i-1][0]
        matriz_costos[0][i] += matriz_costos[0][i-1]
    
    # Para el resto de celdas, hacemos la suma de su costo
    # con el mínimo entre el costo de la celda de arriba y la de la izquierda.
    for i in range(1,N):
        for j in range(1,N):
            matriz_costos[i][j] += min(matriz_costos[i-1][j],matriz_costos[i][j-1])

    # Imprimimos la última celda que contendría el menor costo para llegar a ella.
    print(matriz_costos[N-1][N-1])


recorrido()
    