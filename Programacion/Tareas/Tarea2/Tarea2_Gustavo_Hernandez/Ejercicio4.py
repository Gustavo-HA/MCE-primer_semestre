# Rutas del laberinto

# Solo nos podemos mover hacia la derecha o hacia la izquierda.

# Definimos una variable global que contará el número de 
# soluciones encontradas.

soluciones = 0


def contar_soluciones(n, matriz, x=0, y=0):
    global soluciones
    
    # Si sale de la matriz, no hacer nada 
    if x < 0 or x >= n or y < 0 or y >= n:
        return
    
    # Si está en una pared, no hacer nada
    if matriz[x][y] == 1:
        return
    
    # Si llega al final, sumar al contador de soluciones.
    if x == n-1 and y == n-1:
        soluciones += 1
        return

    for (dx,dy) in [(1,0), (0,1)]: #Exploramos solo hacia abajo o la derecha.
        contar_soluciones(n,matriz, x + dx, y + dy)
    return

#Inputs
n = int(input())
matriz = list()
for i in range(n):
    renglon = input().split(" ")
    if len(renglon) != n: raise Exception(f"La longitud del renglon debe ser igual a {n}.")
    matriz.append([int(x) for x in renglon])

# Solucion
print("Output:")
contar_soluciones(n,matriz)
print(soluciones)