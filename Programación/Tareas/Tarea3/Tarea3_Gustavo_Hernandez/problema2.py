# Estamos perdidos

# El programa buscará todos los caminos posibles y se agregarán
# a un diccionario como key y el valor será su longitud.
caminos = dict()

def camino_mas_corto(m,n,matriz, x, y , final, recorrido=list()):
    global caminos

    # Si sale de la matriz, no hacer nada 
    if x < 0 or x >= m or y < 0 or y >= n:
        return
    
    # Si está en un obstaculo, no hacer nada
    if matriz[x][y] == "x":
        return

    # Si llega al final
    if (x, y) == final:
        caminos[str(recorrido)] = len(recorrido)
        return

    matriz[x][y] = "x" # Marcamos ya visitada.

    for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]: #Exploramos hacia los 4 lados.
        camino_mas_corto(m,n,matriz, x + dx, y + dy, final, recorrido+[(x+dx,y+dy)])



#Inputs
print("Input:")
m, n = input().split(" ")
m, n = int(m), int(n)
matriz = list()
for i in range(m):
    renglon = input()
    if len(renglon) != n: raise Exception(f"La fila debe ser tamaño {n}.")
    # Guardamos la matriz 
    matriz.append(list(renglon.replace("S","*").replace("F","*")))
    # Guardamos el inicio
    if "S" in renglon:
        x, y = (i,renglon.index("S"))
    # Guardamos el final
    if "F" in renglon:
        final = tuple((i,renglon.index("F")))

# Solución
print("----Output----")
camino_mas_corto(m, n, matriz, x = x, y = y, final=final)
if len(caminos) == 0: # Si no hay solucion, imprime None
    print("None")
    exit()

min_pasos = min(caminos.values()) # Menor longitud entre todos los caminos
camino_chido = [camino for camino in caminos if caminos[camino] == min_pasos] # Caminos con menor longitud.
# Si son varios caminos los que cumplen con la longitud, se imprime el primero.
print(camino_chido[0][1:-1])