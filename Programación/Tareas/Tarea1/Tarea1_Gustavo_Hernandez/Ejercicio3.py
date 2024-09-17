# Problema de la diferencia máxima

def mayor_diferencia():
    # Inputs
    longitud = int(input())
    arreglo = input().split(" ")
    arreglo = [int(x) for x in arreglo]

    # Realizamos la comparación buscando el mayor
    mayor = None
    for i in range(longitud):
        for j in range(longitud):
            if mayor is None or mayor < (arreglo[i]-arreglo[j]):
                mayor = arreglo[i]-arreglo[j]
    print(mayor)


mayor_diferencia()