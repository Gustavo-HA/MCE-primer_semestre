# Calculando la distancia

def calculando_distancia():
    #Inputs
    N = int(input())
    ubicaciones = list()
    for i in range(N):
        nueva_ubicacion = tuple(input().split(" ")) # Leemos coordenadas
        ubicaciones.append(nueva_ubicacion)
    ubicaciones = [(int(x),int(y)) for x,y in ubicaciones]

    # Calculamos distancias y las sumamos
    distancia = 0
    inicio = (0,0)
    pos_actual = inicio
    for ubicacion in ubicaciones:
        # Sobre las entregas
        distancia += ((ubicacion[0]-pos_actual[0])**2+(ubicacion[1]-pos_actual[1])**2)**0.5 
        pos_actual = ubicacion
    # Regresando al inicio
    distancia += ((inicio[0]-pos_actual[0])**2+(inicio[1]-pos_actual[1])**2)**0.5 
    
    #Imprimimos
    print(round(distancia,2))

calculando_distancia()