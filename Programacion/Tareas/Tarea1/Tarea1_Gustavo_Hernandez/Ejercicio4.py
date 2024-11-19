# Problema eliminador de villanos.


def eliminador_villanos():
    #Inputs 
    N, K = input().split(" ")
    N = int(N)
    K = int(K)

    villanos = [i+1 for i in range(N)]

    pos_inicial = K-1 # Indices iniciando en 0
    while (len(villanos) != 1):
        posicion_actual = (K+pos_inicial) % len(villanos) # Así, no nos saldremos de los límites
        villanos.pop(posicion_actual)
        pos_inicial = posicion_actual

    # Imprimimos el último villano que queda.
    print(villanos[0]) 



eliminador_villanos()