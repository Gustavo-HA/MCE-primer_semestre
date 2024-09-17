def es_primo(n: int):
    # Verifica si un numero es primo.
    for i in range(2,n//2 + 1):
        if n & i == 0:
            return False
    return True

# Para cada N_i encontrar pares primos gemelos.

def primos_gemelos(n: int):
    # Busca los pares primos consecutivos hasta n.
    for k in range(2,(n+1)-2):
        if (es_primo(k) and es_primo(k+2)):
            print(f"{k}, {k+2}")
    
# Ahora lo hacemos de forma general para el problema.

def problema6():
    # Inputs
    M = int(input())
    consultas = list()
    for i in range(M):
        consultas.append(int(input()))
    print(consultas)

    # Realizamos las consultas de primos gemelos.
    for consulta in consultas:
        primos_gemelos(consulta)
        print("")
