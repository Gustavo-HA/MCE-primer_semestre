# Goldbach mejorado

# Al igual que el problema anterior, haremos una función que verifique si un número es primo.

def es_primo(n: int):
    # Verifica si un numero es primo.
    for i in range(2,n//2 + 1):
        if n % i == 0:
            return False
    return True



def goldbach_mejorado():
    # Inputs
    N = int(input())
    if N < 4 or N > 10**6: raise Exception("Error: 4 <= N <= 10^6")
    if N % 2 != 0: raise Exception("N tiene que ser par")

    for numero in range(2,N//2+1):
        # Verificamos para cada número hasta N//2 si es primo
        # y si lo que le falta para sumar a N también es primo
        if es_primo(numero) and es_primo(N-numero): print(numero,N-numero)
    

goldbach_mejorado()