# Palindromos mágicos.

# Dado un n, buscar el palindromo más pequeño que es divisible por n.

# Haremos una función que nos diga si un número es palíndromo.
def es_palindromo(n):
    n_cadena = str(n)
    if n_cadena == n_cadena[::-1]:
        return True
    return False

def palindromo_magico(n):
    # Retorna el número palindromo mínimo divisible por n.
    # Retorna -1 en caso de que no lo haya hasta 10**7-1
    numero_propuesto = n
    while(numero_propuesto < 10**7 - 1):
        # Verificaremos para cada multiplo de n si es palindromo o no.
        if es_palindromo(numero_propuesto): return numero_propuesto
        numero_propuesto += n
    return -1


#inputs
n = int(input())

#Outputs:
print("Output:")
print(palindromo_magico(n))