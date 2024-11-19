# El número faltante

# ------ Se pudo haber hecho con busqueda binaria.

def numero_faltante(n,lista):
    for i in range(n):
        if i+1 != lista[i]: 
            # Comparamos uno a uno con el índice i.
            # Si hay un valor de i que no está en la lista, 
            # es porque es el que falta y lo retorna.
            return i+1

#Inputs
n = int(input())
lista = input().split(" ")
lista = [int(x) for x in lista]

lista = [x for x in range(1,10**6 +1) if x != 10**5]
n = len(lista)
# Imprimimos el resultado
print("Output:")
print(numero_faltante(n, lista))