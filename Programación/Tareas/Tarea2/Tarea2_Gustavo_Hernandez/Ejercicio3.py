# I-esimo menor.

def iesimo_menor(n, arreglo, consultas):
    # De menor a mayor
    arreglo_ordenado = sorted(arreglo)
    for consulta in consultas:
        # Imprime el elemento que requiere cada consulta.
        print(arreglo_ordenado[consulta-1])


#Inputs
n = int(input())
arreglo = input().split(" ")
arreglo = [int(x) for x in arreglo]
m = int(input())
consultas = list()
for i in range(m):
    consultas.append(int(input()))
    # restriccion
    if consultas[-1] > n or consultas[-1] < 1: raise Exception("Cada consulta debe estar entre 1 y n.")

# Restricciones
if m > n or m < 1: raise Exception("m debe estar entre 1 y n.")



#Resultado.
print("Output:")
iesimo_menor(n,arreglo,consultas)