# Secuencia infinita potencias de 10
from math import log10, ceil


#Input
N = int(input())

potencia_10 = ceil(log10(2**64)) #Hasta qué potencia de 10 es necesario saber el cadena?

cadena = ""
for i in range(potencia_10+1):
    cadena += str(10**i)
    
#Output
print(cadena[N])