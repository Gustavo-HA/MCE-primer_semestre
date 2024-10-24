# Vamonos de viaje







#Input
N, M = map(int,input().split(" "))
amigos = list(range(1,N+1))

relaciones = dict()

for _ in range(M):
    # Grafo de amigos
    amigo1, amigo2 = map(int, input().split(" "))
    if amigo1 not in relaciones:
        relaciones[amigo1] = set()
    if amigo2 not in relaciones:
        relaciones[amigo2] = set()
    relaciones[amigo1].add(amigo2)
    relaciones[amigo2].add(amigo1)

# amigos de amigos
for persona in relaciones.keys():
    amigos_actualies = list(relaciones[persona])
    for amigo in amigos_actualies:
        relaciones[persona].update(relaciones[amigo])
        relaciones[amigo].update(relaciones[persona])

#Output

auxiliar = set()
contador_de_habitaciones = 0

for persona in relaciones.keys():
    if relaciones[persona].isdisjoint(auxiliar):
        auxiliar.update(relaciones[persona])
        contador_de_habitaciones += 1
        
print(contador_de_habitaciones)