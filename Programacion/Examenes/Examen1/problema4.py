# El guardian del arbol genealogico

# Tenemos los nodo y arbol
class Nodo:
    def __init__(self, dato = 0):
        # "dato" puede ser de cualquier tipo, incluso un objeto si se sobrescriben los operadores de comparación
        self.dato = dato
        self.izquierda = None
        self.derecha = None

class Arbol:
    # Funciones privadas
    def __init__(self, dato = 0):
        self.raiz = Nodo(dato)

    def __agregar_recursivo(self, nodo, dato):
        if dato < nodo.dato:
            if nodo.izquierda is None:
                nodo.izquierda = Nodo(dato)
            else:
                self.__agregar_recursivo(nodo.izquierda, dato)
        else:
            if nodo.derecha is None:
                nodo.derecha = Nodo(dato)
            else:
                self.__agregar_recursivo(nodo.derecha, dato)
    
    def agregar(self, dato):
        self.__agregar_recursivo(self.raiz, dato)
    
    def sumaro(self, dato):
        pass
    
#Input
N = int(input())
V = list(map(int, input().split(" ")))
assert len(V) == N
arbol = V[0]
for i in range(1,len(V)):
    arbol.agregar(i)

Q = int(input())
consultas = list()
for _ in range(Q):
    consultas.append(int(input()))

