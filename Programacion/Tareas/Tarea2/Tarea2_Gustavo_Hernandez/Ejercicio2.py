# Encontrando el objetivo.

# Utilizamos la búsqueda binaria, código rescatado de la clase.
def busquedaBinaria(vectorOrdenado, v, ini, fin):
  if fin < ini:
    return -1;
  m = (ini + fin)//2;
  if vectorOrdenado[m] == v:
    return m;
  elif v < vectorOrdenado[m]:
    return busquedaBinaria(vectorOrdenado, v, ini, m-1);
  else:
    return busquedaBinaria(vectorOrdenado, v, m + 1, fin);

# Problema 2
def encontrando_objetivo(n, lista, objetivo):
	# Iteraremos desde 0 hasta la mitad del objetivo.
	for i in range(objetivo//2+1):
		# Buscamos los indices de i y objetivo-i, que suman objetivo.
		sumando1 = busquedaBinaria(lista, i, 0, n-1)
		sumando2 = busquedaBinaria(lista, objetivo-i, 0, n-1)

		# En caso de que ambos representen al mismo elemento, descartamos la solución
		# y buscamos nuevamente en una lista que no contenga este elemento.
		# n-2 ya que quitamos un elemento.
		if sumando1 == sumando2:
			lista_auxiliar = lista.copy()
			lista_auxiliar.pop(sumando2)
			sumando2 = busquedaBinaria(lista_auxiliar, objetivo-i,0,n-2)

		# Ambos deben ser distintos de -1 para que se cumpla que hayan
		# encontrado dos números que sumen "objetivo". 
		# Si no se encuentran, se retornará False.
		if sumando1 != -1 and sumando2 != -1: return True
	return False




#Inputs
n = int(input())
lista = input().split(" ")
lista = [int(x) for x in lista]
objetivo = int(input())

#Imprimimos resultado
print("Output:")
print(encontrando_objetivo(n,lista,objetivo))