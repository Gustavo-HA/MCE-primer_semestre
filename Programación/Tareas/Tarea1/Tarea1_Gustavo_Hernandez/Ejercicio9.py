# Un problema muy X

def problema_x(n: int) -> str:
    # Caso base
    if n == 0: 
        return "X"
    
    anterior = problema_x(n-1)
    # Para el "primer nivel" del texto, está de la forma: problema_x(n-1) + " "*3**(n-1) + problema_x(n-1).
    # Este primer nivel es idéntico al tercero.
    lineas = anterior.splitlines() 
    lineas = [linea + " "*3**(n-1) + linea for linea in lineas] # "Appendizamos" los espacios y el mismo nivel anterior.
    primer_bloque = "\n".join(lineas)
    tercer_bloque = primer_bloque

    # Para el segundo nivel, tiene la forma: " "*3**(n-1) + problema_x(n-1) + " "*3**(n-1)
    lineas = anterior.splitlines()
    lineas = [" "*3**(n-1) + linea + " "*3**(n-1) for linea in lineas] # Agregamos y apendizamos los espacios
    segundo_bloque = "\n".join(lineas)

    texto = [primer_bloque, segundo_bloque, tercer_bloque]
    return "\n".join(texto)


n = int(input())
print(problema_x(n))