# Subcadenas palindromicas.

# Iteraremos sobre cada caracter de la cadena y verificaremos si 
# cada uno de los substrings que se pueden armar a partir de ese caracter
# son palindromos para agregarlos a un set de palabras palindromas.


def subcadenas_palindromicas(cadena):
    # Guardaremos en un set las subcadenas palindromicas,
    # esto nos garantizará no duplicidad.
    sub_palindromica = set()
    
    for i in range(len(cadena)):
        #Agregamos caracter individual
        subcadena = cadena[i]
        sub_palindromica.add(subcadena)

        # Agregaremos substrings si son palindromos.
        for j in range(i+1,len(cadena)):
            subcadena += cadena[j]
            if subcadena == subcadena[::-1]:
                sub_palindromica.add(subcadena)

    # Retornamos el número de subcadenas palindromicas
    return(len(sub_palindromica))


#Inputs
cadena = input()
cadena = "lalalalala"

#Output
print("Output:")
print(subcadenas_palindromicas(cadena))