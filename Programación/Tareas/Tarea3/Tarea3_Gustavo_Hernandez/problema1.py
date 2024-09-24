# Verificador de expresiones matemáticas

# Solo nos preocuparemos por el formateo de los carácteres
# (), [], {}.  Por lo que no verificaremos si la expresión
# como tal tiene errores ( e.g. [1+]1 ).


def esta_formateada(cadena : str):
    # Realizaremos la verificación mediante un
    # contador para cada par de carácteres.
    contador = {"()" : 0,
                "[]" : 0,
                "{}" : 0}
    
    # Sumaremos 1 si encuentra en la cadena un parentesis
    # abierto "(" y restaremos 1 si encuentra uno cerrado ")"
    # por lo que si está mal formateado, en algún punto el resultado
    # será -1.
    for caracter in cadena:
        for parentesis in contador.keys():
            # Verificamos para cada tipo de parentesis si el caracter
            # es el parentesis abierto o cerrado
            if caracter == parentesis[0]:
                contador[parentesis] += 1
                # Guardamos el parentesis abierto
                abierto = parentesis[0]
            elif caracter == parentesis[1]:
                contador[parentesis] -= 1
                # Si el cierre no coincide con el parentesis abierto
                # más profundo, está mal formateado.
                if abierto != parentesis[0] and abierto is not None:
                    print("NO")
                    return
                # Si coincide, reiniciamos a None como valor default
                abierto = None
        
        # Verificamos si alguno es -1
        if -1 in contador.values():
            print("NO")
            return
    
    # Al terminar, todos los caracteres se tuvieron que cerrar
    # entonces todos los contadores deben ser 0
    if [0]*len(contador) != list(contador.values()):
        print("NO")
        return
    
    # Si todo salió bien
    print("YES")
    return

#inputs
cadena = input()

#output
esta_formateada(cadena)