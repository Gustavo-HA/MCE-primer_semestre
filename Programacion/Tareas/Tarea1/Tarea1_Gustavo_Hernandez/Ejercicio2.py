# Creando matrices 

def creando_matrices():
    # Debido a que el problema no pide explicitamente que se guarden en memoria
    # las dos matrices, simplemente haremos que se impriman en la terminal.
    n = int(input())


    # Para la matriz A
    for i in range(n):
        for j in range(n):
            if i == j:
                print(2, end=" ")
            else:
                print(1, end=" ")
        print("")

    print("")

    # Para la matriz B
    for i in range(n):
        for j in range(n):
            if i == j:
                print(3, end=" ")
            else:
                print(1, end=" ")
        print("")
    
creando_matrices()
