import heapq          # Para usar una cola de prioridad (heap) eficiente.
import itertools      # Para generar un contador que sirva como tie-breaker en la frontera.
import random
from termcolor import colored

N = 3

class Tablero:
    def __init__(self):
        self.n = N * N # Tamaño del tablero, acá es 3x3
        
    # Función para crear tablero a partir del estado objetivo    
    def crear_tablero(self):
        estado = [1, 2, 3, 4, 5, 6, 7, 8, 0] # Partimos del estado objetivo 

        for i in range(1000): # Iteramos 1000 veces para desordenar el tablero (puede ser cualquier número)
            casilla_vacia = estado.index(0) # Nos da el índice del '0' en la lista
            movimientos_validos = self.get_movimientos_validos(casilla_vacia) # Obtiene una lista de movimientos válidos para '0'
            if movimientos_validos: # Si hay movimientos válidos
                movimiento = random.choice(movimientos_validos) # Escoge uno al azar
                estado = self.mover_vacia(estado, casilla_vacia, movimiento) # Actualiza el estado del tablero después de cada movimiento
        return estado
    
    # Función para obtener los movimientos válidos que tiene la casilla vacía desde una posición determinada
    def get_movimientos_validos(self, casilla_vacia):
        movimientos = []
        if casilla_vacia >= 3:  # Si el indice de la casilla vacía es mayor o igual a 3, puede moverse arriba
            movimientos.append('Arriba')
        if casilla_vacia <= 5:  # Si el indice de la casilla vacía es menor o igual a 5, puede moverse abajo
            movimientos.append('Abajo')
        if casilla_vacia % 3 != 0:  # Si el indice de la casilla vacía / 3 tiene residuo diferente a 0, puede moverse izquierda
            movimientos.append('Izquierda')
        if casilla_vacia % 3 != 2:  # Si el indice de la casilla vacía / 3 tiene residuo diferente a 2, puede moverse derecha
            movimientos.append('Derecha')
        return movimientos
    
    # Función para mover la casilla vacía
    def mover_vacia(self, estado, casilla_vacia, direccion):
        nuevo_estado = estado[:] # Creamos copia del estado actual
        movimientos = {'Arriba': -3, 'Abajo': 3, 'Izquierda': -1, 'Derecha': 1}
        nueva_pos = casilla_vacia + movimientos[direccion] # Coge la casilla vacia y le suma el desplazamiento del movimiento a hacer
        nuevo_estado[casilla_vacia], nuevo_estado[nueva_pos] = nuevo_estado[nueva_pos], nuevo_estado[casilla_vacia] # Simplemente cambia las casillas de posición
        return nuevo_estado
    
    # Función para mostrar el tablero
    def mostrar_tablero(self, tablero):
        print("+---+---+---+")
        for col in range(0, 9, 3):
            palo = "|"
            for casilla in tablero[col:col + 3]:
                if casilla == 0:  
                    palo += f" {colored(' ', 'yellow')} |"
                else:
                    palo += f" {colored(str(casilla), 'green')} |"
            print(palo)
            print("+---+---+---+")         
            

class Nodo:
    def __init__(self, state, parent=None, action=None, g=0, h=0):
        self.state = state        # Estado actual que representa el nodo.
        self.parent = parent      # Nodo padre (para reconstruir la ruta).
        self.action = action      # Acción tomada desde el padre para llegar aquí.
        self.g = g                # Costo acumulado desde el nodo inicial hasta este nodo.
        self.h = h                # Heurística estimada desde este nodo hasta el objetivo.

    @property
    def f(self):
        # f = g + h, la función de evaluación en A*.
        return self.g + self.h

    def __lt__(self, other):
        # Para que heapq pueda comparar nodos si hace falta (aunque usamos tie-breaker).
        return self.f < other.f


def a_star(problem):
    # Crea el nodo inicial con g=0 y h calculada.
    start = Nodo(state=problem.inicial, g=0, h=problem.h(problem.inicial))
    frontier = []  # frontera será un heap de tuplas (f, contador, nodo)
    counter = itertools.count()  # contador para tie-breaker en caso de empates en f

    # Inserta el nodo inicial en la frontera.
    heapq.heappush(frontier, (start.f, next(counter), start))
    reached = {problem.inicial: start}  # diccionario de mejores nodos conocidos por estado

    # Bucle principal de A*
    while frontier:
        _, _, current = heapq.heappop(frontier)  # extrae el nodo con menor f

        # Si el nodo actual es meta, lo devolvemos (ruta encontrada).
        if problem.is_objetivo(current.state):
            return current

        # Expandir sucesores
        for action in problem.acciones(current.state):
            child_state = problem.resultado(current.state, action)  # estado resultadoante
            tentative_g = current.g + problem.action_cost(current.state, action, child_state)  # costo acumulado

            # Si no se ha alcanzado ese estado antes o se encuentra un mejor camino (menor g)
            if (child_state not in reached) or (tentative_g < reached[child_state].g):
                # Crear el nodo hijo con su g y h
                child = Nodo(
                    state=child_state,
                    parent=current,
                    action=action,
                    g=tentative_g,
                    h=problem.h(child_state)
                )
                reached[child_state] = child  # actualizar mejor nodo para ese estado
                # Añadir a la frontera con tie-breaker para mantener orden estable
                heapq.heappush(frontier, (child.f, next(counter), child))

    # Si se vacía la frontera sin encontrar meta, no hay solución.
    return None


class Problema_8Puzzle:
    def __init__(self, inicial, objetivo):
        self.inicial = tuple(inicial) # Estado inicial 
        self.objetivo = tuple(objetivo) # Estado objetivo

    def acciones(self, state):
        lista_estados = list(state)
        casilla_vacia = lista_estados.index(0)
        acciones = []
        
        if casilla_vacia >= 3:  
            acciones.append('Arriba')
        if casilla_vacia <= 5:  
            acciones.append('Abajo')
        if casilla_vacia % 3 != 0:  
            acciones.append('Izquierda')
        if casilla_vacia % 3 != 2:  
            acciones.append('Derecha')
        
        return acciones

    def resultado(self, state, action):
        lista_estados = list(state)
        casilla_vacia = lista_estados.index(0)
        movimientos = {'Arriba': -3, 'Abajo': 3, 'Izquierda': -1, 'Derecha': 1}
        nueva_pos = casilla_vacia + movimientos[action]
        
        lista_estados[casilla_vacia], lista_estados[nueva_pos] = lista_estados[nueva_pos], lista_estados[casilla_vacia]
        return tuple(lista_estados)

    def action_cost(self, state, action, resultado_state):
        return 1

    def is_objetivo(self, state):
        return state == self.objetivo

    def h(self, state):
        distancia = 0
        for i in range(9): # Iteramos cada casilla del tablero
            if state[i] != 0: # Ignoramos la casilla vacía
                x1, y1 = divmod(i, 3) # Posición actual, por ejemplo divmod(4, 3) = (1, 1); x1 es resultado division y y1 es el residuo
                x2, y2 = divmod(state[i] - 1, 3) # Posición objetivo
                distancia += abs(x1 - x2) + abs(y1 - y2)
        return distancia


# Creamos el tablero y determinamos el estado objetivo
tablero_obj = Tablero()
estado_inicial = tablero_obj.crear_tablero()
estado_objetivo = [1, 2, 3, 4, 5, 6, 7, 8, 0]

print(colored("\nEstado inicial del 8-puzzle:", "blue"))
tablero_obj.mostrar_tablero(estado_inicial)
print()

# Creamos una instancia del problema
problema_8_puzzle = Problema_8Puzzle(
    inicial=estado_inicial,
    objetivo=estado_objetivo
)

# Ejecutamos A* sobre el problema 
print(colored("Resolviendo con A*...", "yellow"))
solucion = a_star(problema_8_puzzle)

# Función para mostrar la solución paso a paso
def mostrar_solucion(solucion, tablero_obj):
    if not solucion:
        print(colored("No se encontró solución.", "red"))
        return
    
    camino = []
    Nodo = solucion
    while Nodo:
        camino.append((Nodo.state, Nodo.action))
        Nodo = Nodo.parent
    camino.reverse()
    
    print(colored(f"Solución encontrada en {len(camino)-1} pasos:", "green"))
    print(colored(f"Costo: {solucion.g}", "green"))
    print()
    
    for i, (state, action) in enumerate(camino):
        if action:
            print(colored(f"Paso {i}: Movimiento {action}", "blue"))
        else:
            print(colored("Estado inicial:", "blue"))
        tablero_obj.mostrar_tablero(list(state))
        print()

# Función para mostrar solo el path de acciones
def mostrar_path_acciones(solucion):
    if not solucion:
        print(colored("No se encontró solución.", "red"))
        return []
    
    # Reconstruir el camino de acciones
    acciones = []
    nodo = solucion
    while nodo and nodo.parent:  # Mientras tenga padre (no sea el nodo inicial)
        acciones.append(nodo.action)
        nodo = nodo.parent
    
    # Invertir para tener el orden correcto
    acciones.reverse()
    
    print(colored(f"\nPath de acciones para llegar al objetivo:", "yellow"))
    print(colored("=" * 50, "yellow"))
    print()

    if acciones:
        # Mostrar como lista
        print(colored(f"Secuencia: {acciones}", "blue"))
        
        # Mostrar como string continuo
        path_string = " → ".join(acciones)
        print(colored(f"\nCamino: {path_string}", "green"))
    
    return acciones

mostrar_solucion(solucion, tablero_obj)

# Mostrar el path de acciones
path_acciones = mostrar_path_acciones(solucion)
