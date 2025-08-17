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
        if casilla_vacia >= 3:  # Puede moverse arriba
            movimientos.append('U')
        if casilla_vacia <= 5:  # Puede moverse abajo
            movimientos.append('D')
        if casilla_vacia % 3 != 0:  # Puede moverse izquierda
            movimientos.append('L')
        if casilla_vacia % 3 != 2:  # Puede moverse derecha
            movimientos.append('R')
        return movimientos
    
    def mover_vacia(self, estado, casilla_vacia, direccion):
        """Mueve el espacio en blanco en la dirección especificada"""
        nuevo_estado = estado[:] # Creamos copia del estado actual
        movimientos = {'U': -3, 'D': 3, 'L': -1, 'R': 1}
        nueva_pos = casilla_vacia + movimientos[direccion]
        nuevo_estado[casilla_vacia], nuevo_estado[nueva_pos] = nuevo_estado[nueva_pos], nuevo_estado[casilla_vacia]
        return nuevo_estado
    
    def mostrar_tablero(self, tablero):
        print("+---+---+---+")
        for col in range(0, 9, 3):
            visual_col = "|"
            for casilla in tablero[col:col + 3]:
                if casilla == 0:  # Blank tile
                    visual_col += f" {colored(' ', 'cyan')} |"
                else:
                    visual_col += f" {colored(str(casilla), 'yellow')} |"
            print(visual_col)
            print("+---+---+---+")         
            

# Clase que representa un nodo en el espacio de búsqueda.
class Node:
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
    start = Node(state=problem.initial, g=0, h=problem.h(problem.initial))
    frontier = []  # frontera será un heap de tuplas (f, contador, nodo)
    counter = itertools.count()  # contador para tie-breaker en caso de empates en f

    # Inserta el nodo inicial en la frontera.
    heapq.heappush(frontier, (start.f, next(counter), start))
    reached = {problem.initial: start}  # diccionario de mejores nodos conocidos por estado

    # Bucle principal de A*
    while frontier:
        _, _, current = heapq.heappop(frontier)  # extrae el nodo con menor f

        # Si el nodo actual es meta, lo devolvemos (ruta encontrada).
        if problem.is_goal(current.state):
            return current

        # Expandir sucesores
        for action in problem.actions(current.state):
            child_state = problem.result(current.state, action)  # estado resultante
            tentative_g = current.g + problem.action_cost(current.state, action, child_state)  # costo acumulado

            # Si no se ha alcanzado ese estado antes o se encuentra un mejor camino (menor g)
            if (child_state not in reached) or (tentative_g < reached[child_state].g):
                # Crear el nodo hijo con su g y h
                child = Node(
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


# Clase que encapsula el problema de búsqueda para el 8-puzzle.
class PuzzleProblem:
    def __init__(self, initial, goal):
        self.initial = tuple(initial)  # Estado inicial del problema (como tupla para ser hasheable).
        self.goal = tuple(goal)        # Estado objetivo.

    def actions(self, state):
        """Devuelve las acciones posibles desde un estado: U, D, L, R"""
        state_list = list(state)
        casilla_vacia = state_list.index(0)
        acciones = []
        
        if casilla_vacia >= 3:  # Puede moverse arriba
            acciones.append('U')
        if casilla_vacia <= 5:  # Puede moverse abajo
            acciones.append('D')
        if casilla_vacia % 3 != 0:  # Puede moverse izquierda
            acciones.append('L')
        if casilla_vacia % 3 != 2:  # Puede moverse derecha
            acciones.append('R')
        
        return acciones

    def result(self, state, action):
        """Devuelve el estado resultante de aplicar una acción"""
        state_list = list(state)
        casilla_vacia = state_list.index(0)
        movimientos = {'U': -3, 'D': 3, 'L': -1, 'R': 1}
        nueva_pos = casilla_vacia + movimientos[action]
        
        # Intercambiar el espacio en blanco con la casilla destino
        state_list[casilla_vacia], state_list[nueva_pos] = state_list[nueva_pos], state_list[casilla_vacia]
        return tuple(state_list)

    def action_cost(self, state, action, result_state):
        """Costo uniforme de 1 para cada movimiento"""
        return 1

    def is_goal(self, state):
        """Comprueba si el estado actual es el objetivo."""
        return state == self.goal

    def h(self, state):
        """Heurística de distancia Manhattan"""
        distance = 0
        for i in range(9):
            if state[i] != 0:
                # Posición actual
                x1, y1 = divmod(i, 3)
                # Posición objetivo
                x2, y2 = divmod(state[i] - 1, 3)
                distance += abs(x1 - x2) + abs(y1 - y2)
        return distance


# Crear una instancia del tablero y generar estado inicial
tablero_obj = Tablero()
estado_inicial = tablero_obj.crear_tablero()
estado_objetivo = [1, 2, 3, 4, 5, 6, 7, 8, 0]

print(colored("Estado inicial del 8-puzzle:", "blue"))
tablero_obj.mostrar_tablero(estado_inicial)
print()

# Instancia del problema de 8-puzzle
problema_8_puzzle = PuzzleProblem(
    initial=estado_inicial,
    goal=estado_objetivo
)

# Ejecuta A* sobre el problema definido.
print(colored("Resolviendo con A*...", "yellow"))
solution = a_star(problema_8_puzzle)

# Función para mostrar la solución paso a paso
def mostrar_solucion(solution, tablero_obj):
    if not solution:
        print(colored("No se encontró solución.", "red"))
        return
    
    path = []
    node = solution
    while node:
        path.append((node.state, node.action))
        node = node.parent
    path.reverse()
    
    print(colored(f"Solución encontrada en {len(path)-1} pasos:", "green"))
    print(colored(f"Costo total: {solution.g}", "green"))
    print()
    
    for i, (state, action) in enumerate(path):
        if action:
            print(colored(f"Paso {i}: Movimiento {action}", "cyan"))
        else:
            print(colored("Estado inicial:", "cyan"))
        tablero_obj.mostrar_tablero(list(state))
        print()

# Mostrar la solución
mostrar_solucion(solution, tablero_obj)
