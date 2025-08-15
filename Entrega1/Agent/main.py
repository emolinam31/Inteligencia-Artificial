import heapq          # Para usar una cola de prioridad (heap) eficiente.
import itertools      # Para generar un contador que sirva como tie-breaker en la frontera.

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


# Clase que encapsula el problema de búsqueda.
class Problem:
    def __init__(self, initial, goal, adjacency, h_func):
        self.initial = initial      # Estado inicial del problema.
        self.goal = goal            # Estado objetivo.
        self._adj = adjacency       # Grafo: diccionario de vecinos con costos.
        self._h = h_func            # Función heurística.

    def actions(self, state):
        # Devuelve las acciones posibles desde un estado: los nombres de los nodos vecinos.
        return list(self._adj.get(state, {}).keys())

    def result(self, state, action):
        # En este modelado, la acción es directamente el nombre del siguiente estado.
        return action

    def action_cost(self, state, action, result_state):
        # Costo de ir de state a result_state por medio de action.
        return self._adj.get(state, {}).get(action, float('inf'))

    def is_goal(self, state):
        # Comprueba si el estado actual es el objetivo.
        return state == self.goal

    def h(self, state):
        # Aplica la heurística sobre un estado.
        return self._h(state)


# Definición del grafo de Rumania con costos entre ciudades.
cost = 1

action = {
    '1': {'2': cost, '4': cost},
    '2': {'1': cost, '5': cost, '3': cost},
    '3': {'2': cost, '6': cost},
    '4': {'1': cost, '7': cost, '5': cost},
    '5': {'2': cost, '4': cost, '6': cost, '8': cost},
    '6': {'3': cost, '5': cost, '9': cost},
    '7': {'4': cost, '8': cost},
    '8': {'7': cost, '5': cost, '9': cost},
    '9': {'8': cost, '6': cost},
}

# Valores heurísticos (distancia en línea recta estimada a Bucharest).
h_values = {
    'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Drobeta': 242, 'Eforie': 161,
    'Fagaras': 176, 'Giurgiu': 77, 'Hirsova': 151, 'Iasi': 226, 'Lugoj': 244,
    'Mehadia': 241, 'Neamt': 234, 'Oradea': 380, 'Pitesti': 100, 'Rimnicu Vilcea': 193,
    'Sibiu': 253, 'Timisoara': 329, 'Urziceni': 80, 'Vaslui': 199, 'Zerind': 374
}

# Instancia del problema con el estado inicial, objetivo, grafo y heurística.
romania_problem = Problem(
    initial='Arad',
    goal='Bucharest',
    adjacency=action,
    h_func=lambda s: h_values[s]
)

# Ejecuta A* sobre el problema definido.
solution = a_star(romania_problem)

# Reconstruye la ruta desde el nodo meta hasta el inicial siguiendo padres.
if solution:
    path = []
    node = solution
    while node:
        path.append(node.state)
        node = node.parent
    path.reverse()  # invertir para que vaya de inicial a objetivo
    print("\nSolution path:", path)
    print("Total cost g:", solution.g, "\n")
else:
    print("No solution found")
