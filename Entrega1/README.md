**Entrega 1 - Inteligencia Artificial.**
Este trabajo fue elaborado por Esteban Molina y Miguel Villegas.

**Resumen:**
**Problema 1: 8-puzzle con algoritmos de búsqueda**
Se trató de resolver el típico 8-puzzle, en el cual el agente (la casilla vacía) debe mover las fichas hasta alcanzar un estado meta. Se analizaron los algoritmos de BFS, A* y Branch and Bound. Se concluyó que A* con heurística de distancia Manhattan es el más adecuado, ya que garantiza completitud y optimalidad con un rendimiento eficiente en tiempo y memoria. En la práctica, A* resolvió el puzzle en 22 pasos (en un caso de ejemplo), explorando 736 nodos en pocos milisegundos.


**Problema 2: Generación de horarios con Algoritmo Genético**
Se trató el problema de planificar un horario semanal con clases, estudio, gimnasio, ocio y descansos, bajo restricciones y preferencias. Se modeló cada horario como un cromosoma de 56 bloques (2h × 7 días × 8 bloques diarios). El algoritmo genético (GA) fue elegido por su capacidad para explorar grandes espacios de búsqueda y cumplir restricciones a través de una función de fitness. Se comprobó que el GA encuentra horarios balanceados y realistas, respetando actividades obligatorias y optimizando el uso del tiempo libre.
