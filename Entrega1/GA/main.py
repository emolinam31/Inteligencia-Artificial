# Entrega 1 IA - Algoritmo Genético para Horarios
# Miguel Villegas y Esteban Molina

import random
from collections import Counter

# Configuración del horario
dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
horas_inicio = [6, 8, 10, 12, 14, 16, 18, 20]  # bloques de 2 horas
duracion_bloque = 2

# Cantidad de bloques totales
n_dias = len(dias)
n_bloques = len(horas_inicio)
total_bloques = n_dias * n_bloques  # 56 bloques

# Objetivos de actividades
objetivo_clases = 25
objetivo_gym = 5
objetivo_estudio = 7
min_libre_por_dia = 1

# Tipos de actividades
CLASE = "CLASE"
GYM = "GYM"
ESTUDIO = "ESTUDIO"
OCIO = "OCIO"
LIBRE = "LIBRE"

# Actividades que pueden cambiar (no las clases fijas)
actividades_mutables = [GYM, ESTUDIO, OCIO, LIBRE]

clases_fijas_usuario = None  # se puede cambiar esto por tu lista real

# Para reproducibilidad
random.seed(42)

# Mapeos útiles
dia_a_indice = {dia: i for i, dia in enumerate(dias)}
hora_a_indice = {hora: i for i, hora in enumerate(horas_inicio)}

def posicion(dia_idx, bloque_idx):
    "Convierte (día, bloque) a posición en el genoma"
    return dia_idx * n_bloques + bloque_idx

def generar_clases_ejemplo(n=25):
    "Genera 25 clases distribuidas en la semana (ejemplo)"
    random.seed(42)
    # Distribución por día: 4,4,4,4,3,3,3 = 25 total
    distribucion = [4, 4, 4, 4, 3, 3, 3]
    clases = []
    
    for dia_idx, cantidad in enumerate(distribucion):
        # Elegir bloques aleatorios para este día
        bloques_disponibles = list(range(n_bloques))
        random.shuffle(bloques_disponibles)
        
        for i in range(cantidad):
            bloque = bloques_disponibles[i]
            clases.append((dias[dia_idx], horas_inicio[bloque]))
    
    return clases

def obtener_clases_fijas():
    "Obtiene las clases fijas (del usuario o ejemplo)"
    if clases_fijas_usuario is None:
        clases = generar_clases_ejemplo(objetivo_clases)
    else:
        clases = clases_fijas_usuario
        if len(clases) != objetivo_clases:
            raise ValueError(f"Debes definir {objetivo_clases} clases, tienes {len(clases)}")
    
    # Convertir a diccionario de posiciones
    fijas = {}
    for dia, hora in clases:
        if dia not in dia_a_indice or hora not in hora_a_indice:
            raise ValueError(f"Clase inválida: {dia} {hora}")
        
        dia_idx = dia_a_indice[dia]
        bloque_idx = hora_a_indice[hora]
        fijas[posicion(dia_idx, bloque_idx)] = CLASE
    
    return fijas

# Obtener clases fijas
clases_fijas = obtener_clases_fijas()

def crear_genoma_vacio():
    "Crea un genoma vacío con clases fijas"
    genoma = [LIBRE] * total_bloques
    for pos in clases_fijas:
        genoma[pos] = CLASE
    return genoma

def crear_individuo():
    "Crea un individuo válido para el horario"
    genoma = crear_genoma_vacio()
    
    # 1. Reservar 1 LIBRE por día
    for dia_idx in range(n_dias):
        bloques_del_dia = [posicion(dia_idx, b) for b in range(n_bloques)]
        libres_disponibles = [pos for pos in bloques_del_dia if genoma[pos] == LIBRE]
        
        if libres_disponibles:
            # Preferir horas centrales para descanso
            libres_disponibles.sort(key=lambda pos: abs(horas_inicio[pos % n_bloques] - 14))
            genoma[libres_disponibles[0]] = LIBRE
    
    # 2. Colocar ESTUDIO (7 bloques) - preferir contiguos
    estudio_restante = objetivo_estudio
    intentos = 0
    
    while estudio_restante > 0 and intentos < 1000:
        dia_idx = random.randrange(n_dias)
        
        # Intentar poner 2 bloques seguidos
        for bloque_idx in range(n_bloques - 1):
            pos1 = posicion(dia_idx, bloque_idx)
            pos2 = posicion(dia_idx, bloque_idx + 1)
            
            if (genoma[pos1] == LIBRE and genoma[pos2] == LIBRE and 
                pos1 not in clases_fijas and pos2 not in clases_fijas):
                
                genoma[pos1] = ESTUDIO
                estudio_restante -= 1
                
                if estudio_restante > 0:
                    genoma[pos2] = ESTUDIO
                    estudio_restante -= 1
                break
        else:
            # Si no hay pares, poner individual
            libres = [posicion(dia_idx, b) for b in range(n_bloques) 
                     if genoma[posicion(dia_idx, b)] == LIBRE and posicion(dia_idx, b) not in clases_fijas]
            if libres:
                genoma[random.choice(libres)] = ESTUDIO
                estudio_restante -= 1
        
        intentos += 1
    
    # 3. Colocar GYM (5 bloques) - preferir tardes
    gym_restante = objetivo_gym
    intentos = 0
    
    while gym_restante > 0 and intentos < 1000:
        dia_idx = random.randrange(n_dias)
        
        # Preferir bloques de tarde (16:00 en adelante)
        bloques_tarde = [b for b, h in enumerate(horas_inicio) if h >= 16]
        random.shuffle(bloques_tarde)
        
        colocado = False
        for bloque_idx in bloques_tarde + list(range(n_bloques)):
            pos = posicion(dia_idx, bloque_idx)
            if genoma[pos] == LIBRE and pos not in clases_fijas:
                genoma[pos] = GYM
                gym_restante -= 1
                colocado = True
                break
        
        intentos += 1
    
    # 4. Rellenar con OCIO
    for pos in range(total_bloques):
        if pos in clases_fijas:
            genoma[pos] = CLASE
        elif genoma[pos] not in [CLASE, ESTUDIO, GYM, LIBRE]:
            genoma[pos] = OCIO
    
    return genoma

def crear_poblacion(tamano=60):
    "Crea una población inicial"
    return [crear_individuo() for _ in range(tamano)]

def fitness(genoma):
    "Calcula el fitness del individuo (mayor = mejor)"
    penalizacion = 0
    recompensa = 0
    
    # Penalizaciones duras
    # 1. Clases fijas respetadas
    for pos, actividad in clases_fijas.items():
        if genoma[pos] != CLASE:
            penalizacion += 500
    
    # 2. Conteos exactos
    conteo = Counter(genoma)
    penalizacion += abs(conteo[CLASE] - objetivo_clases) * 100
    penalizacion += abs(conteo[GYM] - objetivo_gym) * 30
    penalizacion += abs(conteo[ESTUDIO] - objetivo_estudio) * 30
    
    # 3. Al menos 1 LIBRE por día
    for dia_idx in range(n_dias):
        libres_del_dia = sum(1 for b in range(n_bloques) 
                           if genoma[posicion(dia_idx, b)] == LIBRE)
        if libres_del_dia < min_libre_por_dia:
            penalizacion += 40
    
    # Penalizaciones blandas
    # 1. GYM en días consecutivos
    dias_con_gym = []
    for dia_idx in range(n_dias):
        tiene_gym = any(genoma[posicion(dia_idx, b)] == GYM for b in range(n_bloques))
        if tiene_gym:
            dias_con_gym.append(dia_idx)
    
    for i in range(len(dias_con_gym) - 1):
        if dias_con_gym[i+1] - dias_con_gym[i] == 1:
            penalizacion += 5
    
    # 2. Recompensar estudio contiguo
    for dia_idx in range(n_dias):
        for bloque_idx in range(n_bloques - 1):
            pos1 = posicion(dia_idx, bloque_idx)
            pos2 = posicion(dia_idx, bloque_idx + 1)
            if genoma[pos1] == ESTUDIO and genoma[pos2] == ESTUDIO:
                recompensa += 3
    
    # 3. Penalizar huecos LIBRE entre actividades
    for dia_idx in range(n_dias):
        for bloque_idx in range(1, n_bloques - 1):
            pos_anterior = posicion(dia_idx, bloque_idx - 1)
            pos_actual = posicion(dia_idx, bloque_idx)
            pos_siguiente = posicion(dia_idx, bloque_idx + 1)
            
            if (genoma[pos_actual] == LIBRE and 
                genoma[pos_anterior] != LIBRE and 
                genoma[pos_siguiente] != LIBRE):
                penalizacion += 2
    
    # 4. Recompensar ocio en fin de semana
    ocio_finde = sum(1 for dia_idx in [5, 6] for b in range(n_bloques) 
                    if genoma[posicion(dia_idx, b)] == OCIO)
    ocio_semana = sum(1 for dia_idx in [0, 1, 2, 3, 4] for b in range(n_bloques) 
                     if genoma[posicion(dia_idx, b)] == OCIO)
    
    if ocio_finde + ocio_semana > 0:
        recompensa += int(10 * ocio_finde / (ocio_finde + ocio_semana))
    
    return 1000 - penalizacion + recompensa

def seleccion(poblacion):
    "Selección por torneo de 3"
    a, b, c = random.sample(poblacion, 3)
    return max([a, b, c], key=fitness)

def reparar(genoma):
    "Repara un genoma después de cruce/mutación"
    g = genoma[:]
    
    # Fijar clases
    for pos in clases_fijas:
        g[pos] = CLASE
    
    # Ajustar conteos
    conteo = Counter(g)
    
    # Quitar exceso
    exceso_gym = max(0, conteo[GYM] - objetivo_gym)
    exceso_estudio = max(0, conteo[ESTUDIO] - objetivo_estudio)
    
    if exceso_gym:
        posiciones_gym = [i for i, v in enumerate(g) if v == GYM and i not in clases_fijas]
        random.shuffle(posiciones_gym)
        for pos in posiciones_gym[:exceso_gym]:
            g[pos] = OCIO
    
    if exceso_estudio:
        posiciones_estudio = [i for i, v in enumerate(g) if v == ESTUDIO and i not in clases_fijas]
        random.shuffle(posiciones_estudio)
        for pos in posiciones_estudio[:exceso_estudio]:
            g[pos] = OCIO
    
    # Completar faltantes
    def completar_actividad(actividad, objetivo):
        faltan = objetivo - Counter(g)[actividad]
        if faltan <= 0:
            return
        
        # Usar OCIO primero
        pool = [i for i, v in enumerate(g) if v == OCIO and i not in clases_fijas]
        random.shuffle(pool)
        
        while faltan > 0 and pool:
            g[pool.pop()] = actividad
            faltan -= 1
        
        # Si aún faltan, usar LIBRE
        if faltan > 0:
            pool = [i for i, v in enumerate(g) if v == LIBRE and i not in clases_fijas]
            random.shuffle(pool)
            while faltan > 0 and pool:
                g[pool.pop()] = actividad
                faltan -= 1
    
    completar_actividad(GYM, objetivo_gym)
    completar_actividad(ESTUDIO, objetivo_estudio)
    
    # Garantizar 1 LIBRE por día
    for dia_idx in range(n_dias):
        libres_del_dia = sum(1 for b in range(n_bloques) 
                           if g[posicion(dia_idx, b)] == LIBRE)
        
        if libres_del_dia >= min_libre_por_dia:
            continue
        
        # Convertir OCIO a LIBRE si es posible
        candidatos = [posicion(dia_idx, b) for b in range(n_bloques) 
                     if g[posicion(dia_idx, b)] == OCIO and posicion(dia_idx, b) not in clases_fijas]
        
        if candidatos:
            g[random.choice(candidatos)] = LIBRE
    
    return g

def cruce(padre1, padre2, prob_cruce=0.8):
    "Cruce de dos puntos"
    if random.random() > prob_cruce:
        hijo = padre1[:]
    else:
        punto1, punto2 = sorted(random.sample(range(total_bloques), 2))
        hijo = padre1[:punto1] + padre2[punto1:punto2] + padre1[punto2:]
    
    return reparar(hijo)

def mutacion(genoma, prob_mutacion=0.06):
    "Mutación por swap y reasignación"
    g = genoma[:]
    
    # Swap dentro del mismo día
    if random.random() < prob_mutacion:
        dia_idx = random.randrange(n_dias)
        bloque1, bloque2 = random.sample(range(n_bloques), 2)
        pos1 = posicion(dia_idx, bloque1)
        pos2 = posicion(dia_idx, bloque2)
        
        if pos1 not in clases_fijas and pos2 not in clases_fijas:
            g[pos1], g[pos2] = g[pos2], g[pos1]
    
    # Reasignación
    if random.random() < prob_mutacion:
        pos = random.randrange(total_bloques)
        if pos not in clases_fijas:
            g[pos] = random.choice(actividades_mutables)
    
    return reparar(g)

def evolucionar(poblacion, generaciones=200, elitismo=2):
    "Algoritmo genético principal"
    mejor = max(poblacion, key=fitness)
    
    for gen in range(1, generaciones + 1):
        nueva_poblacion = []
        
        # Elitismo
        ordenados = sorted(poblacion, key=fitness, reverse=True)
        nueva_poblacion.extend(ordenados[:elitismo])
        
        # Reproducción
        while len(nueva_poblacion) < len(poblacion):
            padre1 = seleccion(poblacion)
            padre2 = seleccion(poblacion)
            hijo = cruce(padre1, padre2)
            hijo = mutacion(hijo)
            nueva_poblacion.append(hijo)
        
        poblacion = nueva_poblacion
        candidato = max(poblacion, key=fitness)
        
        if fitness(candidato) > fitness(mejor):
            mejor = candidato
        
        if gen % 20 == 0 or gen == 1 or gen == generaciones:
            print(f"Gen {gen:3d} | Mejor fitness: {fitness(mejor):.2f}")
    
    return mejor

def mostrar_horario(genoma):
    "Muestra el horario en formato tabla"
    print("\nHorario semanal (bloques de 2h):\n")
    
    # Encabezado
    cabecera = "Hora      | " + " | ".join(f"{dia:^8}" for dia in dias)
    print(cabecera)
    print("-" * len(cabecera))
    
    # Filas
    for bloque_idx, hora in enumerate(horas_inicio):
        fila = []
        for dia_idx in range(n_dias):
            pos = posicion(dia_idx, bloque_idx)
            actividad = genoma[pos]
            fila.append(f"{actividad:^8}")
        
        print(f"{hora:02d}-{hora + duracion_bloque:02d}   | " + " | ".join(fila))

def mostrar_kpis(genoma):
    "Muestra métricas del horario"
    conteo = Counter(genoma)
    
    # Días con GYM
    dias_gym = []
    for dia_idx in range(n_dias):
        tiene_gym = any(genoma[posicion(dia_idx, b)] == GYM for b in range(n_bloques))
        if tiene_gym:
            dias_gym.append(dias[dia_idx])
    
    # LIBRES por día
    libres_por_dia = {}
    for dia_idx in range(n_dias):
        libres = sum(1 for b in range(n_bloques) 
                    if genoma[posicion(dia_idx, b)] == LIBRE)
        libres_por_dia[dias[dia_idx]] = libres
    
    return {
        "fitness": round(fitness(genoma), 2),
        "conteos": dict(conteo),
        "dias_gym": dias_gym,
        "libres_por_dia": libres_por_dia
    }

def generar_imagen_horario(genoma, archivo="horario_generado.png"):
    "Genera una imagen del horario"
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import os
        
        # Colores para cada actividad
        colores = {
            CLASE: '#FF6B6B',    # Rojo
            ESTUDIO: '#4ECDC4',  # Turquesa
            GYM: '#45B7D1',      # Azul
            OCIO: '#96CEB4',     # Verde
            LIBRE: '#FFEAA7'     # Amarillo
        }
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Dibujar celdas
        for dia_idx in range(n_dias):
            for bloque_idx in range(n_bloques):
                pos = posicion(dia_idx, bloque_idx)
                actividad = genoma[pos]
                color = colores.get(actividad, '#DDDDDD')
                
                # Invertir Y para que 6:00 quede arriba
                y = n_bloques - 1 - bloque_idx
                
                rect = mpatches.Rectangle((dia_idx, y), 1, 1, 
                                        facecolor=color, edgecolor='white', linewidth=1)
                ax.add_patch(rect)
                ax.text(dia_idx + 0.5, y + 0.5, actividad, 
                       ha='center', va='center', fontsize=9)
        
        # Configurar ejes
        ax.set_xlim(0, n_dias)
        ax.set_ylim(0, n_bloques)
        ax.set_xticks([i + 0.5 for i in range(n_dias)])
        ax.set_xticklabels(dias, fontsize=11, fontweight='bold')
        
        # Etiquetas de horas
        horas_labels = [f"{h:02d}-{h + duracion_bloque:02d}" for h in reversed(horas_inicio)]
        ax.set_yticks([i + 0.5 for i in range(n_bloques)])
        ax.set_yticklabels(horas_labels, fontsize=9)
        
        # Cuadrícula
        for x in range(n_dias + 1):
            ax.plot([x, x], [0, n_bloques], color='white', linewidth=1)
        for y in range(n_bloques + 1):
            ax.plot([0, n_dias], [y, y], color='white', linewidth=1)
        
        # Estilo
        ax.set_aspect('equal')
        ax.set_facecolor('#F5F5F5')
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Título
        ax.set_title(f"Horario Semanal Optimizado (Fitness: {fitness(genoma):.2f})", 
                    fontsize=13, pad=14)
        
        # Leyenda
        handles = [mpatches.Patch(color=colores[act], label=act) for act in colores.keys()]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        
        # Guardar
        plt.savefig(archivo, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        
        return archivo
        
    except ImportError:
        print("matplotlib no está instalado. Ejecuta: pip install matplotlib")
        return None

# Ejecutar algoritmo
if __name__ == "__main__":
    print("Algoritmo Genético para Horarios")
    print("=" * 40)
    
    # Crear población inicial
    poblacion = crear_poblacion(80)
    print(f"Población inicial: {len(poblacion)} individuos")
    
    # Evolucionar
    mejor_horario = evolucionar(poblacion, generaciones=200, elitismo=2)
    
    # Mostrar resultados
    mostrar_horario(mejor_horario)
    
    kpis = mostrar_kpis(mejor_horario)
    print(f"\nKPIs del mejor horario:")
    print(f"Fitness: {kpis['fitness']}")
    print(f"Conteos: {kpis['conteos']}")
    print(f"Días con GYM: {kpis['dias_gym']}")
    print(f"LIBRES por día: {kpis['libres_por_dia']}")
    
    # Generar imagen
    print("\nGenerando imagen del horario...")
    archivo_imagen = generar_imagen_horario(mejor_horario)
    if archivo_imagen:
        print(f"Imagen guardada como: {archivo_imagen}")
    
    print("\n ¡Horario optimizado completado!")
