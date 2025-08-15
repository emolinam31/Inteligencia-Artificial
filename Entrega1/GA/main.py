#Punto 2 Entrega 1 IA 2025-2 Miguel Villegas y Esteban Molina

# ===========================
#  GA: Horario Semanal (bloques de 2 h)
#  Autor: tú
#  Objetivo: 56 bloques (7 días × 8 bloques), actividades:
#    - 25 CLASE (fijas)
#    - 5 GYM (1 bloque c/u)
#    - 7 ESTUDIO (1 bloque c/u)
#    - OCIO limitado a disponibilidad
#    - ≥ 1 LIBRE por día
#  Restricciones duras: clases fijas, conteos requeridos, al menos 1 LIBRE/día
#  Restricciones blandas: evitar GYM en días consecutivos, compactar jornadas, estudio contiguo
# ===========================

import random
from collections import Counter
from typing import List, Tuple, Optional

# ---------- Configuración general ----------
DIAS   = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
INICIO = 6     # 06:00
FIN    = 22    # 22:00 (exclusivo)
BLOQUE = 2     # horas por bloque

SLOTS = list(range(INICIO, FIN, BLOQUE))    # [6,8,10,12,14,16,18,20]
N_DIAS, N_SLOTS = len(DIAS), len(SLOTS)
N_GENES = N_DIAS * N_SLOTS                   # 56

# Objetivos/controles de conteo
N_CLASES  = 25
N_GYM     = 5
N_ESTUDIO = 7
MIN_LIBRE_POR_DIA = 1

# Actividades
CLASE = "CLASE"
GYM = "GYM"
ESTUDIO = "ESTUDIO"
OCIO = "OCIO"
LIBRE = "LIBRE"
ACT_MUTABLES = [GYM, ESTUDIO, OCIO, LIBRE]  # CLASE no se crea por mutación

# Fija tu listado real aquí: [(día, hora_inicio), ...], p.ej. [("Lun", 8), ("Lun", 10), ...]
CLASES_FIJAS_USUARIO: Optional[List[Tuple[str, int]]] = None  # <-- pon tu lista o deja None para ejemplo

# Semilla para reproducibilidad
RANDOM_SEED = 42

# ---------- Utilidades de índice ----------
dia_a_idx  = {d:i for i,d in enumerate(DIAS)}
hora_a_slot= {h:i for i,h in enumerate(SLOTS)}
def pos(d, s): return d * N_SLOTS + s

# ---------- Generador de clases fijas (ejemplo) ----------
def generar_clases_fijas_ejemplo(n=25, seed=RANDOM_SEED):
    """
    Distribuye 25 clases en la semana, sin choques, intentando no saturar ningún día.
    Se puede reemplazar por la lista real del usuario.
    """
    random.seed(seed)
    # Distribución diaria aproximada que suma 25 (4,4,4,4,3,3,3)
    dist = [4,4,4,4,3,3,3]
    assert sum(dist) == n
    libres_por_dia = {d:set(range(N_SLOTS)) for d in range(N_DIAS)}
    ejemplo = []
    for d, cupo in enumerate(dist):
        candidatos = list(libres_por_dia[d])
        random.shuffle(candidatos)
        elegidos = candidatos[:cupo]
        for s in elegidos:
            ejemplo.append((DIAS[d], SLOTS[s]))
            libres_por_dia[d].remove(s)
    return ejemplo

def clases_fijas_indices():
    """
    Retorna un diccionario {idx_gen: "CLASE"} con las clases fijas.
    Si el usuario no define su lista, se genera un ejemplo.
    """
    if CLASES_FIJAS_USUARIO is None:
        fijo = generar_clases_fijas_ejemplo(N_CLASES)
    else:
        fijo = CLASES_FIJAS_USUARIO
        if len(fijo) != N_CLASES:
            raise ValueError(f"Debes definir exactamente {N_CLASES} clases fijas; definiste {len(fijo)}.")

    dic = {}
    for (dia, hora) in fijo:
        if dia not in dia_a_idx or hora not in hora_a_slot:
            raise ValueError(f"Clase fija inválida: {(dia, hora)}. Revisa día/hora.")
        d = dia_a_idx[dia]; s = hora_a_slot[hora]
        dic[pos(d,s)] = CLASE
    return dic

FIJAS = clases_fijas_indices()

# ---------- Representación ----------
def nuevo_genoma_vacio()->List[str]:
    g = [LIBRE] * N_GENES
    for i in FIJAS:
        g[i] = CLASE
    return g

# ---------- Construcción de individuo ----------
def crear_individuo()->List[str]:
    """
    Construye una solución inicial factible respecto a:
    - No mover clases fijas
    - Satisfacer conteos objetivo de GYM y ESTUDIO
    - Reservar ≥1 LIBRE por día
    - Rellenar con OCIO
    """
    g = nuevo_genoma_vacio()

    # 1) Reservar 1 LIBRE por día (si el día está muy copado por CLASE, se elegirá un slot libre cualquier)
    for d in range(N_DIAS):
        slots_d = [pos(d,s) for s in range(N_SLOTS)]
        candidatos = [i for i in slots_d if g[i] == LIBRE]
        if candidatos:
            # Prefiere horas centrales para descanso
            candidatos.sort(key=lambda i: abs(SLOTS[i % N_SLOTS]-14))
            g[candidatos[0]] = LIBRE

    # 2) Colocar ESTUDIO (7 bloques): preferimos contiguidad de 2 bloques en un mismo día cuando se pueda
    restantes = N_ESTUDIO
    intentos = 0
    while restantes > 0 and intentos < 1000:
        d = random.randrange(N_DIAS)
        # intentar pareja contigua
        pares = [(s, s+1) for s in range(N_SLOTS-1)]
        random.shuffle(pares)
        puesto = False
        for s1,s2 in pares:
            i1, i2 = pos(d,s1), pos(d,s2)
            if g[i1] == LIBRE and g[i2] == LIBRE and i1 not in FIJAS and i2 not in FIJAS:
                g[i1] = ESTUDIO; restantes -= 1
                if restantes > 0:
                    g[i2] = ESTUDIO; restantes -= 1
                puesto = True
                break
        if not puesto:
            # colocar individual
            libres = [pos(d,s) for s in range(N_SLOTS) if g[pos(d,s)]==LIBRE and pos(d,s) not in FIJAS]
            if libres:
                g[random.choice(libres)] = ESTUDIO
                restantes -= 1
        intentos += 1

    # 3) Colocar GYM (5 bloques): preferir tardes, y evitar (suave) consecutividad diaria (se afina en fitness)
    restantes = N_GYM
    intentos = 0
    while restantes > 0 and intentos < 1000:
        d = random.randrange(N_DIAS)
        slots_tarde = [s for s,h in enumerate(SLOTS) if h>=16]
        random.shuffle(slots_tarde)
        colocado = False
        for s in slots_tarde + list(range(N_SLOTS)):  # si no hay tarde, donde se pueda
            i = pos(d,s)
            if g[i] == LIBRE and i not in FIJAS:
                g[i] = GYM
                restantes -= 1
                colocado = True
                break
        intentos += 1

    # 4) Rellenar con OCIO (lo que quede sin CLASE/ESTUDIO/GYM/LIBRE)
    for i in range(N_GENES):
        if i in FIJAS: 
            g[i] = CLASE
        elif g[i] == LIBRE:
            # Mantenemos LIBRE reservado como está; el resto a OCIO
            pass
    for i in range(N_GENES):
        if g[i] not in (CLASE, ESTUDIO, GYM, LIBRE):
            g[i] = OCIO

    return g

def crear_poblacion(size=60)->List[List[str]]:
    return [crear_individuo() for _ in range(size)]

# ---------- Fitness ----------
def fitness(g:List[str])->float:
    """
    Maximizar.
    Penalizaciones (duras y blandas) y recompensas:
      - [DURA] Clases fijas respetadas
      - [DURA] Conteos exactos: CLASE=25, GYM=5, ESTUDIO=7
      - [DURA] ≥1 LIBRE por día
      - [BLANDA] Evitar GYM en días consecutivos
      - [BLANDA] Recompensar estudio en parejas contiguas
      - [BLANDA] Penalizar huecos LIBRE entre actividades el mismo día (compactar)
      - [BLANDA] Recompensar más OCIO en fin de semana (Sáb/Dom)
    """
    P = 0  # penalizaciones
    R = 0  # recompensas

    # Duras: clases fijas
    for i, act in FIJAS.items():
        if g[i] != CLASE:
            P += 500  # muy alto

    # Duras: conteos exactos
    c = Counter(g)
    P += abs(c[CLASE]  - N_CLASES)  * 100
    # Si por cruce/mutación tocó clases, la reparación suele corregir; esto refuerza.
    P += abs(c[GYM]    - N_GYM)     * 30
    P += abs(c[ESTUDIO]- N_ESTUDIO) * 30

    # Dura: ≥1 LIBRE por día
    for d in range(N_DIAS):
        libres_d = sum(1 for s in range(N_SLOTS) if g[pos(d,s)] == LIBRE)
        if libres_d < MIN_LIBRE_POR_DIA:
            P += 40

    # Blanda: GYM en días consecutivos (penaliza)
    dias_gym = sorted({d for d in range(N_DIAS) if any(g[pos(d,s)]==GYM for s in range(N_SLOTS))})
    for a, b in zip(dias_gym, dias_gym[1:]):
        if b - a == 1:
            P += 5

    # Blanda: estudio contiguo (recompensa)
    for d in range(N_DIAS):
        for s in range(N_SLOTS-1):
            a, b = g[pos(d,s)], g[pos(d,s+1)]
            if a==ESTUDIO and b==ESTUDIO:
                R += 3

    # Blanda: huecos (LIBRE) entre actividades no libres (penaliza)
    for d in range(N_DIAS):
        for s in range(1, N_SLOTS-1):
            a,b,c3 = g[pos(d,s-1)], g[pos(d,s)], g[pos(d,s+1)]
            if b==LIBRE and a!=LIBRE and c3!=LIBRE:
                P += 2

    # Blanda: ocio concentrado fin de semana (recompensa)
    ocio_weekend = sum(1 for d in [5,6] for s in range(N_SLOTS) if g[pos(d,s)]==OCIO)
    ocio_weekday = sum(1 for d in [0,1,2,3,4] for s in range(N_SLOTS) if g[pos(d,s)]==OCIO)
    if ocio_weekend + ocio_weekday > 0:
        R += int(10 * ocio_weekend / (ocio_weekend + ocio_weekday))

    # Nota: OCIO queda automáticamente limitado por la disponibilidad
    return 1000 - P + R

# ---------- Operadores GA ----------
def seleccion(poblacion:List[List[str]])->List[str]:
    a,b,c = random.sample(poblacion, 3)  # torneo de 3
    return max([a,b,c], key=fitness)

def reparar(ind:List[str])->List[str]:
    """
    Repara (post cruce/mutación):
      - Fijar CLASE en posiciones fijas
      - Ajustar conteos de GYM/ESTUDIO al objetivo (usando OCIO/LIBRE como reserva)
      - Garantizar ≥1 LIBRE por día (si es posible)
    """
    g = ind[:]

    # Fijar clases
    for i in FIJAS: g[i] = CLASE

    # Conteos actuales
    c = Counter(g)

    # 1) Quitar exceso de GYM/ESTUDIO convirtiendo a OCIO
    exceso_gym = max(0, c[GYM]-N_GYM)
    exceso_est = max(0, c[ESTUDIO]-N_ESTUDIO)
    if exceso_gym:
        idxs = [i for i,v in enumerate(g) if v==GYM and i not in FIJAS]
        random.shuffle(idxs)
        for i in idxs[:exceso_gym]: g[i] = OCIO
    if exceso_est:
        idxs = [i for i,v in enumerate(g) if v==ESTUDIO and i not in FIJAS]
        random.shuffle(idxs)
        for i in idxs[:exceso_est]: g[i] = OCIO

    # 2) Completar faltantes de GYM/ESTUDIO usando primero OCIO, luego LIBRE
    def completar(label, objetivo):
        faltan = objetivo - Counter(g)[label]
        if faltan <= 0: return
        pool = [i for i,v in enumerate(g) if v==OCIO and i not in FIJAS]
        random.shuffle(pool)
        while faltan>0 and pool:
            g[pool.pop()] = label
            faltan -= 1
        if faltan>0:
            pool = [i for i,v in enumerate(g) if v==LIBRE and i not in FIJAS]
            random.shuffle(pool)
            while faltan>0 and pool:
                g[pool.pop()] = label
                faltan -= 1
    completar(GYM, N_GYM)
    completar(ESTUDIO, N_ESTUDIO)

    # 3) Garantizar ≥1 LIBRE por día (si todavía hay hueco)
    for d in range(N_DIAS):
        if sum(1 for s in range(N_SLOTS) if g[pos(d,s)]==LIBRE) >= MIN_LIBRE_POR_DIA:
            continue
        # Convertir un OCIO a LIBRE (o actividad blanda) si existe
        candidatos = [pos(d,s) for s in range(N_SLOTS) if g[pos(d,s)]==OCIO and pos(d,s) not in FIJAS]
        if candidatos:
            g[random.choice(candidatos)] = LIBRE
        else:
            # Como última opción, si no hay OCIO, intentar convertir GYM/ESTUDIO (muy raro)
            cand2 = [pos(d,s) for s in range(N_SLOTS) if g[pos(d,s)] in (GYM, ESTUDIO) and pos(d,s) not in FIJAS]
            if cand2:
                g[random.choice(cand2)] = LIBRE

    return g

def cruce(p1:List[str], p2:List[str], pc=0.8)->List[str]:
    if random.random() > pc:
        hijo = p1[:]
    else:
        c1,c2 = sorted(random.sample(range(N_GENES), 2))
        hijo = p1[:c1] + p2[c1:c2] + p1[c2:]
    return reparar(hijo)

def mutacion(ind:List[str], pm=0.06)->List[str]:
    g = ind[:]
    # swap dentro del mismo día
    if random.random() < pm:
        d = random.randrange(N_DIAS)
        s1, s2 = random.sample(range(N_SLOTS), 2)
        i1, i2 = pos(d,s1), pos(d,s2)
        if i1 not in FIJAS and i2 not in FIJAS:
            g[i1], g[i2] = g[i2], g[i1]
    # reasignación blanda
    if random.random() < pm:
        i = random.randrange(N_GENES)
        if i not in FIJAS:
            g[i] = random.choice(ACT_MUTABLES)
    return reparar(g)

# ---------- Bucle evolutivo ----------
def evolucion(poblacion:List[List[str]], generaciones=200, elitismo=2, seed=RANDOM_SEED)->List[str]:
    random.seed(seed)
    best = max(poblacion, key=fitness)
    for gen in range(1, generaciones+1):
        nueva = []
        # elitismo
        orden = sorted(poblacion, key=fitness, reverse=True)
        nueva.extend(orden[:elitismo])
        # reproducción
        while len(nueva) < len(poblacion):
            p1 = seleccion(poblacion)
            p2 = seleccion(poblacion)
            hijo = cruce(p1, p2)
            hijo = mutacion(hijo)
            nueva.append(hijo)
        poblacion = nueva
        cand = max(poblacion, key=fitness)
        if fitness(cand) > fitness(best):
            best = cand
        if gen % 20 == 0 or gen==1 or gen==generaciones:
            print(f"Gen {gen:3d} | Best fitness: {fitness(best):.2f}")
    return best

# ---------- Salida / Visualización ----------
def grid(g:List[str]):
    return [[g[pos(d,s)] for s in range(N_SLOTS)] for d in range(N_DIAS)]

def imprimir_horario(g:List[str]):
    tabla = grid(g)
    cab = "Hora      | " + " | ".join(f"{d:^8}" for d in DIAS)
    print("\nHorario (bloques de 2h):\n")
    print(cab)
    print("-"*len(cab))
    for s_idx, h in enumerate(SLOTS):
        fila = [f"{tabla[d][s_idx]:^8}" for d in range(N_DIAS)]
        print(f"{h:02d}-{h+BLOQUE:02d}   | " + " | ".join(fila))

def horario_markdown(g:List[str])->str:
    tabla = grid(g)
    lineas = []
    lineas.append("| Hora  | " + " | ".join(DIAS) + " |")
    lineas.append("|:-----:|" + "|".join([":-----:"]*len(DIAS)) + "|")
    for s_idx,h in enumerate(SLOTS):
        fila = [tabla[d][s_idx] for d in range(N_DIAS)]
        lineas.append(f"| {h:02d}-{h+BLOQUE:02d} | " + " | ".join(fila) + " |")
    return "\n".join(lineas)

def kpis(g:List[str])->dict:
    c = Counter(g)
    dias_gym = sorted({DIAS[d] for d in range(N_DIAS) if any(g[pos(d,s)]==GYM for s in range(N_SLOTS))})
    libres_por_dia = {DIAS[d]: sum(1 for s in range(N_SLOTS) if g[pos(d,s)]==LIBRE) for d in range(N_DIAS)}
    return {
        "fitness": round(fitness(g),2),
        "conteos": dict(c),
        "dias_gym": dias_gym,
        "libres_por_dia": libres_por_dia
    }
    
def export_schedule_image(genome, filename="horario.png", palette=None, dpi=200, title=None):
    """
    Genera una imagen del horario (tabla días × bloques de 2h).
    Requiere que existan: DIAS, SLOTS, BLOQUE, N_DIAS, N_SLOTS, pos(), y las etiquetas CLASE/ESTUDIO/GYM/OCIO/LIBRE.

    Parámetros
    ----------
    genome : list[str]
        Lista de longitud N_GENES con una actividad por bloque.
    filename : str
        Ruta del archivo de salida (PNG).
    palette : dict[str, str] | None
        Mapeo opcional {'CLASE':'#color', 'ESTUDIO':'#color', ...}.
        Si es None, usa el ciclo de colores por defecto de Matplotlib.
    dpi : int
        Resolución de la imagen.
    title : str | None
        Título a mostrar encima. Si None, se usa uno por defecto.

    Retorna
    -------
    str : ruta del archivo generado.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.patches import Rectangle

    # Validación mínima
    if len(genome) != N_DIAS * N_SLOTS:
        raise ValueError(f"El genoma debe tener {N_DIAS*N_SLOTS} genes, recibí {len(genome)}.")

    # Etiquetas que esperaremos colorear
    etiquetas = [CLASE, ESTUDIO, GYM, OCIO, LIBRE]

    # Paleta por defecto: ciclo de Matplotlib (evita hardcodear colores fijos)
    if palette is None:
        ciclo = plt.rcParams.get('axes.prop_cycle', None)
        if ciclo is not None:
            base = ciclo.by_key().get('color', [])
        else:
            base = []
        # Fallback simple si no hay ciclo disponible
        if not base:
            base = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        palette = {etq: base[i % len(base)] for i, etq in enumerate(etiquetas)}

    # Figura proporcional al tamaño de la grilla
    fig_w = 1.4 * N_DIAS
    fig_h = 0.9 * N_SLOTS + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # Dibujar celdas (fila = hora, columna = día)
    # Queremos la hora más temprana arriba, así que invertimos y (usando N_SLOTS-1-r)
    for r in range(N_SLOTS):          # r: índice de bloque horario (0=06-08, ...)
        for c in range(N_DIAS):       # c: índice de día
            etiqueta = genome[pos(c, r)]
            color = palette.get(etiqueta, "#DDDDDD")
            # y visual = invertido para que 06:00 quede arriba
            y = N_SLOTS - 1 - r
            rect = Rectangle((c, y), 1, 1, facecolor=color, edgecolor="white", linewidth=1.2)
            ax.add_patch(rect)
            ax.text(c + 0.5, y + 0.5, etiqueta, ha="center", va="center", fontsize=9)

    # Límites y ticks
    ax.set_xlim(0, N_DIAS)
    ax.set_ylim(0, N_SLOTS)
    ax.set_xticks([i + 0.5 for i in range(N_DIAS)])
    ax.set_xticklabels(DIAS, fontsize=11, fontweight="bold")
    # Etiquetas de horas (invertidas: mostramos de arriba hacia abajo)
    horas_labels = [f"{h:02d}-{h + BLOQUE:02d}" for h in reversed(SLOTS)]
    ax.set_yticks([i + 0.5 for i in range(N_SLOTS)])
    ax.set_yticklabels(horas_labels, fontsize=9)

    # Cuadrícula fina
    for x in range(N_DIAS + 1):
        ax.plot([x, x], [0, N_SLOTS], color="white", linewidth=1.2)
    for y in range(N_SLOTS + 1):
        ax.plot([0, N_DIAS], [y, y], color="white", linewidth=1.2)

    # Estética
    ax.set_aspect("equal")
    ax.set_facecolor("#F5F5F5")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Título
    if title is None:
        title = "Horario semanal (bloques de 2 h)"
    ax.set_title(title, fontsize=13, pad=14)

    # Leyenda
    handles = [mpl.patches.Patch(color=palette[e], label=e) for e in etiquetas]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    # Guardar
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return filename


# ---------- Main ----------
if __name__ == "__main__":
    pop = crear_poblacion(size=80)
    mejor = evolucion(pop, generaciones=200, elitismo=2, seed=RANDOM_SEED)
    imprimir_horario(mejor)
    print("\nKPIs:", kpis(mejor))
    print("\nMarkdown (para Wiki de GitHub):\n")
    print(horario_markdown(mejor))
    
    # Generar imagen del horario
    print("\nGenerando imagen del horario...")
    try:
        import os
        # Paleta de colores personalizada (opcional)
        palette = {
            CLASE: '#FF6B6B',    # Rojo claro
            ESTUDIO: '#4ECDC4',  # Turquesa
            GYM: '#45B7D1',      # Azul
            OCIO: '#96CEB4',     # Verde menta
            LIBRE: '#FFEAA7'     # Amarillo claro
        }
        
        # Determinar la ruta de guardado
        # Si estamos ejecutando desde la carpeta GA, guardamos aquí mismo
        # Si estamos desde el root del proyecto, guardamos en Entrega1/GA/
        if os.path.exists("main.py") and "GA" in os.getcwd():
            # Estamos en la carpeta GA
            ruta_imagen = "horario_generado.png"
        else:
            # Estamos en el root o en otra carpeta
            ruta_imagen = "Entrega1/GA/horario_generado.png"
            # Crear la carpeta si no existe
            os.makedirs("Entrega1/GA", exist_ok=True)
        
        # Generar la imagen
        archivo = export_schedule_image(
            mejor, 
            filename=ruta_imagen, 
            palette=palette,
            dpi=200,
            title=f"Horario Semanal Optimizado (Fitness: {fitness(mejor):.2f})"
        )
        print(f"✓ Imagen guardada como: {archivo}")
    except ImportError:
        print("⚠ matplotlib no está instalado. Ejecuta: pip install matplotlib")

