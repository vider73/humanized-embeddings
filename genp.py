import json
import time
import sys

# ==========================================
# 1. DEFINICIÓN DE LA REALIDAD (100+ Dimensiones)
# ==========================================
# Tuplas formato: (Nombre Dimensión, Anchor 0.0, Anchor 1.0)

DIMENSIONS_DB = [
    # --- FÍSICA: ESPACIO Y TIEMPO ---
    ("Tamaño Físico", "Una partícula subatómica (Quark)", "El Universo Observable completo"),
    ("Masa", "Un fotón (sin masa)", "Singularidad de Agujero Negro Supermasivo"),
    ("Densidad", "Vacío intergaláctico perfecto", "Estrella de neutrones / Singularidad"),
    ("Velocidad Máxima", "Estático absoluto (0 Kelvin)", "Velocidad de la luz (c)"),
    ("Temperatura", "Cero Absoluto (-273.15°C)", "Temperatura de Planck (Big Bang)"),
    ("Duración Temporal", "Tiempo de Planck (Instante mínimo)", "Eternidad / Entropía final"),
    ("Distancia de Interacción", "Contacto intra-atómico", "Alcance cosmológico infinito"),
    ("Gravedad", "Microgravedad despreciable", "Horizonte de sucesos"),
    ("Luminosidad", "Oscuridad absoluta (Vantablack)", "Quasar activo / Supernova"),
    ("Transparencia", "Opacidad total (Plomo denso)", "Invisible / Transparencia perfecta"),
    
    # --- FÍSICA: MATERIA Y MATERIALES ---
    ("Dureza", "Gas noble difuso", "Diamante agregado de nanobarrenas"),
    ("Elasticidad", "Arcilla mojada (Deformación plástica)", "Grafeno tensado / Super-rebote"),
    ("Viscosidad", "Superfluido (Viscosidad cero)", "Sólido amorfo / Brea"),
    ("Fricción", "Levitación magnética en vacío", "Superficie de lija de diamante"),
    ("Fragilidad", "Acero templado indestructible", "Pompa de jabón"),
    ("Solidez", "Gas etéreo", "Bloque de tungsteno macizo"),
    ("Volatilidad Química", "Oro inerte", "Francio reaccionando con agua"),
    ("Conductividad Eléctrica", "Aislante perfecto", "Superconductor a temperatura ambiente"),
    ("Magnetismo", "Material no magnético", "Magnetar"),
    ("Radiactividad", "Isótopo estable", "Núcleo de reactor fundido"),
    ("Porosidad", "Cristal denso perfecto", "Esponja a nivel molecular"),
    ("Maleabilidad", "Vidrio quebradizo", "Oro puro"),

    # --- BIOLOGÍA: VIDA Y ORGANISMO ---
    ("Vitalidad", "Materia inorgánica inerte", "Organismo joven en plenitud"),
    ("Consciencia", "Roca", "Mente humana genial / IA General"),
    ("Instinto", "Objeto inanimado", "Depredador alfa cazando"),
    ("Toxicidad", "Agua pura", "Toxina botulínica pura"),
    ("Valor Nutricional", "Veneno / Metal", "Superalimento completo"),
    ("Edad Biológica", "Cigoto recién fecundado", "Anciano centenario"),
    ("Salud", "Fallo multiorgánico", "Homeostasis perfecta"),
    ("Fertilidad", "Estéril", "Colonia bacteriana exponencial"),
    ("Evolución", "Bacteria primordial", "Ser post-biológico trascendido"),
    ("Capacidad Sensorial", "Ciego/Sordo/Insensible", "Percepción extrasensorial total"),

    # --- CUALIA: LOS SENTIDOS ---
    ("Sonoridad (Volumen)", "Silencio anecoico", "Erupción volcánica / Onda de choque"),
    ("Frecuencia Sonora", "Infrasonido indetectable", "Ultrasonido de alta frecuencia"),
    ("Olor (Intensidad)", "Sin olor (Vacío)", "Amoníaco concentrado / Mofeta"),
    ("Olor (Agradabilidad)", "Cadáver en descomposición", "Perfume celestial perfecto"),
    ("Sabor (Dulzor)", "Agua insípida", "Edulcorante puro concentrado"),
    ("Sabor (Amargor)", "Agua", "Denatonio (Sustancia más amarga)"),
    ("Sabor (Picante)", "Leche", "Capsaicina pura (Escala Scoville Max)"),
    ("Rugosidad Táctil", "Espejo pulido atómicamente", "Papel de lija grano 1"),
    ("Humedad", "Desierto de Atacama", "Océano profundo"),
    ("Saturación de Color", "Escala de grises", "Láser RGB saturado"),
    ("Brillo Superficial", "Mate absoluto", "Espejo perfecto"),
    ("Temperatura Táctil", "Nitrógeno líquido", "Metal al rojo vivo"),
    ("Pegajosidad", "Teflón antiadherente", "Superpegamento industrial"),

    # --- EMOCIÓN: ESPECTRO POSITIVO ---
    ("Felicidad", "Miseria absoluta", "Éxtasis eufórico"),
    ("Amor", "Indiferencia fría", "Amor incondicional profundo"),
    ("Calma", "Pánico histérico", "Paz Zen absoluta"),
    ("Esperanza", "Desesperación nihilista", "Fe inquebrantable en el futuro"),
    ("Confianza", "Paranoia", "Seguridad ciega"),
    ("Orgullo", "Vergüenza humillante", "Honor supremo"),
    ("Valentía", "Cobardía paralizante", "Heroísmo suicida"),
    ("Gratitud", "Resentimiento", "Agradecimiento infinito"),
    ("Curiosidad", "Apatía total", "Obsesión por descubrir"),
    ("Empatía", "Psicopatía insensible", "Conexión emocional total"),

    # --- EMOCIÓN: ESPECTRO NEGATIVO ---
    ("Miedo", "Seguridad total", "Terror primario visceral"),
    ("Ira", "Serenidad", "Furia berserker asesina"),
    ("Tristeza", "Alegría desbordante", "Depresión profunda / Duelo"),
    ("Asco", "Atracción", "Repulsión vomitiva"),
    ("Culpa", "Inocencia", "Remordimiento insoportable"),
    ("Envidia", "Admiración sana", "Celos destructivos"),
    ("Soledad", "Compañía constante", "Aislamiento solipsista"),
    ("Estrés", "Relajación", "Colapso nervioso"),
    ("Odio", "Afecto", "Aversión destructiva total"),
    ("Aburrimiento", "Diversión frenética", "Tedio existencial"),

    # --- INTELECTO Y LOGOS ---
    ("Verdad (Facticidad)", "Falsedad absoluta / Mentira", "Verdad axiomática universal"),
    ("Complejidad Lógica", "Tautología simple (A=A)", "Paradoja irresoluble / Teoría del Todo"),
    ("Racionalidad", "Delirio irracional", "Lógica matemática pura"),
    ("Probabilidad", "Imposible (0%)", "Inevitable (100%)"),
    ("Abstracción", "Objeto físico concreto", "Concepto metafísico puro"),
    ("Conocimiento Requerido", "Instintivo / Innato", "Erudición experta"),
    ("Creatividad", "Repetición mecánica", "Innovación disruptiva genial"),
    ("Memoria", "Olvido instantáneo", "Registro Akáshico total"),
    ("Claridad", "Confusión críptica", "Evidencia cristalina"),
    ("Sabiduría", "Ignorancia necia", "Iluminación omnisciente"),

    # --- SOCIEDAD Y CULTURA ---
    ("Valor Económico", "Basura sin valor", "PIB Mundial combinado"),
    ("Legalidad", "Crimen capital", "Mandato constitucional"),
    ("Ética (Bondad)", "Maldad pura", "Santidad altruista"),
    ("Fama", "Anónimo desconocido", "Icono histórico global"),
    ("Poder Político", "Esclavo sin derechos", "Emperador global"),
    ("Artificialidad", "100% Natural", "100% Sintético / Artificial"),
    ("Industrialización", "Artesanal / Primitivo", "Producción en masa automatizada"),
    ("Lujo", "Austeridad de supervivencia", "Opulencia decadente"),
    ("Tradición", "Novedad radical", "Rito ancestral milenario"),
    ("Religiosidad", "Ateo / Profano", "Sagrado / Divino"),
    ("Urbanismo", "Naturaleza virgen", "Megalópolis Cyberpunk"),
    ("Belleza Estética", "Monstruosidad deforme", "Perfección áurea sublime"),
    ("Formalidad", "Vulgar / Callejero", "Protocolo real estricto"),
    ("Peligrosidad Social", "Ciudadano modelo", "Amenaza pública global"),

    # --- METAFÍSICA Y NARRATIVA ---
    ("Magia", "Mundo físico ordinario", "Omnipotencia mágica"),
    ("Divinidad", "Mortal efímero", "Dios Creador"),
    ("Futurismo", "Prehistoria", "Ciencia Ficción año 3000"),
    ("Misterio", "Hecho evidente", "Enigma irresoluble"),
    ("Destino", "Libre albedrío total", "Profecía ineludible"),
    ("Caos (Entropía)", "Orden cristalino", "Anarquía total"),
    ("Onirismo", "Realidad vigil", "Sueño lúcido psicodélico"),
    
    # --- ACCIÓN Y VERBOS ---
    ("Dificultad de Ejecución", "Automático / Sin esfuerzo", "Imposible humanamente"),
    ("Velocidad de Acción", "Letargo glaciar", "Instantáneo"),
    ("Frecuencia", "Evento único (Cisne Negro)", "Constante / Continuo"),
    ("Necesidad", "Capricho opcional", "Necesidad vital estricta"),
    ("Impacto", "Inocuo", "Devastador / Trascendental"),
    ("Control", "Accidental / Aleatorio", "Control quirúrgico absoluto"),
    ("Intencionalidad", "Involuntario", "Planificado meticulosamente"),
    ("Durabilidad del Efecto", "Efímero (segundos)", "Permanente / Irreversible")
]

# ==========================================
# 2. GENERADOR DE PROMPTS (LLAMA 3 FORMAT)
# ==========================================

def build_llama3_prompt(dim_name, anchor_0, anchor_1):
    """
    Construye el string exacto con tokens de control para Llama 3.
    Mantiene {concepto} como variable para inyectar después.
    """
    
    # 1. Definición del sistema (System Prompt)
    sys_msg = (
        f"Eres un instrumento de medición semántica calibrado para la dimensión: '{dim_name}'. "
        "Tu única función es calcular la posición relativa de un concepto entre dos extremos absolutos."
    )
    
    # 2. Instrucción del usuario (User Prompt)
    user_msg = (
        f"Analiza el concepto: '{{concepto}}'.\n\n"
        f"ESCALA DE REFERENCIA - {dim_name}:\n"
        f"[0.0000000000] -> {anchor_0} (Mínimo posible)\n"
        f"[1.0000000000] -> {anchor_1} (Máximo posible)\n\n"
        "REGLAS:\n"
        "1. Ubica el concepto en esta escala basándote en física, semántica o psicología.\n"
        "2. Sé extremadamente preciso (usa los 10 decimales).\n"
        "3. Responde SOLAMENTE con el número. Sin texto adicional.\n\n"
        f"Puntuación para '{{concepto}}':"
    )
    
    # 3. Ensamblaje con tokens especiales Llama 3
    # <|begin_of_text|> es el inicio
    # <|start_header_id|>rol<|end_header_id|> marca los bloques
    # <|eot_id|> marca el fin de turno
    full_prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{sys_msg}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    return full_prompt

# ==========================================
# 3. EJECUCIÓN PRINCIPAL
# ==========================================

def main():
    print(f"\n🧠 INICIANDO ARQUITECTURA SEMÁNTICA HUMANIZADA")
    print(f"🎯 Objetivo: Generar prompts para {len(DIMENSIONS_DB)} dimensiones.")
    print("---------------------------------------------------------------")
    
    master_json = []
    
    # Configuración visual
    total = len(DIMENSIONS_DB)
    
    for i, (name, a0, a1) in enumerate(DIMENSIONS_DB):
        # Crear ID único y limpio
        clean_name = name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        dim_id = f"d{i:03d}_{clean_name}"
        
        # Generar prompt
        prompt_template = build_llama3_prompt(name, a0, a1)
        
        # Crear objeto de datos
        dim_obj = {
            "id": dim_id,
            "category_index": i,
            "name": name,
            "scale": {"min": a0, "max": a1},
            "llama3_prompt": prompt_template
        }
        
        master_json.append(dim_obj)
        
        # --- PREVISUALIZACIÓN VISUAL (User Feedback) ---
        # Barra de progreso simple
        percent = (i + 1) / total * 100
        bar_length = 20
        filled_length = int(bar_length * (i + 1) // total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Imprimir estado actual borrando línea anterior si es posible o lineal
        sys.stdout.write(f"\r[{bar}] {percent:5.1f}% | Indexando: {name:30}")
        sys.stdout.flush()
        
        # Opcional: Imprimir detalle completo cada 10 elementos o si se quiere ver todo:
        # Aquí imprimimos una línea nueva para ver el log completo como pediste
        print(f"\n   ↳ 0.0: {a0[:40]:<40} ... 1.0: {a1[:40]}")
        
        # Pequeño delay para efecto visual (Matrix style)
        time.sleep(0.01)

    print("\n---------------------------------------------------------------")
    
    # Guardar archivo
    filename = "master_dimensions_prompts.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(master_json, f, indent=2, ensure_ascii=False)
        
    print(f"✅ PROCESO COMPLETADO.")
    print(f"📂 Archivo generado: {filename}")
    print(f"📊 Total Dimensiones: {len(master_json)}")
    print(f"💾 Tamaño aprox: {len(json.dumps(master_json))/1024:.2f} KB")
    print("\nEjemplo de prompt generado (Dimensión 0):")
    print(master_json[0]['llama3_prompt'].replace('\n', '\\n')[:100] + "...")

if __name__ == "__main__":
    main()