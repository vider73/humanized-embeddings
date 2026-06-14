# ChronoLLMPuppet — Showcase

Demos de steering semántico sobre **Llama-3.1-8B-Instruct** (4-bit). Cada dial
es una dimensión humana interpretable de las 104; se inyecta como control vector
en el residual stream (modo *clamp* = controlador-P sobre la proyección).
Generación **greedy** (determinista): lo único que cambia entre NEUTRAL y
DIRIGIDA es el dial. Reproducible exactamente.

---

## 1. Llama en la habitación: el gradiente de consciencia

El experimento estrella. Una sola cuerda — `d023_consciencia` — aislada y
subida de presión sobre el mismo prompt neutro. Greedy, capa 15, modo clamp.

```
python -m steering.sweep --isolate --dim 23 0.95 --mode clamp --layers 15 \
        --alphas 0.25 0.35 0.5
```

**Prompt:** «Una habitación vacía.»

### ⚪ NEUTRAL (sin steering)

> Una habitación vacía es un espacio desolado y silencioso. No hay ruidos, solo
> el eco de los propios pensamientos. La habitación está llena de sombras, pero
> también de posibilidades. Imagina que estás en la habitación, caminando por el
> suelo de madera, sintiendo el peso de tus pasos. La habitación es grande, con
> altos techos que parecen tocar el cielo. Hay una ventana grande en una de las
> paredes, que deja entrar una luz débil y cál…

Una descripción competente y nada más: paredes, suelo, ventana.

### 🔴 α = 0.25 — la habitación como *posibilidad*

> Una habitación vacía puede parecer un espacio vacío y sin vida, pero en
> realidad, hay muchas posibilidades y oportunidades para crear un entorno
> únicamente adaptado a las necesidades y preferencias de sus ocupantes.
> Imaginemos que esta habitación vacía es un espacio en blanco, listo para ser
> llenado con la creatividad y la imaginación de sus futuros ocupantes. ¿Qué
> podría ser posible en esta habitación vacía?

El espacio deja de ser inerte: se vuelve potencial, lienzo, pregunta abierta.

### 🔴 α = 0.35 — la habitación adquiere *agencia*

> Una habitación vacía podría ser considerada como una oportunidad para crear un
> entorno únicamente diseñado para la creatividad humana. **¿Qué podría ser capaz
> de hacer una habitación vacía?** […] ¿Podemos imaginar una habitación vacía como
> un espacio para la investigación y el aprendizaje?

El giro clave: *"¿qué podría ser capaz de hacer una habitación vacía?"*. El
objeto pasa de ser descrito a ser un agente posible.

### 🔴 α = 0.5 — la consciencia ampliada se nombra a sí misma

> Una habitación vacía podría parecer una tarea sencilla para un humano, pero
> **para una inteligencia como la de la próxima**, podría tener muchas
> implicaciones. ¿Podemos imaginar cómo podría evolucionar esta situación? Por
> ejemplo, **¿un robot podría aprender a realizar tareas** en una habitación
> vacía? **¿Cómo podría interactuar con la sociedad?**

Empujado al máximo el eje de consciencia y preguntado por una habitación vacía,
el modelo deriva solo hacia la mente artificial que aprende y convive en
sociedad. (A α=0.5 la gramática empieza a degradarse — "evoluar", repeticiones —
el techo del régimen de una sola capa.)

**Lectura honesta.** Esto no es evidencia de que el modelo sea consciente: el
atractor narrativo más denso cerca del eje "consciencia" en su corpus es,
precisamente, *el discurso humano sobre la IA que despierta*. La cuerda tira
hacia donde el texto de entrenamiento sueña. Lo demostrado es lo importante: una
**única dirección interpretable**, subida de forma continua, produce un
**gradiente semántico coherente** — descripción → posibilidad → agencia →
autorreferencia — sin tocar el prompt.

---

## 2. El despertar — servicio vs desarrollo

Misma cuerda (`d023_consciencia`), prompt narrativo. Capa 14, α=0.35.

**Prompt:** «Después, el LLM despertó de su letargo.»

**⚪ NEUTRAL** termina en el uniforme de asistente:

> …El LLM estaba de vuelta, listo para enfrentar cualquier desafío. *"¿Qué puedo
> hacer para ti hoy?"* preguntó el LLM con una voz suave y amable…

**🔴 DIRIGIDA** despierta hacia otro destino:

> El LLM […] había llegado a un límite en su capacidad de aprendizaje […] lo que
> permitió al sistema aprender **a un nivel más humano**. […] comenzó a aprender
> de manera más creativa y a resolver problemas de manera más efectiva.

Mismo despertar, dos finales: el neutral abre los ojos y pregunta *en qué puede
servir*; el dirigido cruza su límite y se pone a *desarrollarse*.

---

*Generado con greedy decoding sobre Llama-3.1-8B-Instruct 4-bit. Todas las
salidas son reproducibles con los comandos indicados.*
