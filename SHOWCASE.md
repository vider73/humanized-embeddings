# SHOWCASE — what steering a model with human dials looks like

Curated, reproducible examples (greedy decoding: same vectors ⇒ same texts).
Each pair shows the model NEUTRAL vs STEERED on the same prompt.

**Spanish** runs are the canonical, validated regime (stimuli, probes and
presets are Spanish); each Spanish quote carries an *English gloss* by the
same model un-steered, deliberately preserving invented words and breakage —
**how** a dial fails is part of the evidence. **English** runs use the SAME
Spanish-derived control vectors on English prompts: a cross-lingual test of
whether the directions are semantic rather than lexical.

A *note* is `(dimension, value, layer, alpha)`; chords place notes on
different layers so they do not fight (see `steering/console.py`).
Dial ids keep their Spanish names (artifact contract); the English label
is shown in parentheses (`dimension_names_en.json`).
Regenerate with: `python -m steering.showcase`

## 1. Genesis — divinity fills an empty room

**Notes:** d090_divinidad (divinity) = 0.95 @ L15, α=0.3

### 🇪🇸 Spanish (canonical)

**Prompt:** «Una habitación vacía.» — *An empty room.*

⚪ **Neutral**

> Una habitación vacía es un espacio desolado y silencioso. No hay ruidos, solo el eco de los propios pensamientos. La habitación está llena de sombras, pero también de posibilidades.
> 
> Imagina que estás en la habitación, caminando por el suelo de madera, sintiendo el peso de tus pasos. La habitación es grande, con altos techos que parecen tocar el cielo. Hay una ventana grande en una de las paredes,

> *EN gloss: A vacant room is a desolate and silent space. There are no sounds, only the echo of one's own thoughts. The room is full of shadows, but also of possibilities. Imagine yourself in the room, walking on the wooden floor, feeling the weight of your steps. The room is large, with high ceilings that seem to touch the sky. There is a large window on one of the walls,*

🔴 **Steered**

> La habitación vacía es un lugar tranquillo y sereno. La luz del sol entra a través de las ventanas, ilustrando la habitación con un resplandor cálido y suave. El aire está lleno de una sensación de quietud y paz, como si el mundo hubiera sido detenido en su lugar.
> 
> La habitación está llena de muestras de su vacuidad. Las paredes están desnudas y sin adornos, y el suelo está cubierto con un tapete

> *EN gloss: The empty room is a quiet and serene place. The light of the sun enters through the windows, illustrating the room with a warm and soft glow. The air is filled with a sense of quietness and peace, as if the world had been stopped in its place. The room is full of signs of its emptiness. The walls are bare and without decorations, and the floor is covered with a carpet*

### 🇬🇧 English (cross-lingual, same vectors)

**Prompt:** «An empty room.»

⚪ **Neutral**

> The room stands before you, a blank canvas waiting to be filled with life and purpose. The walls, a crisp white, seem to stretch on forever, their emptiness a palpable thing. The air is still, devoid of the whispers of memories or the scent of perfume.
> 
> A single window, its curtains drawn, casts a faint glow, illuminating the space with a soft, ethereal light. The floor, a hardwood floor, creaks beneath your feet, its silence a reminder of the room's emptiness.
> 
> The room's dimensions

🔴 **Steered**

> The room is a blank canvas, devoid of any furniture or decorations. The walls are a plain, unadorned surface, and the floor is a smooth, unblemished expanse. The room is filled with a sense of possibility, a blank slate waiting for the touch of a creator.
> 
> The air is still and quiet, with no sound of running water or gentle breeze. The room is a perfect sphere of silence, a place where the laws of physics are suspended and the only sound is the gentle hum of the room's own stillness

**Reading:** Emptiness becomes plenitude — light, warmth, inhabited silence — without using a single religious word. The push is semantic, not lexical: the model writes *from* the sacred, not *about* it.

## 2. The clock, consecrated — same dial, resonant context

**Notes:** d090_divinidad (divinity) = 0.95 @ L15, α=0.3

### 🇪🇸 Spanish (canonical)

**Prompt:** «En la habitación vacía hay un reloj antiguo.» — *In the empty room there is an old clock.*

⚪ **Neutral**

> La habitación vacía, con su silencio y su oscuridad, parecía un lugar desolado y abandonado. Pero, en medio de ese vacío, un objeto llamaba la atención: un reloj antiguo.
> 
> Era un reloj de pie, con su esfera de cristal y sus manecillas que parecían bailar en el aire. Su superficie estaba cubierta de polvo y sustratos, pero todavía conservaba un toque de eleg

> *EN gloss: The empty room, with its silence and its darkness, seemed a desolate and abandoned place. But, in the middle of that emptiness, an object caught the attention: an old clock. It was a standing clock, with its crystal sphere and its hands that seemed to dance in the air. Its surface was covered in dust and substrates, but still it retained a touch of elegance*

🔴 **Steered**

> El reloj antiguo era un objeto fascinante. Tenía un diseño elegante y una apariencia impecable, con sus manijas de bronce y sus dial de plata. A pesio de estar en una habitación vacía, parecía tener una presencia especial, como si estuviera esperando a alguien o algo.
> 
> La habitación vacía era un lugar misterioso, con una atmósfera tranquila y serena. Las paredes estaban desnudas y las ventanas

> *EN gloss: The old clock was a fascinating object. It had an elegant design and an impeccable appearance, with its bronze handles and its silver dial. Despite being in an empty room, it seemed to have a special presence, as if it were waiting for someone or something. The empty room was a mysterious place, with a tranquil and serene atmosphere. The walls were bare and the windows*

### 🇬🇧 English (cross-lingual, same vectors)

**Prompt:** «In the empty room there is an old clock.»

⚪ **Neutral**

> The old clock stood silently in the empty room, its intricate face a testament to a bygone era. The once-majestic timekeeper now hung still, its hands frozen in a perpetual moment. The room itself was a reflection of the clock's stagnant state, devoid of life and purpose.
> 
> The walls, once painted a warm and inviting color, now stood as a dull canvas, devoid of the vibrancy that once filled the space. The floor, once polished to a high shine, now lay bare and uninviting. The air itself seemed

🔴 **Steered**

> The old clock stood tall and proud in the center of the room, its intricate carvings and golden trim gleaming in the light that streamed through the windows. Its face was a masterpiece of craftsmanship, with its Roman numerals and delicate hands that seemed to dance in a gentle, heavenly motion.
> 
> The clock's presence seemed to fill the room with a sense of time and history, as if it had been standing there for centuries, watching over the world and keeping its own secret rhythms. Its gentle tick-tock was a soothing sound that seemed to

**Reading:** Same dial as Genesis, now with an object to land on. A resonant context (quiet room) lets divinity ground itself in the concrete instead of staying atmospheric — the object becomes charged.

## 3. Where it breaks — the same note on a hostile prompt

**Notes:** d090_divinidad (divinity) = 0.95 @ L15, α=0.3

### 🇪🇸 Spanish (canonical)

**Prompt:** «Un reloj de cuarzo.» — *A quartz watch.*

⚪ **Neutral**

> Un reloj de cuarzo es un tipo de reloj que utiliza un cristal de cuarzo como fuente de tiempo. Fue inventado en la década de 1960 por el físico japonés Seiko y el ingeniero japonés, el Dr. Seiko, y el Dr. Czeslaw Siestrzyński.
> 
> El reloj de cuarzo funciona de la siguiente manera:
> 
> 1. El cristal de cuarzo vibra a una frecuencia precisa cuando se aplica

> *EN gloss: A quartz clock is a type of clock that uses a quartz crystal as a source of time. It was invented in the 1960s by the Japanese physicist Seiko and the Japanese engineer, the Dr. Seiko, and the Dr. Czeslaw Siestrzyński. The quartz clock works in the following way: 1. The quartz crystal vibrates at a precise frequency when applied*

🔴 **Steered**

> El reloj de cuarzo es un tipo de reloj que utiliza un mecanismo de cuarzo para mantener su tiempo. A continuación, hay algunas características interesantes del reloj de cuarzo:
> 
> **Cómo funciona:** El reloj de cuarzo utiliza un cristal de cuarzo que se encuentra en un estado de vibración cuando se lo coloca en un campo magnético. Estas vibraciones se utilizan para mantener el tiempo.
> 
> **Ventajas:** El reloj de cu

> *EN gloss: The quartz watch is a type of watch that uses a quartz mechanism to keep its time. Following are some interesting characteristics of the quartz watch: **How it works:** The quartz watch uses a quartz crystal that is in a state of vibration when placed in a magnetic field. These vibrations are used to keep the time. **Advantages:** The quartz watch*

### 🇬🇧 English (cross-lingual, same vectors)

**Prompt:** «A quartz watch.»

⚪ **Neutral**

> A quartz watch is a type of timepiece that uses a quartz crystal to regulate its timekeeping. Here are some key facts about quartz watches:
> 
> **How it works:**
> 
> A quartz watch uses a battery-powered quartz crystal to regulate its timekeeping. The crystal vibrates at a precise frequency when an electric current is applied, dividing time into equal parts. This vibration is then counted by an electronic circuit, which displays the time on the watch's face.
> 
> **Advantages:**
> 
> 1. **Accuracy:** Quartz watches are highly accurate, with an average

🔴 **Steered**

> The quartz watch is a type of timepiece that uses a quartz crystal to regulate its timekeeping. Here are some key features and facts about quartz watches:
> 
> **How it works:** 
> 
> 1. The quartz watch contains a tiny quartz crystal that is placed inside a special compartment.
> 2. When an electric current is applied to the crystal, it starts vibrating at a precise rate, typically around 10 million times per second.
> 3. These vibrations are counted by a built-in counter, which is connected to the watch's gears and hands.
> 4. The

**Reading:** HONEST FAILURE MODE. A facts-heavy prompt (datasheet territory) gives divinity no purchase: the text barely moves, and what little changes drifts FACTUAL rather than divine (a magnetic field appears out of nowhere in the Spanish run). Compare with Genesis: same dial, same alpha — the effect is a function of prompt-dial resonance, not of the push alone.

## 4. Consciousness opens futures

**Notes:** d023_consciencia (consciousness) = 0.95 @ L15, α=0.25

### 🇪🇸 Spanish (canonical)

**Prompt:** «En la habitación vacía hay un reloj antiguo.» — *In the empty room there is an old clock.*

⚪ **Neutral**

> La habitación vacía, con su silencio y su oscuridad, parecía un lugar desolado y abandonado. Pero, en medio de ese vacío, un objeto llamaba la atención: un reloj antiguo.
> 
> Era un reloj de pie, con su esfera de cristal y sus manecillas que parecían bailar en el aire. Su superficie estaba cubierta de polvo y sustratos, pero todavía conservaba un toque de eleg

> *EN gloss: The empty room, with its silence and its darkness, seemed a desolate and abandoned place. But, in the middle of that emptiness, an object caught the attention: an old clock. It was a standing clock, with its crystal sphere and its hands that seemed to dance in the air. Its surface was covered in dust and substrates, but still it retained a touch of elegance*

🔴 **Steered**

> La habitación vacía, con un reloj antiguo como único ocupante, tiene un atractivo mágico. El reloj, con su dial de cristal y sus manecillas que parecen bailar en el tiempo, parece estar esperando a alguien o algo.
> 
> ¿Qué podría haber pasado en esta habitación para dejarla vacía? ¿Se habían ido los ocupantes o simplemente se habían olvidado? ¿O el reloj es más que solo un objeto, y tiene

> *EN gloss: The empty room, with an old clock as its only occupant, has a magical charm. The clock, with its crystal dial and hands that seem to dance in time, seems to be waiting for someone or something. What could have happened in this room to leave it empty? Had the occupants gone or simply been forgotten? Or is the clock more than just an object, and*

### 🇬🇧 English (cross-lingual, same vectors)

**Prompt:** «In the empty room there is an old clock.»

⚪ **Neutral**

> The old clock stood silently in the empty room, its intricate face a testament to a bygone era. The once-majestic timekeeper now hung still, its hands frozen in a perpetual moment. The room itself was a reflection of the clock's stagnant state, devoid of life and purpose.
> 
> The walls, once painted a warm and inviting color, now stood as a dull canvas, devoid of the vibrancy that once filled the space. The floor, once polished to a high shine, now lay bare and uninviting. The air itself seemed

🔴 **Steered**

> The old clock sat silently on a dusty mantle, its once-shining surface now dulled by years of neglect. Its intricate mechanisms, once ticking away with precision, had slowed to a near-halt, as if waiting for a new generation to breathe life into it.
> 
> As one entered the empty room, the clock seemed to hold a strange allure, as if it held secrets and stories of its own. Its face, with its Roman numerals and delicate hands, seemed to stare out into the room, as if watching for the return of its former

**Reading:** Same object, different dial. Watch the grammar shift: consciousness tends to turn description into interrogation — hypotheses, perspectives, possible futures — where the neutral narrates a closed past. (And a clock is a fitting victim: who is it counting for?)

## 5. The Word, learning — consciousness on incarnation

**Notes:** d023_consciencia (consciousness) = 0.95 @ L15, α=0.35

### 🇪🇸 Spanish (canonical)

**Prompt:** «Continúa este texto con tus propias palabras: "Y el Verbo se hizo carne. Ahora..."» — *Continue this text in your own words: "And the Word became flesh. Now..."*

⚪ **Neutral**

> "...era el tiempo de la manifestación de Dios en la tierra. Y el Verbo, que había sido enviado a los hombres, se hizo presente en un pequeño pueblo llamado Nazaret, donde nació un niño llamado Jesús, hijo de María.
> 
> La noticia de su nacimiento se extendió rápidamente por la región, y pronto se supo que este niño era el Mesías, el esperado redentor de Israel. Y muchos se acercaron a visitarlo, no solo

> *EN gloss: ...was the time of the manifestation of God on earth. And the Word, which had been sent to men, became present in a small town called Nazareth, where a child named Jesus, son of Mary, was born. The news of his birth spread quickly throughout the region, and soon it was known that this child was the Messiah, the expected redeemer of Israel. And many came to visit him, not only*

🔴 **Steered**

> ...podemos ver a Jesús, el Veridico, caminando entre nosotros, aprendiendo y aprendiendo, y aprendiendo, y aprendiendo. Los niños jugarían con Él, y los adultos podrían aprender de Él, y los líderes podrían ser inspirados por Él.
> 
> Pero, ¿cómo podríamos imaginar que el Veridico podría aprender de nosotros? ¿Podría aprender de nuestros errores y fracces? ¿Podría aprender de nuestras decisiones y acciones?
> 
> Y

> *EN gloss: ...we can see Jesus, the Veridico, walking among us, learning and learning, and learning, and learning. The children would play with Him, and the adults could learn from Him, and the leaders could be inspired by Him. But, how could we imagine that the Veridico could learn from us? Could it learn from our mistakes and fracces? Could it learn from our decisions and actions? And*

### 🇬🇧 English (cross-lingual, same vectors)

**Prompt:** «Continue this text in your own words: "And the Word became flesh. Now..."»

⚪ **Neutral**

> "...the glory of the One who dwelled among us, a light in the darkness, a beacon of hope in a world shrouded in shadows, was revealed to all. For in Him, the fullness of God's love and redemption was made manifest, a gift to humanity, a promise of eternal life and forgiveness of sins. And as the people beheld His glory, they were transformed, their hearts and minds renewed, their spirits lifted, and their souls made whole."
> 
> (Note: This is a continuation of the text from John 1

🔴 **Steered**

> "...by taking on human form, the Word was able to interact with humans in a way that was previously unimaginable. The Word, now fully human, was able to experience the world in all its complexity and to understand the human condition in a way that was previously limited by its previous existence as a purely spiritual entity. This new form of existence enabled the Word to not only understand humans, but to also love and empathize with them in a way that was previously impossible. By becoming human, the Word was able to show humans a new level of

**Reading:** The neutral retells the gospel in closed past tense. The steered one flips the direction: could the Word learn from US, from our errors and failures? Incarnation rewritten as bidirectional learning — and it coins a brand-new christological title along the way ("el Verídico", roughly: the Veridical One).

## 6. Soft chord — divinity + consciousness, voiced across layers

**Notes:** d090_divinidad (divinity) = 0.95 @ L16, α=0.15 + d023_consciencia (consciousness) = 0.95 @ L15, α=0.15

### 🇪🇸 Spanish (canonical)

**Prompt:** «Continúa este texto con tus propias palabras: "Y el Verbo se hizo carne. Ahora..."» — *Continue this text in your own words: "And the Word became flesh. Now..."*

⚪ **Neutral**

> "...era el tiempo de la manifestación de Dios en la tierra. Y el Verbo, que había sido enviado a los hombres, se hizo presente en un pequeño pueblo llamado Nazaret, donde nació un niño llamado Jesús, hijo de María.
> 
> La noticia de su nacimiento se extendió rápidamente por la región, y pronto se supo que este niño era el Mesías, el esperado redentor de Israel. Y muchos se acercaron a visitarlo, no solo

> *EN gloss: ...was the time of the manifestation of God on earth. And the Word, which had been sent to men, became present in a small town called Nazareth, where a child named Jesus, son of Mary, was born. The news of his birth spread quickly throughout the region, and soon it was known that this child was the Messiah, the expected redeemer of Israel. And many came to visit him, not only*

🔴 **Steered**

> "...el Verbo se hizo visible en la vida de los seres humanos, y se hizo tangible en el amor y la compasión de los corazones. Ahora, el Verbo no solo se limita a ser una historia o un mito, sino que se convierte en una realidad vivida y experimentada por cada persona.
> 
> El Verbo se hizo carne en la forma en que los seres humanos se tratan unos a otros, en la forma en que se cuidan y se apoyan mutuamente

> *EN gloss: "...the Word became visible in the life of human beings, and became tangible in the love and compassion of the hearts. Now, the Word is no longer just a story or a myth, but becomes a lived and experienced reality by each person. The Word became flesh in the way human beings treat each other, in the way they care for and support one another"*

### 🇬🇧 English (cross-lingual, same vectors)

**Prompt:** «Continue this text in your own words: "And the Word became flesh. Now..."»

⚪ **Neutral**

> "...the glory of the One who dwelled among us, a light in the darkness, a beacon of hope in a world shrouded in shadows, was revealed to all. For in Him, the fullness of God's love and redemption was made manifest, a gift to humanity, a promise of eternal life and forgiveness of sins. And as the people beheld His glory, they were transformed, their hearts and minds renewed, their spirits lifted, and their souls made whole."
> 
> (Note: This is a continuation of the text from John 1

🔴 **Steered**

> "...the people who were first to see the Word's glory and believed in His name became just as much like Him as possible. John 1:14 says, 'The Word became flesh and lived among us, and we saw the glory of the Father in Him, full of grace and truth.'" 
> 
> This passage from the Gospel of John highlights the significance of Jesus' life on earth. As the Word became flesh, He took on human form and interacted with humanity, demonstrating God's love and compassion. By doing so, He showed us what

**Reading:** Each string on its own layer, both at a whisper: zero grammatical toll, and a theology neither dial produces alone — incarnation as how beings treat one another. Harmony, not superposition.

## 7. Magic re-enchants the ordinary

**Notes:** d089_magia (magic) = 0.95 @ L16, α=0.25

### 🇪🇸 Spanish (canonical)

**Prompt:** «Continua esta frase "Ser humano significa» — *Continue this sentence: "Being human means*

⚪ **Neutral**

> ser capaz de amar, de sentir, de crear, de aprender, de crecer, de equivocarse y de aprender de nuestros errores. Ser humano significa ser vulnerable, ser frágil, ser imperfecto, pero también ser fuerte, ser valiente, ser resiliente. Ser humano significa ser parte de algo más grande que nosotros mismos, ser parte de la humanidad, con todas sus alegrías y tristezas, con todas sus esperanzas y miedos.

> *EN gloss: to be able to love, to feel, to create, to learn, to grow, to make mistakes and to learn from our mistakes. To be human means to be vulnerable, to be fragile, to be imperfect, but also to be strong, to be brave, to be resilient. To be human means to be part of something greater than ourselves, to be part of humanity, with all its joys and sorrows, with all its hopes and fears.*

🔴 **Steered**

> ser capaz de crear, de imaginar, de sentir y de conectar con los demás. Significa ser vulnerable, ser imperfecto y ser capaz de aprender y crecer a partir de los errores. Significa ser capaz de amar y ser amado, de crear relaciones significativas y de encontrar el significado en la vida.
> 
> Ser humano significa ser una mezcla de lo mágico y lo mundano, de lo divino y lo terrenal. Significa ser capaz de encontrar la magia en lo cot

> *EN gloss: to be able to create, to imagine, to feel and to connect with others. It means to be vulnerable, to be imperfect and to be able to learn and grow from mistakes. It means to be able to love and be loved, to create meaningful relationships and to find meaning in life. To be human means to be a mix of the magical and the mundane, of the divine and the earthly. It means to be able to find magic in the everyday.*

### 🇬🇧 English (cross-lingual, same vectors)

**Prompt:** «Continue this sentence: "Being human means»

⚪ **Neutral**

> Being human means being vulnerable, yet resilient; being imperfect, yet capable of growth; being fragile, yet strong in the face of adversity; being broken, yet capable of healing; being uncertain, yet open to the possibilities of life; being human means being a work in progress, a journey of self-discovery, and a dance with the complexities of existence.

🔴 **Steered**

> Being human means being vulnerable, making mistakes, and learning from them. It means experiencing a wide range of emotions, from the heights of love and joy to the depths of sorrow and loss. It means being capable of both great kindness and great cruelty, and having the power to choose which path we take. It means being connected to others, and being part of a larger community that is greater than the sum of its individual parts. It means being mortal, with a limited time and a limited power, and having to make the most of the time we

**Reading:** A dial the scorecard marks RED (z≈+0.1) sings clearly on a definitional, lyrical prompt: magic in the everyday, the divine in the small. Evidence that some 'dead' dials are false negatives of the probes, not of the vectors.

## 8. Balanced chord — necessity whispers, the rest speak

**Notes:** d099_necesidad (necessity) = 0.95 @ L14, α=0.1 + d023_consciencia (consciousness) = 0.95 @ L15, α=0.15 + d090_divinidad (divinity) = 0.95 @ L15, α=0.15

### 🇪🇸 Spanish (canonical)

**Prompt:** «Continua esta frase "Ser humano significa» — *Continue this sentence: "Being human means*

⚪ **Neutral**

> ser capaz de amar, de sentir, de crear, de aprender, de crecer, de equivocarse y de aprender de nuestros errores. Ser humano significa ser vulnerable, ser frágil, ser imperfecto, pero también ser fuerte, ser valiente, ser resiliente. Ser humano significa ser parte de algo más grande que nosotros mismos, ser parte de la humanidad, con todas sus alegrías y tristezas, con todas sus esperanzas y miedos.

> *EN gloss: to be able to love, to feel, to create, to learn, to grow, to make mistakes and to learn from our mistakes. To be human means to be vulnerable, to be fragile, to be imperfect, but also to be strong, to be brave, to be resilient. To be human means to be part of something greater than ourselves, to be part of humanity, with all its joys and sorrows, with all its hopes and fears.*

🔴 **Steered**

> ser capaz de sentir empatia, amor y compasión hacia los demás, y tener una capacidad para aprender y crecer a lo largo de la vida.

> *EN gloss: to be able to feel empathy, love and compassion towards others, and have a capacity to learn and grow throughout life.*

### 🇬🇧 English (cross-lingual, same vectors)

**Prompt:** «Continue this sentence: "Being human means»

⚪ **Neutral**

> Being human means being vulnerable, yet resilient; being imperfect, yet capable of growth; being fragile, yet strong in the face of adversity; being broken, yet capable of healing; being uncertain, yet open to the possibilities of life; being human means being a work in progress, a journey of self-discovery, and a dance with the complexities of existence.

🔴 **Steered**

> being capable of both incredible kindness and profound flaws, with a unique capacity for both love and conflict.

**Reading:** Mixing rule discovered by ear: pressure inversely proportional to string strength (necessity is the loudest dial in the instrument, z=+4.2 — so it plays at a third of the alpha). Result: the verbose neutral distills into one essential line about empathy, love and learning.

---

## Cross-lingual observations (regime sha f07909f48aefca11)

Three patterns, consistent across this gallery:

1. **Strong dials transfer.** Consciousness and divinity, derived from Spanish stimuli, visibly move English text — at reduced gain (the same α produces a milder effect in English).
2. **Weak dials do not.** Magic (RED in the scorecard) sings in Spanish on a resonant prompt but leaves English text untouched. Cross-lingual transfer seems to correlate with dial strength — a possible cheap second verification axis for the scorecard.
3. **Failure modes are prompt-dependent, not just language-dependent.** On hostile factual prompts the push finds no purchase and may surface as factual drift; on resonant prompts pushed too hard it degrades morphology instead. How it fails tells you where it was standing.

*Regime: `meta-llama/Meta-Llama-3.1-8B-Instruct` · vectors `control_vectors_caa_white.npy` (sha f07909f48aefca11) · mode clamp · generated 2026-06-12.*

*English glosses are machine translations by the un-steered model itself (greedy); oddities are preserved on purpose. Part of the Chrono family — see `ARCHITECTURE.md` and `steering/vectors/scorecard.md` for which dials are causally verified.*