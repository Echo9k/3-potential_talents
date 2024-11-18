### Proyecto de NLP: Detección y Clasificación de Talento

---

#### **Contexto**

Como empresa de reclutamiento y gestión de talento, nuestro objetivo es encontrar personas altamente capacitadas para cubrir roles en compañías tecnológicas. Esta tarea es desafiante por varias razones:

1. **Comprensión del rol**: Para encontrar al candidato ideal, debemos entender muy bien las necesidades y expectativas del cliente para el puesto en cuestión.
2. **Identificación de habilidades clave**: Es esencial reconocer las cualidades que hacen destacar a un candidato específico en cada rol.
3. **Ubicación del talento**: Encontrar fuentes donde se concentran estos candidatos también es un reto.

El proceso actual implica un gran esfuerzo manual. Para automatizar y optimizar este trabajo, buscamos desarrollar una solución que ahorre tiempo y nos permita identificar candidatos potenciales de manera más precisa. Con el tiempo, queremos construir un sistema de aprendizaje automático que no solo encuentre a los candidatos, sino que también los clasifique en función de qué tan bien se ajustan al rol.

Actualmente, estamos en una etapa en la que seleccionamos candidatos de manera semi-automatizada. Por lo tanto, el primer enfoque será clasificar a los candidatos en función de su ajuste para el puesto. Este proceso se basa en la búsqueda de palabras clave como "ingeniero de software full-stack", "gerente de ingeniería" o "aspirante a recursos humanos", y las palabras clave cambiarán según el rol.

Después de listar y clasificar a los candidatos, realizamos una **revisión manual** para evaluar qué tan bien encajan en el rol. Durante esta revisión, podríamos decidir que el candidato más adecuado no sea el primero de la lista, sino quizás el séptimo. En este caso, es importante que el sistema pueda **reordenar** la lista basándose en esta señal supervisada (al "marcar" a este séptimo candidato como ideal para el rol). Esperamos que cada vez que marquemos a un candidato, la lista se reorganice para reflejar mejor las preferencias para el puesto.

---

#### **Descripción de los Datos**

La información de los candidatos proviene de nuestros procesos de selección y ha sido anonimizada para proteger su privacidad. A cada candidato se le asignó un identificador único.

- **id**: Identificador único del candidato (*numérico*).
- **job_title**: Título del trabajo del candidato (*texto*).
- **location**: Ubicación geográfica del candidato (*texto*).
- **connections**: Número de conexiones que tiene el candidato (500+ significa más de 500, *texto*).
- **fit**: Nivel de ajuste del candidato para el rol, entre 0 y 1 (*numérico, probabilidad*).
- **keywords**: Palabras clave asociadas como “aspirante a recursos humanos” o “buscando recursos humanos”.

---

#### **Objetivos del Proyecto**

- **Objetivo Principal**: Predecir el nivel de ajuste de los candidatos (variable *fit*) basado en su información.
  
- **Métricas de Éxito**:
  - Clasificar a los candidatos en función de su puntaje de ajuste (*fit*).
  - Reordenar la lista de candidatos cada vez que uno sea marcado como ideal.

---

#### **Desafíos Actuales**

1. **Desarrollo de un Algoritmo Robusto**: Queremos un algoritmo confiable. Necesitamos una explicación clara de cómo funciona la solución y cómo mejora la clasificación después de cada acción de marcado.
  
2. **Filtrado de Candidatos No Aptos**: ¿Cómo podemos filtrar a los candidatos que no deberían estar en la lista desde el principio?

3. **Definición de un Punto de Corte**: ¿Es posible establecer un umbral que funcione para otros roles sin perder a candidatos de alto potencial?

4. **Reducción del Sesgo Humano**: ¿Tienes ideas para automatizar aún más el proceso y reducir el sesgo humano en la selección?

