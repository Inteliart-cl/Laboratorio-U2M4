# Actividad práctica 2 · Clasificación y DBSCAN

**INACAP · Diplomado en Inteligencia Artificial · Módulo 4 · Unidad 2 · Clase 2**

## A y B. Ejecución original y clasificación supervisada

Se ejecutó [laboratorio_2.py](../../originales/laboratorio_2.py) con el entorno de la Clase 1. El conjunto de clasificación tiene 500 registros. \`X\` contiene los seis indicadores numéricos \`indicador_1\` a \`indicador_6\`. La variable \`y\` contiene la clase de \`riesgo\`, codificada como 0 o 1.

\`test_size=0.2\` reservó 100 filas para prueba y dejó 400 para entrenamiento. \`stratify=y\` mantuvo aproximadamente las proporciones de ambas clases en la división y \`random_state=42\` permitió reproducirla. El \`Pipeline\` ajustó \`StandardScaler\` con las filas de entrenamiento, aplicó esa escala a las de prueba y entrenó la regresión logística con las etiquetas conocidas.

| Categoría | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Clase 0 | 0,6207 | 0,7059 | 0,6606 | 51 |
| Clase 1 | 0,6429 | 0,5510 | 0,5934 | 49 |
| Promedio macro | 0,6318 | 0,6285 | 0,6270 | 100 |
| Promedio ponderado | 0,6316 | 0,6300 | 0,6276 | 100 |

La exactitud global fue **63 %**. Para la clase de riesgo, la precisión indica qué fracción de las alertas correspondió a casos de riesgo. El *recall* indica qué fracción de todos los casos de riesgo fue detectada. Un *recall* alto en la clase 1 significa que se omiten pocos casos de riesgo. Aquí se detectaron 27 de 49 y quedaron 22 sin detectar. F1 es la media armónica de precisión y *recall*. El promedio macro da el mismo peso a cada clase y el ponderado usa el soporte de cada clase.

En un conjunto desbalanceado, acertar la clase mayoritaria puede producir una exactitud alta aunque se omita gran parte de la clase de riesgo. \`max_iter=1000\` establece el máximo de iteraciones del optimizador de la regresión logística. La precisión se calcula después de predecir.

## C. Agrupamiento con DBSCAN

\`Z\` contiene 360 observaciones con dos características numéricas, \`frecuencia\` y \`valor\`. DBSCAN recibe esas características sin etiquetas de clase. \`eps\` fija el radio de la vecindad y \`min_samples\` indica el mínimo de muestras en ella, incluida la propia observación, para que sea un punto central. La etiqueta \`-1\` identifica ruido bajo la configuración usada.

Estandarizar hace comparables las escalas de las dos variables para calcular distancias. Una observación marcada como ruido puede ser un caso legítimo poco común, un error de registro o un caso que requiera revisión. Para establecer fraude se necesitan antecedentes adicionales.

## D. Experimentos controlados

Cada fila compara una configuración con la original. Se mantuvieron los datos, la estandarización y la semilla 42. En A y B se varió solo \`eps\`. En C se repuso \`eps=0,28\` y se varió solo \`min_samples\`.

| Experimento | \`eps\` | \`min_samples\` | Grupos | Registros de ruido |
|---|---:|---:|---:|---:|
| Original | 0,28 | 5 | 3 | 0 |
| A | 0,20 | 5 | 3 | 0 |
| B | 0,45 | 5 | 3 | 0 |
| C | 0,28 | 8 | 3 | 0 |

Los cuatro ensayos dieron tres grupos y cero registros de ruido. La estabilidad observada corresponde a esos conteos en los parámetros ensayados. El script original conserva \`eps=0,28\` y \`min_samples=5\`.

## E. Preguntas empresariales

1. **¿Qué modelo usaría para cada necesidad?** Usaría regresión logística para estimar el riesgo a partir de casos etiquetados. Usaría DBSCAN para explorar segmentos por frecuencia y valor cuando todavía se desconocen las categorías.
2. **¿Qué agrega el reporte de clasificación?** Mide los aciertos y errores frente a categorías conocidas, por clase. El conteo de grupos resume la estructura detectada por densidad y cuántas observaciones quedaron como ruido.
3. **¿Qué cambios observé al variar los parámetros?** Con \`eps\` de 0,20 y 0,45 se mantuvieron los tres grupos y cero puntos de ruido. Subir \`min_samples\` a 8 con el radio original produjo los mismos conteos.
4. **¿Qué riesgos tiene tratar el ruido como fraude?** Podría acusar o bloquear operaciones legítimas, aumentar el trabajo de revisión y tomar decisiones adversas con evidencia insuficiente. La alerta puede iniciar una revisión humana que contraste la operación con antecedentes del negocio.
5. **¿Qué decisión tomaría y qué limita esta prueba?** Probaría usar las alertas de riesgo para ordenar una cola de revisión, y los grupos para explorar perfiles de comportamiento. Validaría ambas salidas con casos reales, porque los patrones sintéticos y sus proporciones pueden diferir de los de la empresa.

## Evidencias

- Ejecución original. [Registro](original.txt) · [Captura](original.png) · [Script](../../originales/laboratorio_2.py)
- Experimentos. [Registro](experimentos.txt) · [Captura](experimentos.png) · [Script](../../desafios/clase_02.py)
- Tablas. [Métricas de clasificación](clasificacion.csv) · [Configuraciones DBSCAN](dbscan.csv)
