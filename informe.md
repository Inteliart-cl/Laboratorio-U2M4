# Informe de los laboratorios de Machine Learning

**INACAP · Diplomado en Inteligencia Artificial · Módulo 4 · Unidad 2**

## Datos y método

El proyecto SIAE-ML simula tres problemas empresariales: estimar ventas, clasificar riesgo y segmentar observaciones según frecuencia y valor. Los ocho scripts originales terminaron correctamente. Sus salidas quedaron en [evidencias/originales](evidencias/originales/).

| Conjunto sintético | Observaciones | Variables y objetivo |
|---|---:|---|
| Regresión | 400 | Cinco indicadores numéricos y una variable continua de ventas estimadas |
| Clasificación | 500 | Seis indicadores numéricos y la clase binaria de riesgo |
| Segmentación | 360 | Frecuencia y valor, sin etiquetas para el análisis |
| Anomalías | 372 | El conjunto de segmentación más 12 puntos inyectados |

Se conservó la semilla 42. En los desafíos supervisados se usó una partición 80/20, estratificada en clasificación. Al comparar modelos o hiperparámetros, se aplicaron cinco pliegues sobre el 80 % de entrenamiento y se evaluó la configuración elegida una vez en el 20 % de prueba. Cuando el modelo lo requería, el escalado quedó dentro de la canalización para ajustarlo con los datos de entrenamiento de cada pliegue. En los ejercicios no supervisados se trabajó con el conjunto completo.

En regresión se verificó que había cero nulos en las entradas y el objetivo, y cero duplicados en las entradas. Los generadores originales producen datos numéricos completos. Se conservaron los puntos extremos, incluidos los 12 puntos inyectados de L6, porque son parte del ejercicio de detección. Los indicadores son genéricos y las ventas simuladas carecen de una unidad monetaria definida.

## L1 Regresión y exploración

| Configuración | `fit_intercept` | `positive` | MAE CV | R² CV |
|---|---|---|---:|---:|
| Base | Sí | No | 14,3901 | 0,9665 |
| Sin intercepto | No | No | 14,3408 | 0,9664 |
| Coeficientes no negativos | Sí | Sí | 14,3901 | 0,9665 |

Quitar el intercepto redujo el MAE medio en 0,0493 unidades. Restringir los coeficientes a valores no negativos dio la misma salida a cuatro decimales porque los coeficientes del ajuste base ya fueron positivos en los cinco pliegues. Se seleccionó la regresión sin intercepto por MAE y obtuvo **15,1882 de MAE** y **0,9617 de R²** en prueba.

**Interpretación empresarial.** Un MAE de 15,19 significa que la predicción difirió del valor simulado en unas 15,19 unidades en promedio, considerando el tamaño del error sin su signo. Para usarlo en planificación de ventas habría que conocer la unidad, la escala y el margen de error admisible para la decisión.

[Ejecución L1](evidencias/desafios/L1.txt) · [Tabla CSV](evidencias/resultados/L1.csv) · [Captura](evidencias/capturas/L1.png)

## L2 Clasificación y DBSCAN

| `eps` | `min_samples` | Grupos | Observaciones marcadas como ruido |
|---:|---:|---:|---:|
| 0,28 | 5 | 3 | 0 |
| 0,15 | 5 | 3 | 3 |
| 0,45 | 5 | 3 | 0 |
| 0,28 | 10 | 3 | 0 |

Con `eps` de 0,15, tres observaciones quedaron fuera de los grupos. Duplicar el mínimo de muestras por vecindad con `eps` de 0,28 mantuvo el número de grupos y el conteo de ruido.

**¿Por qué no basta el accuracy para evaluar riesgo?** La regresión logística logró 63 % de aciertos, pero identificó 27 de 49 casos de riesgo y dejó 22 sin detectar. El *recall* de riesgo fue 55,10 %. Para decidir si el modelo serviría para priorizar revisiones, habría que considerar el costo de esos 22 falsos negativos y el de las 15 alertas sobre casos sin riesgo. DBSCAN podría apoyar una segmentación exploratoria en un problema distinto y requeriría verificar si los grupos tienen sentido con datos reales.

[Ejecución L2](evidencias/desafios/L2.txt) · [Tabla CSV](evidencias/resultados/L2.csv) · [Captura](evidencias/capturas/L2.png)

## L3 PCA, árboles y ensambles

| Modelo | `max_depth` | `min_samples_leaf` | Accuracy CV |
|---|---:|---:|---:|
| Árbol base | 4 | 1 | 78,00 % |
| Árbol más profundo | 8 | 1 | 77,25 % |
| Árbol con hojas mayores | 4 | 8 | 77,75 % |
| Bosque de 100 árboles | Sin límite | 1 | 85,75 % |

Profundizar el árbol o exigir ocho muestras por hoja redujo levemente el acierto medio respecto del árbol base. El bosque obtuvo 83,00 % en prueba. En el árbol base, `indicador_6` concentró 0,402 de la importancia por reducción de impureza. En el bosque, `indicador_5` e `indicador_6` quedaron cerca de 0,218 cada uno.

**Interpretación empresarial.** Las importancias ayudan a orientar qué variables revisar en el modelo. Por sí solas no establecen causalidad. En la segmentación, PCA explicó 70,93 % de la varianza con PC1 y 29,07 % con PC2. Como había dos variables y se conservaron las dos componentes, se retuvo el 100 % de la varianza de los datos escalados. Usar solo PC1 para resumir los perfiles descartaría el 29,07 %.

[Ejecución L3](evidencias/desafios/L3.txt) · [Tabla CSV](evidencias/resultados/L3.csv) · [Captura](evidencias/capturas/L3.png)

## L4 Sobreajuste

| `max_depth` | `min_samples_leaf` | Accuracy entrenamiento CV | Accuracy validación CV | Brecha |
|---|---:|---:|---:|---:|
| 2 | 1 | 78,06 % | 75,00 % | 3,06 pp |
| 4 | 1 | 86,06 % | 78,00 % | 8,06 pp |
| 8 | 1 | 94,44 % | 77,25 % | 17,19 pp |
| Sin límite | 1 | 100,00 % | 75,50 % | 24,50 pp |
| 4 | 8 | 83,69 % | 77,75 % | 5,94 pp |
| 8 | 8 | 87,00 % | 77,25 % | 9,75 pp |
| Sin límite | 8 | 87,37 % | 77,00 % | 10,37 pp |

**¿Qué profundidad generalizó mejor?** Entre las configuraciones probadas, profundidad 4 y mínimo de una muestra por hoja logró el mayor promedio de validación, 78,00 %. Obtuvo 79,00 % en prueba. El árbol sin límite y con mínimo de una muestra por hoja acertó todo en entrenamiento, pero bajó a 75,50 % en validación. Exigir ocho muestras por hoja redujo la brecha del árbol sin límite de 24,50 a 10,37 puntos porcentuales. Para un modelo de apoyo a decisiones, el resultado muestra por qué conviene revisar la validación al elegir cuán complejas serán las reglas.

[Ejecución L4](evidencias/desafios/L4.txt) · [Tabla CSV](evidencias/resultados/L4.csv) · [Captura](evidencias/capturas/L4.png)

## L5 Algoritmos alternativos

| Clasificador | `C` | `gamma` | Accuracy CV |
|---|---:|---|---:|
| SVM base | 1 | `scale` | 83,25 % |
| SVM con mayor `C` | 10 | `scale` | 83,75 % |
| SVM con `gamma` de 0,01 | 1 | 0,01 | 70,75 % |
| Naive Bayes | — | — | 69,75 % |
| K-NN con cinco vecinos | — | — | 84,00 % |

Aumentar `C` mejoró el acierto medio del SVM en 0,50 puntos porcentuales. Fijar `gamma` en 0,01 lo redujo en 12,50 puntos respecto del SVM base. K-NN quedó primero por un margen de 0,25 puntos sobre el mejor SVM y obtuvo 86,00 % en prueba. La diferencia de validación fue pequeña frente a la dispersión observada entre pliegues.

**¿Cuándo usar agrupamiento?** Si hay etiquetas de riesgo y se quiere predecirlas, corresponde clasificación. Si el objetivo es descubrir perfiles sin categorías previas, corresponde agrupamiento. K-Means y el método jerárquico produjeron tres grupos y un *silhouette* de 0,8843 en el espacio escalado. La [proyección t-SNE](evidencias/graficos/L5_tsne.png) muestra esos perfiles coloreados según K-Means. El *silhouette* se calculó sobre las variables escaladas originales.

[Ejecución L5](evidencias/desafios/L5.txt) · [CSV clasificación](evidencias/resultados/L5_clasificacion.csv) · [CSV agrupamiento](evidencias/resultados/L5_agrupamiento.csv) · [Captura](evidencias/capturas/L5.png)

## L6 Detección de anomalías

| `contamination` | `n_estimators` | Alertas | Puntos inyectados detectados | Alertas en datos base |
|---:|---:|---:|---:|---:|
| 0,04 | 100 | 15 | 11 de 12 | 4 |
| 0,02 | 100 | 8 | 8 de 12 | 0 |
| 0,08 | 100 | 30 | 11 de 12 | 19 |
| 0,04 | 250 | 15 | 11 de 12 | 4 |

La contaminación configurada modificó el volumen de alertas. Subirla de 4 % a 8 % duplicó las alertas y detectó el mismo número de puntos inyectados. Aumentar de 100 a 250 árboles mantuvo los conteos mostrados. La procedencia de cada observación se usó solo para interpretar la salida, después del ajuste.

**¿Una anomalía implica fraude?** La alerta indica un patrón atípico. Puede corresponder a un error de registro, un evento legítimo poco frecuente o un caso que requiera investigación. En esta simulación, la configuración de 4 % recuperó 11 de los 12 puntos inyectados con 15 alertas totales. Ese resultado ayuda a ilustrar cómo equilibrar la cobertura con la capacidad de revisión. Para un uso real habría que evaluar alertas con casos revisados.

[Ejecución L6](evidencias/desafios/L6.txt) · [Tabla CSV](evidencias/resultados/L6.csv) · [Captura](evidencias/capturas/L6.png)

## L7 Validación y optimización

| Grilla | Valores probados de `n_neighbors` | Valores probados de `weights` | Mejor configuración | Accuracy CV |
|---|---|---|---|---:|
| Base | 3, 5, 7, 9 | Uniforme | 7, uniforme | 84,25 % |
| Más vecinos candidatos | 1, 3, 5, 7, 9, 11, 13, 15 | Uniforme | 7, uniforme | 84,25 % |
| Ponderación por distancia | 3, 5, 7, 9 | Uniforme y distancia | 5, distancia | 84,25 % |

Ampliar los candidatos de `k` mantuvo la mejor configuración. Incorporar pesos por distancia produjo otra configuración con el mismo promedio. El desempate conservó la grilla base y el modelo elegido obtuvo 83,00 % en prueba.

**¿Por qué reservar el conjunto de prueba?** Si se elige el hiperparámetro según su rendimiento en prueba, ese conjunto pasa a influir en la configuración. Su puntuación deja de ser una evaluación independiente. Aquí `GridSearchCV` usó solo entrenamiento. En una evaluación empresarial, este procedimiento permite estimar el rendimiento de la configuración elegida en casos reservados.

[Ejecución L7](evidencias/desafios/L7.txt) · [Tabla CSV](evidencias/resultados/L7.csv) · [Captura](evidencias/capturas/L7.png)

## L8 Integración y recomendación

| Modelo | Árboles | Profundidad máxima | Accuracy CV | Recall de riesgo CV | Precisión de riesgo CV |
|---|---:|---|---:|---:|---:|
| Regresión logística | — | — | 71,25 % | 71,19 % | 71,17 % |
| Bosque base | 100 | Sin límite | 85,75 % | 82,31 % | 88,52 % |
| SVM | — | — | 83,25 % | 79,81 % | 85,97 % |
| Bosque con más árboles | 250 | Sin límite | 85,25 % | 81,79 % | 87,95 % |
| Bosque con profundidad 6 | 100 | 6 | 83,75 % | 77,76 % | 88,39 % |

El bosque base obtuvo el mayor acierto medio. Aumentar a 250 árboles redujo el promedio en 0,50 puntos y limitar la profundidad a 6 lo redujo en 2,00 puntos. En prueba, el bosque base acertó 83 de 100 casos. Detectó 41 de los 49 casos de riesgo y omitió 8. Generó además 9 alertas sobre casos sin riesgo. Su *recall* de riesgo fue 83,67 % y su precisión de riesgo fue 82,00 %.

**Recomendación empresarial.** Usaría el bosque de 100 árboles como candidato inicial para priorizar revisiones humanas de riesgo. Antes de aplicarlo, lo validaría con datos representativos del proceso y estimaría el costo de los casos omitidos y las alertas innecesarias. Si omitir un caso tiene un costo mayor, revisaría el umbral con datos de validación y mediría cuántas revisiones adicionales genera. Los indicadores genéricos, los datos sintéticos y el tamaño del conjunto de prueba limitan el alcance de la recomendación.

[Ejecución L8](evidencias/desafios/L8.txt) · [Tabla CSV](evidencias/resultados/L8.csv) · [Captura](evidencias/capturas/L8.png)
