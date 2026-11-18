# Actividad práctica 1 · Del dato a la primera predicción

**INACAP · Diplomado en Inteligencia Artificial · Módulo 4 · Unidad 2 · Clase 1**

## A. Ejemplos empresariales

Una regresión puede estimar la demanda semanal de un producto y devolver una cantidad numérica. Una clasificación puede asignar un pedido a las categorías «riesgo bajo» o «riesgo alto». En el primer caso, la estimación orienta cuánto reponer. En el segundo, la categoría ayuda a decidir qué pedidos revisar.

## B y C. Ejecución original y exploración

Se ejecutó [laboratorio_1.py](../../originales/laboratorio_1.py) con el entorno del proyecto. El script usa \`X.shape\` para consultar dimensiones, \`isna()\` para buscar nulos, \`duplicated()\` para identificar filas duplicadas y \`describe()\` para obtener los estadísticos descriptivos.

| Dato consultado | Resultado observado |
|---|---|
| Dimensiones de \`X\` | 400 filas y 5 columnas |
| Variables de entrada | \`indicador_1\`, \`indicador_2\`, \`indicador_3\`, \`indicador_4\` e \`indicador_5\` |
| Variable objetivo \`y\` | \`ventas_estimadas\` |
| Nulos | 0 en \`X\` y 0 en \`y\` |
| Filas duplicadas en \`X\` | 0 |
| Media de \`indicador_1\` | 0,0537 |
| Desviación estándar de \`indicador_1\` | 0,9625 |

Explorar los datos antes del entrenamiento permite detectar registros repetidos, faltantes o valores fuera de rango. Por ejemplo, una fila duplicada podría aparecer en entrenamiento y prueba e inflar el rendimiento medido.

## D. Entrenamiento y evaluación

\`train_test_split\` con \`test_size=0.2\` y semilla 42 reservó el 20 % de las filas para prueba. \`LinearRegression().fit(Xtr, ytr)\` estimó los coeficientes con las filas de entrenamiento. \`predict(Xte)\` generó las estimaciones que se compararon con los valores observados de prueba.

| Medida original | Valor | Interpretación |
|---|---:|---|
| MAE | 15,33 | La diferencia absoluta media fue de 15,33 unidades de la variable objetivo. |
| R² | 0,961 | El modelo explicó alrededor del 96,1 % de la variabilidad de la variable objetivo en esta prueba. |
| Filas de entrenamiento | 320 | Se usaron para ajustar los coeficientes. |
| Filas de prueba | 80 | Se usaron para medir el rendimiento del modelo ajustado. |

## E. Experimento controlado

Se mantuvieron la regresión lineal, los datos y la semilla 42. Entre los escenarios A y B solo cambió \`test_size\`.

| Escenario | Prueba | Entrenamiento | Prueba en filas | MAE | R² |
|---|---:|---:|---:|---:|---:|
| A · Original | 20 % | 320 | 80 | 15,3294 | 0,9607 |
| B · Exploratorio | 30 % | 280 | 120 | 14,3080 | 0,9656 |

Al volver al escenario A, la referencia predijo **9,9974** para las 80 filas de prueba, la media calculada con las 320 filas de entrenamiento.

| Predicción sobre las mismas 80 filas | MAE |
|---|---:|
| Regresión lineal | 15,3294 |
| Media de entrenamiento | 76,7753 |

La regresión superó a la referencia por 61,4459 unidades de MAE. Cambiar la proporción modifica cuántas filas intervienen en el ajuste y qué observaciones se usan para medir los errores. Aunque el escenario B obtuvo un MAE menor, la comparación describe la sensibilidad a la partición. La configuración de referencia para continuar es la original del 20 %.

## F. Recomendación gerencial

Usaría la regresión para estimar la demanda semanal y orientar la planificación de compras.  
Reuniría un historial real de ventas con períodos, unidades y definiciones claras de los indicadores.  
Mediría el error en períodos recientes y lo compararía con una referencia sencilla.  
Fijaría con el equipo un error admisible según el costo de quiebres de stock y exceso de inventario.  
Si el modelo cumple ese criterio, probaría su uso en una planificación piloto con revisión humana.

El conjunto sintético carece de unidades comerciales definidas y puede omitir patrones reales como la estacionalidad o los efectos de promociones. La Clase 2 incorpora clasificación de riesgo y agrupamiento mediante DBSCAN al proyecto.

## Preguntas de reflexión

1. **¿En qué se diferencia una predicción numérica de una etiqueta de clase?** La predicción numérica estima una magnitud, como 120 unidades. La etiqueta indica la categoría asignada, como «riesgo alto».
2. **¿Por qué separar entrenamiento y prueba?** Permite medir el error en observaciones que el modelo no usó para aprender sus coeficientes.
3. **¿Un MAE de 15 es bueno o malo?** Depende de la unidad, la escala habitual de las ventas, el error que la empresa puede tolerar y el rendimiento de una referencia comparable.
4. **¿Por qué los resultados sintéticos no prueban un beneficio económico real?** El ejercicio mide el ajuste a datos generados artificialmente. Para demostrar un beneficio habría que medir el desempeño y el impacto de las decisiones con datos y costos reales.

## Evidencias

- Ejecución original. [Registro](original.txt) · [Captura](original.png) · [Script](../../originales/laboratorio_1.py)
- Experimento y referencia de la media. [Registro](experimentos.txt) · [Captura](experimentos.png) · [Script](../../desafios/clase_01.py)
- Tablas. [Particiones](particiones.csv) · [Comparación con la media](referencia_media.csv)
