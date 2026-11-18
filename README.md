# Laboratorio U2M4 SIAE-ML

Prácticas 1 a 8 del Módulo 4, Unidad 2, del diplomado INACAP. El [informe](informe.md) contiene el análisis de los ocho desafíos y sus conclusiones.

## Ejecutar en Windows

Abra esta carpeta en VS Code y use la terminal integrada de PowerShell. El proyecto ya incluye un entorno .venv preparado con Python 3.12. Para ejecutar los laboratorios, use directamente su intérprete.

~~~
.\.venv\Scripts\python.exe .\ejecutar.py
~~~

Si va a preparar el proyecto en otro equipo y allí aún no existe .venv, créelo con Python 3.12. Con el iniciador de Python para Windows instalado, use estos comandos.

~~~
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements-verificados.txt
~~~

El ejecutor lanza los ocho scripts originales y luego los ocho desafíos. Para ejecutar solo el desafío 4 y guardar su evidencia:

```powershell
.\.venv\Scripts\python.exe .\ejecutar.py 4 --solo-desafios
```

Desde esta misma carpeta también puede ejecutar un desafío directamente con `.\.venv\Scripts\python.exe -m desafios.laboratorio_4`. Ese comando imprime las métricas y actualiza el CSV correspondiente. El ejecutor agrega el registro de consola y la imagen de la salida.

## Actividades formativas de las clases 1 y 2

La [Actividad 1](evidencias/clase_01/actividad.md) compara dos particiones y la regresión con una referencia que predice la media de entrenamiento. La [Actividad 2](evidencias/clase_02/actividad.md) incluye el reporte de clasificación y los cuatro experimentos DBSCAN solicitados en la guía.

Para volver a generar las tablas, los registros y las capturas de ambas actividades, ejecute:

~~~powershell
.\.venv\Scripts\python.exe .\ejecutar.py --actividades
~~~

Para ejecutar solo una, agregue el número de clase al comando.

## Archivos y evidencias

| Ubicación | Contenido |
|---|---|
| [originales](originales/) | Copia del ZIP del curso con `base.py`, dependencias y scripts L1 a L8. |
| [desafios](desafios/) | Scripts de los ocho desafíos. Importan los generadores originales de datos sintéticos. |
| [evidencias/originales](evidencias/originales/) | Salidas completas de los scripts originales. |
| [evidencias/desafios](evidencias/desafios/) | Salidas completas de los desafíos. |
| [evidencias/capturas](evidencias/capturas/) | Imágenes PNG generadas desde la salida real de consola de cada desafío. |
| [evidencias/resultados](evidencias/resultados/) | Tablas CSV con los parámetros y las métricas de comparación. |
| [evidencias/graficos](evidencias/graficos/) | Proyección t-SNE de la práctica 5. |

El caso y los datos de la Actividad Sumativa N.º 2 se entregan por separado.
