import argparse
from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
EVIDENCIAS = ROOT / "evidencias"
ENTORNO = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def guardar_captura(ruta, encabezado, comando, salida):
    fuente = Path(find_spec("matplotlib").origin).parent / "mpl-data" / "fonts" / "ttf"
    mono = ImageFont.truetype(str(fuente / "DejaVuSansMono.ttf"), 17)
    titulo = ImageFont.truetype(str(fuente / "DejaVuSans-Bold.ttf"), 23)
    lineas = salida.rstrip().splitlines()
    ancho = max(850, int(max(map(mono.getlength, lineas), default=0)) + 56)
    alto = 122 + 25 * len(lineas)
    imagen = Image.new("RGB", (ancho, alto), "#101820")
    dibujo = ImageDraw.Draw(imagen)
    dibujo.rectangle((0, 0, ancho, 67), fill="#1b2b38")
    dibujo.text((26, 19), encabezado, font=titulo, fill="#f4f7fb")
    dibujo.text((26, 80), f"> {comando}", font=mono, fill="#9cd8f2")
    for i, linea in enumerate(lineas):
        dibujo.text((26, 108 + i * 25), linea, font=mono, fill="#e2ebf3")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    imagen.save(ruta)


def registrar(argumentos, carpeta_trabajo, entorno, registro):
    proceso = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", *argumentos],
        cwd=carpeta_trabajo,
        env=entorno, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8",
    )
    registro.parent.mkdir(parents=True, exist_ok=True)
    registro.write_text(proceso.stdout, encoding="utf-8")
    print(f"{registro.relative_to(ROOT)}: código {proceso.returncode}", flush=True)
    if proceso.returncode:
        print(proceso.stdout, file=sys.stderr)
        raise SystemExit(proceso.returncode)
    return proceso.stdout


def ejecutar(laboratorio, etapa, entorno):
    modulo = f"laboratorio_{laboratorio}"
    registro = EVIDENCIAS / etapa / f"L{laboratorio}.txt"
    if etapa == "originales":
        registrar([f"{modulo}.py"], ROOT / "originales", entorno, registro)
    else:
        salida = registrar(["-m", f"desafios.{modulo}"], ROOT, entorno, registro)
        guardar_captura(
            EVIDENCIAS / "capturas" / f"L{laboratorio}.png",
            f"SIAE-ML | Laboratorio {laboratorio}", f"python -m desafios.{modulo}", salida,
        )


def ejecutar_actividad(numero, entorno):
    clase = f"clase_{numero:02d}"
    carpeta = EVIDENCIAS / clase
    original = f"laboratorio_{numero}.py"
    modulo = f"desafios.{clase}"
    salida = registrar([original], ROOT / "originales", entorno, carpeta / "original.txt")
    guardar_captura(
        carpeta / "original.png", f"SIAE-ML | Clase {numero} · Ejecución original",
        f"python {original}", salida,
    )
    salida = registrar(["-m", modulo], ROOT, entorno, carpeta / "experimentos.txt")
    guardar_captura(
        carpeta / "experimentos.png", f"SIAE-ML | Clase {numero} · Experimentos",
        f"python -m {modulo}", salida,
    )


def main():
    parser = argparse.ArgumentParser(description="Ejecuta los laboratorios y guarda su salida.")
    parser.add_argument("laboratorios", metavar="N", type=int, nargs="*",
                        help="números de laboratorio, del 1 al 8")
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--solo-desafios", action="store_true")
    modo.add_argument("--actividades", action="store_true", help="guarda las evidencias de las clases 1 y 2")
    opciones = parser.parse_args()
    disponibles = range(1, 3) if opciones.actividades else range(1, 9)
    laboratorios = opciones.laboratorios or disponibles
    if any(numero not in disponibles for numero in laboratorios):
        parser.error(f"Los números deben estar entre 1 y {disponibles[-1]} para este modo.")
    with TemporaryDirectory(prefix="siae-ml-") as cache:
        entorno = {**ENTORNO, "MPLCONFIGDIR": cache}
        if opciones.actividades:
            for numero in laboratorios:
                ejecutar_actividad(numero, entorno)
            return
        if not opciones.solo_desafios:
            for numero in laboratorios:
                ejecutar(numero, "originales", entorno)
        for numero in laboratorios:
            ejecutar(numero, "desafios", entorno)


if __name__ == "__main__":
    main()
