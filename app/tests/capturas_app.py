"""
Capturas de pantalla de la app para la presentacion (deliverables/fase4/capturas/).

Levanta la app con Streamlit en un puerto local, la abre con Playwright usando el Google
Chrome instalado (channel="chrome", sin descargar navegadores) en escritorio (1280 px) y
celular (390 px), y guarda formulario, resultados (bajo, medio, alto) y la seccion Premium.
Las consultas se abren con parametros en la URL (?origen=...&calcular=1).

Requisito extra (solo para capturas): pip install playwright
Uso: .venv/bin/python app/tests/capturas_app.py
"""
import os
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlencode

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "deliverables", "fase4", "capturas")
PUERTO = 8599
URL = f"http://localhost:{PUERTO}/"
VUELOS = {
    "bajo": dict(origen="HNL", destino="OGG", aerolinea="HA", fecha="2026-12-01", hora=8),
    "medio": dict(origen="ATL", destino="LGA", aerolinea="DL", fecha="2026-12-02", hora=13),
    "alto": dict(origen="DFW", destino="ORD", aerolinea="AA", fecha="2026-11-27", hora=18),
}
PANTALLAS = {"escritorio": dict(width=1280, height=900, scale=1),
             "celular": dict(width=390, height=844, scale=2)}


def levantar():
    streamlit = os.path.join(os.path.dirname(sys.executable), "streamlit")
    proc = subprocess.Popen([streamlit, "run", os.path.join(ROOT, "app", "streamlit_app.py"),
                             "--server.headless", "true", "--server.port", str(PUERTO),
                             "--browser.gatherUsageStats", "false",
                             "--client.toolbarMode", "minimal"],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen(URL + "_stcore/health", timeout=1)
            return proc
        except OSError:
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("la app no levanto")


def esperar(page, texto):
    page.get_by_text(texto, exact=False).first.wait_for(timeout=60000)
    page.wait_for_timeout(1500)      # graficas de Vega y fuentes


def pagina_completa(page, ruta, ancho):
    """Streamlit desplaza dentro del contenedor principal: se agranda la ventana a su alto."""
    alto = page.evaluate("() => { const m = document.querySelector('[data-testid=\"stMain\"]')"
                         " || document.body; return Math.max(m.scrollHeight, document.body.scrollHeight); }")
    page.set_viewport_size({"width": ancho, "height": int(alto) + 40})
    page.wait_for_timeout(800)
    page.screenshot(path=ruta, full_page=True)


def main():
    from playwright.sync_api import sync_playwright
    os.makedirs(OUT, exist_ok=True)
    proc = levantar()
    guardadas = []
    try:
        with sync_playwright() as pw:
            nav = pw.chromium.launch(channel="chrome", headless=True)
            for nombre, d in PANTALLAS.items():
                ctx = nav.new_context(viewport={"width": d["width"], "height": d["height"]},
                                      device_scale_factor=d["scale"], locale="es-MX",
                                      is_mobile=nombre == "celular", has_touch=nombre == "celular")
                page = ctx.new_page()
                # Formulario (sin calcular).
                page.goto(URL + "?" + urlencode(VUELOS["alto"]))
                esperar(page, "Estimar retraso")
                ruta = os.path.join(OUT, f"{nombre}_1_formulario.png")
                page.screenshot(path=ruta)
                guardadas.append(ruta)
                # Resultados.
                for i, nivel in enumerate(["alto", "medio", "bajo"], start=2):
                    page.set_viewport_size({"width": d["width"], "height": d["height"]})
                    page.goto(URL + "?" + urlencode({**VUELOS[nivel], "calcular": 1}))
                    esperar(page, "Mejor hora para volar")
                    ruta = os.path.join(OUT, f"{nombre}_{i}_resultado_{nivel}.png")
                    pagina_completa(page, ruta, d["width"])
                    guardadas.append(ruta)
                # Seccion Premium (vista en pantalla).
                page.set_viewport_size({"width": d["width"], "height": d["height"]})
                page.goto(URL + "?" + urlencode(VUELOS["alto"]))
                esperar(page, "Estimar retraso")
                page.get_by_text("Próximamente").first.scroll_into_view_if_needed()
                page.wait_for_timeout(800)
                ruta = os.path.join(OUT, f"{nombre}_5_premium.png")
                page.screenshot(path=ruta)
                guardadas.append(ruta)
                ctx.close()
            nav.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    for r in guardadas:
        print(f"{os.path.relpath(r, ROOT)}  {os.path.getsize(r) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
