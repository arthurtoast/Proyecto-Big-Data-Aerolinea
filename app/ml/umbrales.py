"""
Umbrales de las banderas de clima y clima_con_banderas(). Solo biblioteca estandar, para que
app/weather.py lo use sin cargar pandas. app/ml/features.py lo reexporta.

Umbrales copiados de src/silver.py (RAIN_MM, SNOW_CM, HIGH_WIND_KMH).
"""
RAIN_MM = 1.0
SNOW_CM = 0.0
HIGH_WIND_KMH = 30.0


def clima_con_banderas(precip_mm, snow_cm, wind_max_kmh, temp_max_c, temp_min_c) -> dict:
    """Clima del origen (pronostico o promedio) -> dict listo para flight_features, con las
    banderas calculadas con las mismas reglas que Silver y has_weather=1."""
    return {"precip_mm": precip_mm, "snow_cm": snow_cm, "wind_max_kmh": wind_max_kmh,
            "temp_max_c": temp_max_c, "temp_min_c": temp_min_c,
            "rain_flag": int(precip_mm >= RAIN_MM), "snow_flag": int(snow_cm > SNOW_CM),
            "high_wind_flag": int(wind_max_kmh >= HIGH_WIND_KMH), "has_weather": 1}
