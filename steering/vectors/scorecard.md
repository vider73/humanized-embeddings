# Scorecard de fidelidad — diales reales de la consola

Fuente: `fidelity_report_old_legacy.json` · alpha=0.15 · n_probes=2 · vectors=control_vectors_caa_white.npy · probe_version=2
Juez: `semantic_translator.pth`

Criterio: z_kin ≥ 1.5 con signo correcto. Parentesco: |r| > 0.4 en la tabla humana.

**26 diales reales de 104 medidos.**

> ⚠ Este criterio NO incluye control nulo: un dial verde aquí puede serlo también con vectores aleatorios. El veredicto con control nulo está en `scorecard_v2.md` (`python -m steering.falsify_report`); el contexto, en `STATUS.md`.

| dim | z_kin | rank_kin | signo | familia | dominio |
|---|---|---|---|---|---|
| 🟢 d099_necesidad | +4.2 | 0 | OK | 6 | sociedad |
| 🟢 d097_velocidad_de_acción | +3.6 | 0 | OK | 10 | mente |
| 🟢 d053_curiosidad | +3.6 | 0 | OK | 1 | mente |
| 🟢 d056_ira | +3.2 | 0 | OK | 16 | mente |
| 🟢 d010_dureza | +3.1 | 0 | OK | 13 | materia |
| 🟢 d024_instinto | +3.1 | 1 | OK | 8 | vida |
| 🟢 d037_sabor_amargor | +2.8 | 1 | OK | 9 | sentidos |
| 🟢 d023_consciencia | +2.7 | 1 | OK | 18 | mente |
| 🟢 d095_onirismo | +2.5 | 2 | OK | 9 | mente |
| 🟢 d101_control | +2.5 | 18 | OK | 9 | mente |
| 🟢 d046_amor | +2.4 | 1 | OK | 11 | mente |
| 🟢 d029_fertilidad | +2.4 | 6 | OK | 2 | vida |
| 🟢 d036_sabor_dulzor | +2.3 | 16 | OK | 5 | sentidos |
| 🟢 d059_culpa | +2.3 | 0 | OK | 15 | mente |
| 🟢 d044_pegajosidad | +1.9 | 14 | OK | 2 | materia |
| 🟢 d065_verdad_facticidad | +1.9 | 7 | OK | 4 | sociedad |
| 🟢 d016_volatilidad_química | +1.8 | 19 | OK | 0 | materia |
| 🟢 d077_ética_bondad | +1.7 | 6 | OK | 13 | mente |
| 🟢 d060_envidia | +1.7 | 19 | OK | 9 | mente |
| 🟢 d067_racionalidad | +1.7 | 11 | OK | 5 | mente |
| 🟢 d076_legalidad | +1.7 | 10 | OK | 10 | sociedad |
| 🟢 d082_lujo | +1.7 | 6 | OK | 9 | sociedad |
| 🟢 d022_vitalidad | +1.6 | 5 | OK | 12 | vida |
| 🟢 d039_rugosidad_táctil | +1.6 | 12 | OK | 3 | sentidos |
| 🟢 d080_artificialidad | +1.6 | 14 | OK | 0 | sociedad |
| 🟢 d043_temperatura_táctil | +1.6 | 4 | OK | 12 | sentidos |
| 🟡 d063_odio | +1.5 | 6 | OK | 15 | mente |
| 🟡 d098_frecuencia | +1.4 | 14 | OK | 14 | materia |
| 🟡 d103_durabilidad_del_efecto | +1.4 | 21 | OK | 5 | materia |
| 🟡 d015_solidez | +1.3 | 31 | OK | 3 | materia |
| 🟡 d061_soledad | +1.2 | 15 | OK | 11 | vida |
| 🟡 d092_misterio | +1.2 | 32 | OK | 1 | sociedad |
| 🟡 d051_valentía | +1.1 | 26 | OK | 6 | mente |
| 🔴 d050_orgullo | +1.0 | 43 | OK | 10 | mente |
| 🔴 d035_olor_agradabilidad | +1.0 | 48 | OK | 8 | sentidos |
| 🔴 d066_complejidad_lógica | +0.9 | 12 | OK | 12 | mente |
| 🔴 d028_salud | +0.8 | 31 | OK | 2 | vida |
| 🔴 d062_estrés | +0.7 | 31 | OK | 17 | mente |
| 🔴 d009_transparencia | +0.7 | 44 | OK | 2 | materia |
| 🔴 d021_maleabilidad | +0.7 | 65 | OK | 2 | materia |
| 🔴 d047_calma | +0.7 | 31 | OK | 0 | mente |
| 🔴 d027_edad_biológica | +0.7 | 63 | OK | 0 | vida |
| 🔴 d088_peligrosidad_social | +0.7 | 80 | OK | 10 | sociedad |
| 🔴 d068_probabilidad | +0.6 | 42 | OK | 6 | sociedad |
| 🔴 d013_fricción | +0.6 | 91 | INV | 2 | materia |
| 🔴 d002_densidad | +0.5 | 89 | OK | 8 | materia |
| 🔴 d012_viscosidad | +0.5 | 94 | OK | 1 | materia |
| 🔴 d038_sabor_picante | +0.5 | 82 | INV | 0 | sentidos |
| 🔴 d070_conocimiento_requerido | +0.5 | 32 | OK | 1 | mente |
| 🔴 d030_evolución | +0.4 | 31 | OK | 15 | vida |
| 🔴 d071_creatividad | +0.4 | 30 | OK | 4 | mente |
| 🔴 d084_religiosidad | +0.4 | 55 | OK | 5 | sociedad |
| 🔴 d025_toxicidad | +0.4 | 50 | OK | 9 | materia |
| 🔴 d074_sabiduría | +0.4 | 57 | OK | 15 | mente |
| 🔴 d045_felicidad | +0.3 | 36 | OK | 12 | mente |
| 🔴 d093_destino | +0.3 | 54 | OK | 1 | mente |
| 🔴 d096_dificultad_de_ejecución | +0.3 | 43 | OK | 7 | sociedad |
| 🔴 d017_conductividad_eléctrica | +0.2 | 95 | OK | 2 | materia |
| 🔴 d005_duración_temporal | +0.2 | 100 | INV | 1 | materia |
| 🔴 d042_brillo_superficial | +0.2 | 68 | OK | 5 | sentidos |
| 🔴 d073_claridad | +0.2 | 52 | OK | 4 | mente |
| 🔴 d008_luminosidad | +0.2 | 96 | INV | 2 | materia |
| 🔴 d018_magnetismo | +0.2 | 83 | INV | 0 | materia |
| 🔴 d001_masa | +0.1 | 100 | OK | 1 | materia |
| 🔴 d085_urbanismo | +0.1 | 51 | OK | 9 | sociedad |
| 🔴 d100_impacto | +0.1 | 41 | INV | 14 | mente |
| 🔴 d004_temperatura | +0.1 | 77 | INV | 0 | materia |
| 🔴 d089_magia | +0.1 | 39 | OK | 8 | sociedad |
| 🔴 d000_tamaño_físico | +0.1 | 102 | INV | 1 | materia |
| 🔴 d041_saturación_de_color | +0.1 | 75 | OK | 1 | sentidos |
| 🔴 d032_sonoridad_volumen | +0.0 | 79 | INV | 0 | sentidos |
| 🔴 d040_humedad | +0.0 | 42 | INV | 7 | materia |
| 🔴 d081_industrialización | +0.0 | 64 | OK | 1 | sociedad |
| 🔴 d019_radiactividad | +0.0 | 102 | INV | 0 | materia |
| 🔴 d049_confianza | -0.1 | 48 | INV | 5 | mente |
| 🔴 d058_asco | -0.1 | 37 | INV | 19 | mente |
| 🔴 d079_poder_político | -0.1 | 90 | INV | 0 | mente |
| 🔴 d057_tristeza | -0.1 | 64 | OK | 10 | mente |
| 🔴 d075_valor_económico | -0.1 | 96 | INV | 0 | sociedad |
| 🔴 d033_frecuencia_sonora | -0.1 | 72 | INV | 9 | sentidos |
| 🔴 d064_aburrimiento | -0.2 | 68 | OK | 7 | mente |
| 🔴 d054_empatía | -0.2 | 45 | OK | 16 | mente |
| 🔴 d069_abstracción | -0.3 | 56 | OK | 8 | mente |
| 🔴 d006_distancia_de_interacción | -0.3 | 91 | OK | 1 | materia |
| 🔴 d014_fragilidad | -0.3 | 65 | OK | 13 | materia |
| 🔴 d034_olor_intensidad | -0.3 | 65 | OK | 4 | sentidos |
| 🔴 d003_velocidad_máxima | -0.3 | 100 | INV | 0 | materia |
| 🔴 d078_fama | -0.4 | 78 | OK | 5 | mente |
| 🔴 d102_intencionalidad | -0.4 | 67 | INV | 5 | mente |
| 🔴 d091_futurismo | -0.4 | 30 | INV | 10 | sociedad |
| 🔴 d083_tradición | -0.4 | 37 | INV | 11 | sociedad |
| 🔴 d026_valor_nutricional | -0.6 | 28 | INV | 1 | vida |
| 🔴 d007_gravedad | -0.7 | 24 | INV | 16 | materia |
| 🔴 d055_miedo | -0.7 | 16 | INV | 16 | mente |
| 🔴 d048_esperanza | -0.7 | 54 | INV | 7 | mente |
| 🔴 d031_capacidad_sensorial | -0.8 | 80 | OK | 6 | sentidos |
| 🔴 d090_divinidad | -0.8 | 21 | INV | 6 | sociedad |
| 🔴 d052_gratitud | -1.0 | 22 | INV | 11 | mente |
| 🔴 d086_belleza_estética | -1.1 | 14 | INV | 9 | sociedad |
| 🔴 d072_memoria | -1.1 | 34 | INV | 11 | mente |
| 🔴 d011_elasticidad | -1.3 | 32 | INV | 2 | materia |
| 🔴 d094_caos_entropía | -1.5 | 13 | INV | 15 | materia |
| 🔴 d020_porosidad | -1.5 | 24 | INV | 9 | materia |
| 🔴 d087_formalidad | -3.3 | 0 | INV | 3 | sociedad |