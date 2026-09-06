# Model Card — Predicción de abandono de clientes

## 1. Información general

| Campo | Descripción |
|---|---|
| Nombre | Modelo de predicción de abandono de clientes |
| Versión | 1.0.0 |
| Tipo de modelo | Regresión Logística L2 balanceada |
| Framework | scikit-learn |
| Variable objetivo | `Churn` |
| Tipo de problema | Clasificación binaria |
| Autora | Valesca Fuenzalida |
| Estado | Modelo productivo del proyecto académico |

## 2. Objetivo

El modelo estima la probabilidad de que un cliente abandone una empresa de telecomunicaciones.

Su propósito es apoyar la priorización de acciones de retención, identificando clientes con mayor riesgo de abandono.

La predicción se interpreta de la siguiente manera:

- `0`: no se genera alerta de abandono.
- `1`: se genera alerta de posible abandono.

## 3. Uso previsto

El modelo puede utilizarse para:

- Priorizar clientes para campañas de retención.
- Generar listados de alertas de riesgo.
- Apoyar el análisis comercial.
- Comparar segmentos con distintas probabilidades de abandono.
- Servir como base para una aplicación o sistema de scoring.

El resultado debe considerarse como una herramienta de apoyo y no como una decisión automática definitiva.

## 4. Usos no recomendados

El modelo no debería utilizarse para:

- Suspender o limitar servicios automáticamente.
- Discriminar clientes.
- Tomar decisiones sin revisión del contexto comercial.
- Aplicarse directamente en otra empresa sin validación.
- Interpretar la probabilidad como una relación causal.
- Entrenar nuevamente utilizando el conjunto TEST para ajustar decisiones.

## 5. Datos utilizados

Fuente: [Telco Customer Churn de Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

Características principales:

- Registros originales: 7.043
- Registros después de la limpieza: 7.032
- Variable objetivo: `Churn`
- Predictores: 19
- Prevalencia de abandono: 26,58 %

La variable `customerID` se excluye del entrenamiento y se utiliza solamente como identificador.

## 6. División de los datos

| Conjunto | Registros | Finalidad |
|---|---:|---|
| TRAIN | 4.922 | Entrenamiento y aprendizaje |
| VALIDATION | 1.055 | Comparación de modelos y selección del umbral |
| TEST | 1.055 | Evaluación final independiente |

La división se realizó de manera estratificada.

Se verificó que no existieran coincidencias de `customerID` entre TRAIN, VALIDATION y TEST.

## 7. Variables del modelo

### Variables numéricas

- `tenure`
- `MonthlyCharges`
- `TotalCharges`

### Variables categóricas

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`

## 8. Preprocesamiento

El modelo forma parte de un `Pipeline` de scikit-learn.

Para las variables numéricas se aplica:

1. Imputación mediante la mediana.
2. Estandarización con `StandardScaler`.

Para las variables categóricas se aplica:

1. Imputación mediante la categoría más frecuente.
2. Codificación con `OneHotEncoder`.
3. Manejo de categorías nuevas con `handle_unknown="ignore"`.

El preprocesamiento se ajusta únicamente con los datos correspondientes al entrenamiento.

## 9. Modelos evaluados

Se compararon los siguientes modelos:

- DummyClassifier.
- Regresión Logística L2 balanceada.
- Random Forest balanceado.
- Gradient Boosting.

La selección se realizó utilizando exclusivamente el conjunto VALIDATION.

## 10. Criterio de selección

El problema presenta desbalance de clases y el objetivo de negocio es detectar la mayor cantidad posible de clientes que realmente abandonarán.

Por esta razón se priorizaron:

1. Recall.
2. Balanced Accuracy.
3. F1.
4. ROC-AUC y PR-AUC.

Se seleccionó la Regresión Logística L2 balanceada porque obtuvo en VALIDATION:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.7479 |
| Balanced Accuracy | 0.7681 |
| Precision | 0.5170 |
| Recall | 0.8114 |
| F1 | 0.6316 |
| ROC-AUC | 0.8531 |
| PR-AUC | 0.6542 |

## 11. Umbral operativo

El umbral se seleccionó utilizando únicamente el conjunto VALIDATION.

Criterio:

- Recall mínimo de 0,80.
- Maximización de Precision dentro de esa restricción.

Umbral seleccionado:

```text
0.5256
```

Resultados del umbral en VALIDATION:

| Precision | Recall | F1 |
|---:|---:|---:|
| 0.5294 | 0.8007 | 0.6374 |

## 12. Entrenamiento final

Una vez seleccionado el modelo y el umbral, el pipeline final fue reentrenado con la unión de TRAIN y VALIDATION.

Registros utilizados:

```text
5.977
```

El conjunto TEST permaneció aislado hasta la evaluación final.

## 13. Rendimiento final en TEST

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.7374 |
| Balanced Accuracy | 0.7392 |
| Precision | 0.5036 |
| Recall | 0.7429 |
| F1 | 0.6003 |
| ROC-AUC | 0.8224 |
| PR-AUC | 0.5835 |

### Matriz de confusión

| Resultado real | Predice: No abandona | Predice: Abandona |
|---|---:|---:|
| No abandona | 570 | 205 |
| Abandona | 72 | 208 |

Interpretación:

- Se detectaron correctamente 208 de 280 abandonos reales.
- El Recall final fue de 74,29 %.
- Se generaron 205 falsos positivos.
- No se detectaron 72 clientes que finalmente abandonaron.

## 14. Consideraciones éticas

El modelo utiliza características personales y contractuales, por lo que sus resultados deben manejarse responsablemente.

Recomendaciones:

- No utilizar género o condición de adulto mayor para generar un trato perjudicial.
- No ejecutar decisiones adversas automáticamente.
- Revisar posibles diferencias de rendimiento entre grupos.
- Restringir el acceso a los datos personales.
- Mantener trazabilidad de las predicciones.
- Utilizar las alertas para ofrecer apoyo o beneficios de retención.

## 15. Limitaciones

- Los datos corresponden a un dataset público.
- No existe una dimensión temporal suficiente para medir cambios de comportamiento.
- El modelo no demuestra causalidad.
- El rendimiento puede cambiar al aplicarlo a nuevos clientes.
- La precisión de 50,36 % implica que aproximadamente la mitad de las alertas positivas podrían ser falsos positivos.
- El umbral deberá revisarse si cambian los costos o prioridades del negocio.
- No se realizó una evaluación de equidad detallada entre grupos demográficos.

## 16. Riesgos

Los principales riesgos son:

- Falsos positivos que generen campañas innecesarias.
- Falsos negativos que dejen clientes en riesgo sin intervención.
- Cambios en los patrones de clientes.
- Categorías nuevas no observadas durante el entrenamiento.
- Uso del modelo fuera del contexto para el cual fue desarrollado.
- Interpretación incorrecta de la probabilidad como certeza.

## 17. Monitoreo recomendado

### Métricas operativas

- Número de predicciones.
- Latencia de predicción.
- Porcentaje de alertas.
- Distribución de probabilidades.
- Valores faltantes.
- Categorías desconocidas.
- Errores durante la ejecución.

### Métricas de rendimiento

Cuando se conozca el resultado real:

- Recall.
- Precision.
- F1.
- Balanced Accuracy.
- ROC-AUC.
- PR-AUC.
- Matriz de confusión.

### Deriva de datos

Se recomienda comparar periódicamente:

- Distribución de `tenure`.
- Distribución de `MonthlyCharges`.
- Distribución de `TotalCharges`.
- Tipos de contrato.
- Servicios de internet.
- Métodos de pago.
- Prevalencia real de abandono.

## 18. Criterios de revisión

El modelo debería revisarse cuando ocurra alguna de las siguientes condiciones:

- Disminución relevante del Recall.
- Aumento sostenido de falsos positivos.
- Cambios importantes en las variables de entrada.
- Incorporación de nuevos productos o contratos.
- Cambio en la estrategia de retención.
- Existencia de un volumen suficiente de datos nuevos.

## 19. Artefactos

- Pipeline local: `models/modelo_logistico_churn_v1.joblib`
- Metadatos: `models/metadata_modelo_v1.json`
- Métricas finales: `reports/metricas_test_final.csv`
- Predicciones: `reports/predicciones_test_final.csv`
- Registro MLflow: `telco_churn_model`, versión 1

El modelo `.joblib` se excluye del repositorio y debe generarse mediante el script de entrenamiento.

## 20. Reproducibilidad

Para reproducir el modelo:

```powershell
python scripts\preprocess.py
python scripts\train_model.py
python scripts\predict.py
```

Las dependencias se encuentran declaradas en `requirements.txt`.