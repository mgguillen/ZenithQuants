# ZenithQuants
Repositorio creado para el desarrollo del Trabajo de Fin de Maestría en Ciencia de Datos por la UOC.

**Resumen**

Superar el rendimiento del índice S&P 500 de forma consistente es un reto complejo debido a una diversidad de factores como la eficiencia y la volatilidad del mercado, los costos de transacción, la selección adecuada de los activos y los tiempos correctos para la compra y la venta. En este trabajo, trataremos de solventar algunas de estas dificultades para conseguir una estrategia de inversión avanzada basada en el análisis de datos que iniciará con una extensa recopilación de datos provenientes de fuentes abiertas, como indicadores del mercado financiero de EE. UU., datos económicos significativos y características técnicas desarrolladas mediante ingeniería de datos. Optimizaremos la selección de variables críticas mediante los métodos más apropiados, buscando las combinaciones más efectivas para maximizar el desempeño de nuestros modelos predictivos, seleccionados de un conjunto de técnicas de machine learning y deep learning. Estas herramientas nos permitirán predecir con precisión la rentabilidad de los activos y elegir mensualmente los prospectos más prometedores. La estrategia se perfeccionará a través de la optimización de la cartera, basándonos en principios económicos sólidos y se validará en un entorno de backtesting, con el objetivo de demostrar la viabilidad de superar el índice S&P 500 de forma consistente, aplicando rigurosamente ciencia de datos y análisis económico.


## Flujo de Trabajo
El proyecto se estructura en varias etapas clave, cada una gestionada por scripts específicos que facilitan la replicabilidad y comprensión del proceso:

### Configuración Inicial
- **`DataHandler.py`**: Gestiona la carga y preparación de datos históricos de precios.
- **Definición de parámetros**: Establecimiento del capital inicial, selección de ETFs y configuración del benchmark (SPY).

### Análisis Diario
Durante cada día de trading:
- **`AlphaFactors.py`**: Calcula factores alfa para identificar el potencial de rendimiento ajustado al riesgo de cada ETF.
- **`FeatureSelector.py`**: Selecciona las características más relevantes para la predicción mediante métodos como SHAP, causalidad y SelectKBest.

### Modelado y Optimización
- **`ModelBuilder.py`**: Construye y ajusta modelos predictivos utilizando técnicas como RandomForest, XGBoost, y LGBM.
- **`PortfolioOptimizer.py`**: Optimiza la asignación de capital entre los ETFs seleccionados para maximizar el ratio de Sortino.

### Ejecución y Evaluación
- **`VectorizedBacktester.py`**: Simula la estrategia de inversión, ejecutando decisiones de compra/venta según las señales del modelo y ajustando la cartera en consecuencia.
- **Evaluaciones**: Se registra el rendimiento y se ajusta la estrategia basándose en la comparación con el benchmark.

### Documentación y Análisis Final
- Se generan gráficos y análisis estadísticos para evaluar el rendimiento y ajustar la estrategia si es necesario.

## Cómo Usar
Para replicar o probar la estrategia:
1. Clonar el repositorio.
2. Instalar las dependencias necesarias.
3. Ejecutar el script `main_evaluation.py` si se quieren probar diferentes combinaciones de los elementos para iniciar el proceso de backtesting desde `2019-01-02` hasta `2020-01-04`.
4. Revisar los resultados y realizar ajustes conforme a los análisis obtenidos.
5. Ejecutar el script `main_vectorizer.py` si se quiere ejecutar el backtesting, rend es cero si nuestro objetivo es el rendimiento logaritmico y rend = 1 si queremos ejecutar la diferencia con el benchmark


Este flujo de trabajo asegura que cada etapa del proceso de inversión está bien definida y automatizada, facilitando la revisión y ajustes continuos para mejorar la estrategia de trading.