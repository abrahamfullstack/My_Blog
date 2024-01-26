---
title: "Entendiendo el Data Analytics"
seoTitle: "Data Analytics"
datePublished: Tue Jan 16 2024 22:48:29 GMT+0000 (Coordinated Universal Time)
cuid: clrgy3qpb000209ic5g7r0kh1
slug: entendiendo-el-data-analytics
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1705331289575/c8a506ad-dfbe-44a6-a874-28027eb7fa0d.jpeg
tags: data-analysis, data-analytics

---

## Inicio

El otro día un colega del equipo de ventas me pregunto "*¿Qué haces en tu puesto?* ". La verdad me sentí en conflicto con la pregunta pues tengo claro lo que hago día a día, mis responsabilidades e impacto de mi trabajo en la organización. Pero realmente no pude responder en sencillas palabras, pues (supongo) que otras profesiones son más fáciles de explicar o al menos todos tenemos una idea de lo que trata y quien pregunta queda conforme con la respuesta. Por ejemplo, a un arquitecto, una médico o abogado sabemos que hay respuestas cortas o ni siquiera necesitan explicarse.

Pues sí, me quede en blanco por un par de segundos y conteste: “Mi puesto es BI/Reporting Engineer”. Titulo que se me asigno en la empresa que laboro, pero en otras partes también se conoce como BI Analyst, Analytics Engineer, Data Analyst entre muchas opciones más dependiendo las actividades específicas del trabajo. Creo que mi colega no quedo conforme dada su reacción de “*aaa, suena interesante*”.

Desde entonces no pude ignorar el hecho de que no tengo una respuesta corta y sencilla para esa pregunta y dar solo el nombre del puesto no dice mucho para aquellos que no están familiarizados con el Data Analytics (análisis de datos) pero que igual les interesa saber más. Para esas personas es este artículo, tratare de no ser tan técnico y a la vez me esforzare a que los conceptos aquí mostrados sean claros y que tal vez quieran iniciar una carrera en el Data Analytics.

## Y a todo esto, ¿qué es el data analytics?

El data analytics (análisis de datos) en palabras simples es:

***El proceso de extraer información valiosa de los datos históricos disponibles sobre un objeto o evento.***

A veces la simpleza no es suficiente para entender un concepto y, para aclar un poco esta descripción veamos la definición de las partes más importantes de la sentencia anterior, ***Datos e Información***.

* **Datos** – Un dato describe una característica de un objeto. Por ejemplo, un lápiz tiene **largo** y **peso**. Para cada una de estas dos características podemos asignar un valor como: largo, 17.5 centímetros y peso 30 gramos.
    
* **Información** – Si decimos 17.5, así sin mas no sabríamos exactamente a que nos referimos, en cambio decir que un lápiz mide 17.5 centímetros de alto ya nos da más contexto de lo que hablamos. Ahora si decimos que el mismo lápiz pesa 30 gramos obtenemos información más completa. ¿Y si agregamos la dureza del carbón, o la forma del lápiz, o el color de este?
    

Entendiendo estos dos conceptos se puede deducir que a mayor cantidad de datos más información tenemos acerca de un objeto o evento. ¿Y que pasa si tenemos más datos acerca de distintos lápices?, ¿de otras marcas? Bueno, el resultado del análisis sería más relevante.

El resultado puede ser desde un reporte con graficas sencillas hasta toda una implementación de un proceso continuo de Data Analytics. También este se puede extender a utilizar tecnologías más complejas con el apoyo de **AI** (Artificial Intelligence) y sus consecuentes herramientas como el **ML** (Machine Learning) o el **DL** (Deep Learning).

El Data Analytics se puede dividir en cuatro tipos:

* **Descriptive Analytics (Análisis Descriptivo) –** Es la forma más simple de análisis que permite dar un resumen de los datos. Su gran ventaja es que la información que provee da claridad del estado de un proceso o evento. Estadísticas básicas son aplicadas en este tipo de análisis y algunos casos de uso es monitoreo continuo de un proceso o representaciones y comparación de eventos de periodos pasados contra el periodo actual.
    
* **Diagnostic Analytics (Análisis de Diagnostico) –** Enfocado en del diagnóstico de eventos considerando el desempeño para entender y rastrear porque dicho evento sucedió. Este tipo de análisis se utiliza con amplitud en el control y calidad de proceso o incluso para monitorear si cierto cambio en la compañía está entregando los beneficios esperados.
    
* **Predictive analytics (Análisis Predictivo) –** Enfocado en hacer predicciones de cuáles serían los posibles resultados de un evento.  En este tipo de análisis se aplican técnicas de Machine Learning (Aprendizaje Automático) y modelos estadísticos. Algunos casos comunes son el Forecast (Pronóstico) de eventos en periodos futuros, detección de fraudes, Clustering (Segmentacion) de clientes entre muchos más.
    
* **Prescriptive Analytics (Análisis Prescriptivo)** – Enfatiza la recomendación o prescripción de múltiples cursos de acción en función de cómo se han analizado los datos. Responde a las preguntas como ¿Qué deberíamos hacer? Al igual que el punto anterior, aquí también se empleas técnicas del AI (Inteligencia Artificial).
    

Es aquí donde reside el verdadero beneficio del Data Analytics. Dar guía a las empresas de las decisiones que hay que tomar para alcanzar sus objetivos.

## El proceso del Data Analytics

Para cualquiera de los cuatro tipos de Data Analytics se emplean pasos similares. Aquí una forma general de este proceso.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1705443907389/3f1f8070-8ede-4c24-bb55-6e2c5bb087ef.png align="center")

* **Recolección de datos** – Es la actividad para obtener acceso completo a los datos relacionados a los eventos que queremos analizar. Ya sea en libros de Excel o en una DB (Base de Datos) confirmamos el acceso y control de los datos.
    
* **Limpieza de datos** – Sabemos que la cantidad de datos importa, pero la calidad de los datos es igual de relevante. En el ejemplo de los lápices podemos deducir que un largo razonable de un lápiz es de 17 cm y para fines de un análisis significativo, si encontramos un dato como 170 cm en la base de datos puede afectar el análisis. Por ejemplo, si queremos saber el promedio de altura de los lápices registrados con los siguientes valores:
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1705444067710/cb288f61-e412-4d32-b8c6-e8a9821ef297.png align="center")
    
    El promedio de estos valores es 33.46. Pero si retiramos los valores Outliers (Atípicos):
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1705444581017/3a9862a2-92dc-4e93-9c26-0f182cdb6ff8.png align="center")
    
    El promedio de la altura es de 17.45 lo cual parece más lógico. En practica el proceso de limpieza de datos puede llegar a ser bastante complejo y consumir una cantidad considerable de tiempo en el proyecto, pero es una parte fundamental para que el resultado del análisis sea relevante.
    
* **Transformación de datos** – El enfoque principal de esta actividad es realizar la manipulación y conversión de los datos a formatos que sean aceptados por las herramientas de análisis de datos. Además, en esta etapa se puede iniciar con la creación de variables que desde el recurso original de datos no se encuentran. Por ejemplo, variables de tiempo tales como: cuanto se tardó en cerrar una venta desde que se contactó al cliente hasta la entrega del producto.
    
* **Data Analytics** – Una vez que se cuenta con los datos completos y limpios con las transformaciones necesarias hacemos el uso de las herramientas especificas dependiendo del tipo de análisis que queremos generar (Descriptive, Diagnostic, Predictive, Prescriptive). Se hace uso de herramientas graficas para la representación de los datos.
    
* **Presentación de conclusiones** – A pesar de que los pasos anteriores pueden ser muy complicados cuando nos iniciamos en el Data Analytics lo cierto es que mucho de ese esfuerzo puede verse mermado si no transmitimos eficientemente los resultados de este. No importa si nuestra solución es muy sofisticada, con gráficos intuitivos y descubrimiento de patrones espectaculares, si no atrapamos la atención de quien recibe nuestras conclusiones. Como lectura adicional, recomiendo altamente el libro de Story Telling With Data.
    
* **Toma de decisiones** – Si las conclusiones fueron apropiadamente transmitidas, los interesados del análisis (directores, gerentes, supervisores, CEOs, etc.) pueden iniciar con la toma de decisiones para lo cual el Data Analyst estará al tanto de esas conversaciones en caso de que análisis extras sean requeridos.
    

Aquí expliqué de forma muy general lo que es el Data Analytics y su proceso. Pero es verdad que este escrito es muy básico y no pretende convertirte un experto en el tema. Espero además que mis palabras solo sirvan de orientación para adentrarse a las profundidades de cada elemento y herramienta que se utilizan. Es por ello que también ofrezco una breve lista de lo que se necesita para iniciar.

## Herramientas básicas para el Data Analytics

Esta es una lista de las herramientas y conceptos básicos con la que todos los lectores pueden comenzar en el Data Analytics. En algunos casos podrás encontrar recursos gratuitos con tan solo un click y algunos tienen la opción del idioma español. En algunos casos se repiten las herramientas ya que todas ellas tienen múltiples funciones.

### Fundamentos básicos

* Estadísticas
    
* Probabilidad
    
* Programación básica, especialmente en el lenguaje [Python](https://www.w3schools.com/python/default.asp)
    

### No completamente necesario al inicio, pero recomendado

* Algebra lineal
    
* Calculo
    

### Herramientas para programar

* [Anaconda](https://www.anaconda.com/)
    
* [Google Colab](https://colab.research.google.com/) o [Jupyter Lab](https://jupyter.org/)
    

### Recolección de datos

* Conceptos de bases de datos (estructura, tipos, sistemas disponibles)
    
* [SQL (Structure Query Language)](https://www.w3schools.com/sql/default.asp)
    

### Limpieza de datos

* Excel
    
* [Power BI](https://www.microsoft.com/en-us/power-platform/products/power-bi/) o [Tableau](https://www.tableau.com/why-tableau/what-is-tableau)
    
* En Python, [Matplot](https://www.w3schools.com/python/matplotlib_intro.asp) y [Pandas](https://www.w3schools.com/python/pandas/default.asp)
    

### Transformación de datos

* En Excel: [Power Query](https://powerquery.microsoft.com/en-us/)
    
* En Power BI: [Power Query](https://powerquery.microsoft.com/en-us/)
    
* [SQL (Structure Query Language)](https://www.w3schools.com/sql/default.asp)
    

### Data Analytics y Estadísticas

* Con Python: [Numpy](https://www.w3schools.com/python/numpy/default.asp), [Sklearn](https://www.w3schools.com/statistics/index.php), [Scipy](https://www.w3schools.com/python/scipy/index.php)
    

### Visualizaciones

* Excel
    
* [Power BI](https://www.microsoft.com/en-us/power-platform/products/power-bi/) o [Tableau](https://www.tableau.com/why-tableau/what-is-tableau)
    

# Inicia con lo que tienes

Como conclusión para aquellos que quieren iniciar en el Data Analytics mi consejo es que a la par de que estudien un tema busquen la manera de aplicar esos conocimientos con datos reales. Ya sea que seas un profesionista que tiene acceso a las herramientas más básicas o un estudiante de medicina. Recordemos que hoy en día tenemos acceso a casi todo en internet y muchas de las empresas comparten información histórica con la que podemos experimentar. Un ejemplo al alcance de todos los lectores son los reportes financieros de Yahoo!, los cuales son reales y tienen un significado en la economía global.

La idea es buscar aquí y allá, para encontrar oportunidades de experimentar y ganar experiencia, aunque no tengas estudios a fin de las tecnologías de la información. Además, hay muchos recursos gratuitos en línea y otros que no lo son, pero son accesibles si así lo prefieres.

Espero que este articulo sea de ayuda.