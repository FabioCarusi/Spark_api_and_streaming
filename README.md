# Descrizione del task

Introduzione
------------

Il presente documento ha lo scopo di descrivere lo studio effettuato sull'utilizzo di PySpark con dati provenienti da fonti esterne:

- API pubbliche;

- Streaming.

Ambiente di sviluppo
--------------------

Lo studio è stato effettuato utilizzando il community cloud di **Databricks** che offre la possibilità di utilizzare gratuitamente il suo notebook integrato, una workspace per singolo utente e un cluster con Spark 2.4.5. \[1\]

Dati
----

I dati utilizzati per questo studio sono stati presi dal sito **data.seattle.gov**, il portale della città di Seattle dedicato alla generazione e condivisione dei dati riguardanti la città. In particolare, è stata selezionata la Categoria "Pubblic Safety" e laraccolta di dati "Seattle Real Time Fire 911 Calls" sotto forma di API\[2\].
I dati sono in "real-time" e vengono aggiornati ogni 5 minuti.
Ogni chiamata GET restituisce 1000 JSON così strutturati:

```json
{

 "address": "105 5th Av S",

 "type": "Triaged Incident",

 "datetime": "2020-03-15T00:16:00.000",

 "latitude": "47.601643",

 "longitude": "-122.327656",

 "report_location": {

 "type": "Point",

 "coordinates": [

 -122.327656,

 47.601643

 ]

 },

 "incident_number": "F200026453",

 ":@computed_region_ru88_fbhk": "27",

 ":@computed_region_kuhn_3gp2": "22",

 ":@computed_region_q256_3sug": "18379"

}
```
Fasi del progetto
-----------------

Il progetto è stato strutturato in quattro fasi:

- Acquisizione dei file json e salvataggio su directory locale;

- Creazione, analisi e lavorazione del dataframe in modalità statica;

- Utilizzo di un modello di classificazione (Randoforest tree
  classifier);

- Streaming dei dati acquisiti e query SQL in tempo reale.

Esecuzione
==========

Fase preliminare
----------------

### Acquisizione dei file Json

I file json sono stati recuperati attraverso una chiamata GET:

```python
response = request.urlopen(url)
```

e salvati in *data*:

```python
source = response.read()

data = json.loads(source)
```

### Salvataggio su directory locale

Successivamente è stata creata la struttura del filename:

```python
timestr = time.strftime('%Y%m%d-%H%M%S')
filename = '/dbfs{}{}{}'.format(path,file,timestr)
```

e fatto il dump di data per generare i diversi file json locali e
salvarli nella path:

```python
with open(filename+'-'+str(i)+extension, 'w') as f:
      json.dump(d, f)
```

Static processing
-----------------

### Creazione del dataframe

Per la creazione del dataframe ci siamo affidati a Spark utilizzando il metodo **spark.read()** che ci permette di leggere file con diverse estensioni.

Al metodo è stato chiesto di utilizzare la **path** in cui erano i file e lo **schema** che abbiamo precedentemente definito e in cui abbiamo specificato quali chiavi erano presenti nel json e a quali "tipi" appartenessero:
```python
schema = StructType([ StructField("address", StringType(), True),
                      StructField("type", StringType(), True),
                      StructField("datetime", StringType(), True),
                      StructField("latitude", StringType(), True),
                      StructField("longitude", StringType(), True),
                      StructField("report_location", StringType(), True),
                      StructField("incident_number", StringType(), True),
                      StructField(":@computed_region_ru88_fbhk", StringType(), True),
                      StructField(":@computed_region_kuhn_3gp2", StringType(), True),
                      StructField(":@computed_region_q256_3sug", StringType(), True),
                    ])
```
### Data analysis

Nella fase di analisi dei dati ci siamo concentrati sulla trasformazione dei features da stringa e numeri (interi e float) e nella sostituzione dei **null** con una media dei valori presenti in tabella.
```python
df = df.withColumn(":@computed_region_ru88_fbhk", df[":@computed_region_ru88_fbhk"].cast(IntegerType()))
df = df.withColumn(":@computed_region_kuhn_3gp2", df[":@computed_region_kuhn_3gp2"].cast(IntegerType()))
df = df.withColumn(":@computed_region_q256_3sug", df[":@computed_region_q256_3sug"].cast(IntegerType()))
df = df.withColumn("latitude", col("latitude").cast("float"))
df = df.withColumn("longitude", col("longitude").cast("float"))
```

Calcolo della media.

```python
def fill_with_mean(this_df, exclude=set()):
  stats = this_df.agg(*(fc.avg(c).alias(c) for c in this_df.columns if c not in exclude))
  return this_df.na.fill(stats.first().asDict())
```
Per il modello di Machine Learning abbiamo selezionato i 4 interventi con il maggior numero di casi presenti nella colonna **type** e trasformati da stringhe a interi. Per queste operazioni abbiamo usato SQL e il metodo **StringIndexer()** \[5\] della libreria
**pyspark.sql.functions**:

```python
df.createOrReplaceTempView('Json')
df = spark.sql("SELECT * FROM Json WHERE type in ('Aid Response', 'Medic Response', 'Trans to AMR', 'Auto Fire Alarm')")
```

```python
indexer = ft.StringIndexer(inputCol="type", outputCol="typeIndex")
indexed = indexer.fit(df_final).transform(df_final)
indexed = indexed.drop('type')
```

Infine, abbiamo creato i features da utilizzare nel modello di ML:

```python
['Latitude', 'longitude', ':@computed_region_ru88_fbhk', ':@computed_region_kuhn_3gp2', ':@computed_region_q256_3sug']
```

E, successivamente suddiviso il dataframe finale in train e test, con un rapporto 70%-30% :

```python
train, test = indexed.randomSplit([0.7, 0.3], seed = 1)
```

Modello di classificazione (Randoforest tree classifier)
--------------------------------------------------------

### Definizione del VectorAssemler

Una volta selezionati i features, abbiamo creato il VectorAssembler, un trasformatore della libreria **pyspark.ml.feature**, che combina un determinato elenco di colonne in una singola colonna vettoriale. \[4\]
Questo metodo è utile per combinare funzioni grezze e funzioni generate da trasformatori in un singolo vettore di caratteristiche, al fine di formare modelli ML come la regressione logistica e gli alberi decisionali.

```python
assem = ft.VectorAssembler(inputCols=[col for col in features], outputCol='features')## Definizione del modello
```

Il modello utilizzato per il ML è il RandomForestClassifier, che in spark.ml è utilizzato per la classificazione binaria e multiclasse e per la regressione, utilizzando sia funzionalità continue che categoriche.
\[3\]

Nell modello abbiamo specificato quale fosse la colonna di label da predire ('typeIndex'), il numero di alberi e la profondità massima del modello:

```python
rf = RandomForestClassifier(labelCol='typeIndex', numTrees=20, maxDepth=30)
```

### Definizione della Pipeline

La Pipeline è una sequenza di comandi che viene utilizzata nei processi
di Machine Learning e che permette di eseguire in sequenza tutte le
trasformazioni necessarie al modello. Nel nostro progetto abbiamo
inserito nella Pipeline il VectorAssembler e il modello, mentre abbiamo
omesso la label da predire, avendola già definita all'interno del
modello.

```python
pipeline = Pipeline(stages=\[assem, rf\])
```

Infine, abbiamo effettuato il train e il fit

```python
model = pipeline.fit(train)
predictions = model.transform(test)
```

### Valutazione del modello

La valutazione dei dati è stata fatta con un classificatore binario (BinaryClassificationEvaluator) e una matrice di confusione.

Streaming processing
--------------------

### Inizializzazione dello streaming

Prima di avviare lo streaming è necessario definire l'input di provenienza dei dati, nel nostro caso folder nella quale erano memorizzati i file json e per lo **schema**, lo stesso che abbiamo utilizzato nello "[Static processing](\l)".

Successivamente è stato creato un primo dataframe di input utilizzando il metodo ***spark.readStream()*,** lo schema, la **path** e indicando in **option** il numero di file da prendere per ogni trigger.

```python
streamingInputDF = (
  spark
    .readStream
    .schema(schema
    .option("maxFilesPerTrigger", 20)  # Treat a sequence of files as a stream by picking 20 files at a time
    .json(inputPath)
)
```
Un secondo dataframe è stato creato utilizzando il primo come imput e raggruppando i dati per 'type'. I dati sono stati raccolti utilizzando una finestra temporale di un minuto e il metodo **count()** per raggupparli.

```python
streamingCountsDF = (
  streamingInputDF
     .groupBy(
      streamingInputDF.type,
      window(streamingInputDF.datetime, "1 minute"))
      .count() 
)

```

### Avvio dello streaming

Per avviare lo streaming abbiamo definito un **sink** e l'abbiamo avviato. Il nostro obiettivo era eseguire una query interattiva dei 'type' nell'intervallo definito precedentemente. Per farlo abbiamo utilizzato il secondo dataframe come input e avviato lo streaming (writeStream) specificando il formato dell'output (**memory**), in nome della query (**jsons**) e la tipologia dell'output (**complete**).

```python
query = (
  streamingCountsDF
    .writeStream
    .format("memory")       
    .queryName("jsons")     
    .outputMode("complete")  
    .start()
)
```


### Query
Durante la fase di streaming è possibile effettuare delle query utilizzando **spark.sql()** e il **queryName()** specificato prima, nel nostro caso 'jsons'.

Le query possono essere salvate all'interno di una variabile per costruire dei dataset:

```python
df = spark.sql("SELECT * FROM jsons WHERE type in ('Aid Response', 'Medic Response', 'Trans to AMR', 'Auto Fire Alarm')")
```

Fonti:

1. <https://databricks.com/>

2. <https://data.seattle.gov/Public-Safety/Seattle-Real-Time-Fire-911-Calls/upug-ckch>

3. <https://spark.apache.org/docs/latest/ml-guide.html>

4. <https://spark.apache.org/docs/latest/ml-features#vectorassembler>

5. <https://spark.apache.org/docs/latest/ml-features#StringIndexer>
