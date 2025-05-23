import sys
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from datetime import datetime
#Inicio de Contexto
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark.conf.set("mapreduce.fileoutputcommitter.algorithm.version", "1")
spark.conf.set("spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a", "org.apache.hadoop.mapreduce.lib.output.FileOutputCommitterFactory")
logger = glueContext.get_logger()
#Configuracion
input_path = "s3://nequijmmarinq/processed/clean_data/"
model_output_path = "s3://nequijmmarinq/models/msg-classifier/"
test_path = "s3://nequijmmarinq/test/"
version = "v_1.0"
trained = datetime.utcnow().isoformat()
#Leer los datos limpios 
logger.info("Leyendo datos limpios desde S3...")
df = spark.read.parquet(input_path)
df.schema
#Mismas cantidad de los datos del job anterior
df.count()
df.show()
#Target
df = df.filter(F.col("inbound").isNotNull())
#Tokenizador, convertir palabras en vectores
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
wordsData = tokenizer.transform(df)
wordsData.show()
#Hashing, proceso de convertir un vector en longitud fija
tf = HashingTF(inputCol="tokens", outputCol="rawFeatures", numFeatures=10000)
hashingTF_model = tf.transform(wordsData)
hashingTF_model.show()
#Calculo de frecuencias en las palabras
idf = IDF(inputCol="rawFeatures", outputCol="features").fit(hashingTF_model)
tfidf = idf.transform(hashingTF_model)
tfidf.show()
tfidf.select("inbound", "rawFeatures","features").show(truncate=False)
#Estas relaciones son las que entran al modelo, al pipeline 
#Se propone una regresion logistica 
lr = LogisticRegression(featuresCol="features", labelCol="inbound", maxIter=20)
#Ahora con todos esos componentes se propone el pipeline completo 
pipeline = Pipeline(stages=[tokenizer, tf, idf, lr])
#Dividir el dataset en un 80% train y 20% para la evaluación 
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
#Entrenamiento 
logger.info("Entrenando modelo...")
model = pipeline.fit(train_df)
logger.info("Evaluando modelo...")
predictions = model.transform(test_df)
evaluator = BinaryClassificationEvaluator(labelCol="inbound", rawPredictionCol="rawPrediction")
auc = evaluator.evaluate(predictions)
logger.info(f"AUC en test set: {auc}")
#logger.info("Guardando modelo entrenado en S3...")
model.write().overwrite().save(model_output_path + f"{version} +_+{trained}")
#Guardar el test para la inferencia
test_df.write.mode("overwrite").parquet(test_path+f"{version} +_+{trained}")
