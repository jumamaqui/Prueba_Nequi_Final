import sys
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType
from pyspark.ml.feature import Tokenizer

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
logger = glueContext.get_logger()

output_path_predictions = "s3://nequijmmarinq/predictions/"
model_output_path = "s3://nequijmmarinq/models/msg-classifier"
test_path = "s3://nequijmmarinq/test/"

try:
    from pyspark.ml import PipelineModel
    logger.info("Cargando modelo desde S3...")
    model = PipelineModel.load(model_output_path)
    logger.info("Realizando inferencia en batch...")
    df = spark.read.parquet(test_path)
    predictions = model.transform(df)
    predictions.select("text", "prediction").write.mode("overwrite").parquet(output_path_predictions)

    logger.info("Inferencia completada y resultados guardados.")

except Exception as e:
    logger.warn(f"Modelo no encontrado o error en inferencia: {str(e)}")

logger.info("Pipeline completo.")
