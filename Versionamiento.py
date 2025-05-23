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

import boto3
bucket = 'nequijmmarinq'
prefix = 'test/'
s3 = boto3.client('s3')
result = s3.get_paginator('list_objects_v2')
response_iterator = result.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')
carpetas = []
for response in response_iterator:
    if 'CommonPrefixes' in response:
        for common_prefix in response['CommonPrefixes']:
            carpetas.append(common_prefix['Prefix'])
version = len(carpetas)