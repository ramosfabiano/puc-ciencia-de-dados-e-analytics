import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue import DynamicFrame

args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Script generated for node Data Catalog table
DataCatalogtable_node1 = glueContext.create_dynamic_frame.from_catalog(
    database="airline-db",
    table_name="airline_tbl",
    transformation_ctx="DataCatalogtable_node1",
)

# Script generated for node Change Schema
ChangeSchema_node2 = ApplyMapping.apply(
    frame=DataCatalogtable_node1,
    mappings=[
        ("passenger_id", "string", "passenger_id", "string"),
        ("gender", "string", "gender", "string"),
        ("age", "int", "age", "int"),
        ("nationality", "string", "nationality", "string"),
        ("airport_name", "string", "airport_name", "string"),
        ("airport_country_code", "string", "airport_country_code", "string"),
        ("country_name", "string", "country_name", "string"),
        ("airport_continent", "string", "airport_continent", "string"),
        ("continents", "string", "continents", "string"),
        ("departure_date", "string", "departure_date", "string"),
        ("arrival_airport", "string", "arrival_airport", "string"),
        ("flight_status", "string", "flight_status", "string"),
    ],
    transformation_ctx="ChangeSchema_node2",
)

# Script generated for node Amazon Redshift
AmazonRedshift_node3 = glueContext.write_dynamic_frame.from_options(
    frame=ChangeSchema_node2,
    connection_type="redshift",
    connection_options={
        "redshiftTmpDir": "s3://aws-glue-assets-746033461814-us-east-1/temporary/",
        "useConnectionProperties": "true",
        "dbtable": "public.airline",
        "connectionName": "redshift-connection",
        "preactions": "DROP TABLE IF EXISTS public.airline; CREATE TABLE IF NOT EXISTS public.airline (passenger_id VARCHAR, gender VARCHAR, age INTEGER, nationality VARCHAR, airport_name VARCHAR, airport_country_code VARCHAR, country_name VARCHAR, airport_continent VARCHAR, continents VARCHAR, departure_date VARCHAR, arrival_airport VARCHAR, flight_status VARCHAR);",
    },
    transformation_ctx="AmazonRedshift_node3",
)

job.commit()

