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
    database="movimentacoes",
    table_name="movimentacoes",
    transformation_ctx="DataCatalogtable_node1",
)

# Script generated for node Change Schema
ChangeSchema_node2 = ApplyMapping.apply(
    frame=DataCatalogtable_node1,
    mappings=[
        ("ano", "string", "ano", "string"),
        ("mes", "string", "mes", "string"),
        ("nr_aeroporto_referencia", "string", "nr_aeroporto_referencia", "string"),
        ("nr_movimento_tipo", "string", "nr_movimento_tipo", "string"),
        ("nr_aeronave_marcas", "string", "nr_aeronave_marcas", "string"),
        ("nr_aeronave_tipo", "string", "nr_aeronave_tipo", "string"),
        ("nr_aeronave_operador", "string", "nr_aeronave_operador", "string"),
        ("nr_voo_outro_aeroporto", "string", "nr_voo_outro_aeroporto", "string"),
        ("nr_voo_numero", "string", "nr_voo_numero", "string"),
        ("nr_service_type", "string", "nr_service_type", "string"),
        ("nr_natureza", "string", "nr_natureza", "string"),
        ("dt_previsto", "string", "dt_previsto", "string"),
        ("hh_previsto", "string", "hh_previsto", "string"),
        ("dt_calco", "string", "dt_calco", "string"),
        ("hh_calco", "string", "hh_calco", "string"),
        ("dt_toque", "string", "dt_toque", "string"),
        ("hh_toque", "string", "hh_toque", "string"),
        ("nr_cabeceira", "string", "nr_cabeceira", "string"),
        ("nr_box", "string", "nr_box", "string"),
        ("nr_ponte_conector_remoto", "string", "nr_ponte_conector_remoto", "string"),
        ("nr_terminal", "string", "nr_terminal", "string"),
        ("qt_pax_local", "string", "qt_pax_local", "string"),
        ("qt_pax_conexao_domestico", "string", "qt_pax_conexao_domestico", "string"),
        (
            "qt_pax_conexao_internacional",
            "string",
            "qt_pax_conexao_internacional",
            "string",
        ),
        ("qt_correio", "string", "qt_correio", "string"),
        ("qt_carga", "string", "qt_carga", "string"),
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
        "dbtable": "public.movimentacoes",
        "connectionName": "redshift_connection",
        "preactions": "CREATE TABLE IF NOT EXISTS public.movimentacoes (ano VARCHAR, mes VARCHAR, nr_aeroporto_referencia VARCHAR, nr_movimento_tipo VARCHAR, nr_aeronave_marcas VARCHAR, nr_aeronave_tipo VARCHAR, nr_aeronave_operador VARCHAR, nr_voo_outro_aeroporto VARCHAR, nr_voo_numero VARCHAR, nr_service_type VARCHAR, nr_natureza VARCHAR, dt_previsto VARCHAR, hh_previsto VARCHAR, dt_calco VARCHAR, hh_calco VARCHAR, dt_toque VARCHAR, hh_toque VARCHAR, nr_cabeceira VARCHAR, nr_box VARCHAR, nr_ponte_conector_remoto VARCHAR, nr_terminal VARCHAR, qt_pax_local VARCHAR, qt_pax_conexao_domestico VARCHAR, qt_pax_conexao_internacional VARCHAR, qt_correio VARCHAR, qt_carga VARCHAR); TRUNCATE TABLE public.movimentacoes;",
    },
    transformation_ctx="AmazonRedshift_node3",
)

job.commit()

