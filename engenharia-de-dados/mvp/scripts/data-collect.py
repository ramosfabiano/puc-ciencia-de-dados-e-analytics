import requests
import boto3

s3_client = boto3.client('s3', region_name='us-east-1')

req = requests.get(url=f'https://raw.githubusercontent.com/pragmaerror/puc-ciencia-de-dados-e-analytics/main/engenharia-de-dados/mvp/data/Airline_Dataset_Updated.csv')
csv_content = req.text
s3_client.put_object(Body=csv_content, Bucket='aws-glue-rawdata', Key=f'airline-tbl/Airline_Dataset_Updated.csv') 
