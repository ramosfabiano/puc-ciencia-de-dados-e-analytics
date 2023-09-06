import requests
import boto3

# s3 setup
s3_client = boto3.client('s3', region_name='us-east-1')

# areronaves
req = requests.get(url='https://www.anac.gov.br/acesso-a-informacao/dados-abertos/areas-de-atuacao/aeronaves/registro-aeronautico-brasileiro/aeronaves-registradas-no-registro-aeronautico-brasileiro-csv')
s3_client.put_object(Body=req.text, Bucket='aws-glue-rawdata', Key='aeronaves.csv')

# aerodromos
req = requests.get(url='https://sistemas.anac.gov.br/dadosabertos/Aerodromos/Lista%20de%20aer%C3%B3dromos%20p%C3%BAblicos/AerodromosPublicos.csv')
s3_client.put_object(Body=req.text, Bucket='aws-glue-rawdata', Key='aerodromos.csv')

# movimentacoes
for m in range(1,13):
    m_str = f"{m:02}"
    req = requests.get(url=f'https://sistemas.anac.gov.br/dadosabertos/Operador%20Aeroportu%C3%A1rio/Dados%20de%20Movimenta%C3%A7%C3%A3o%20Aeroportu%C3%A1rias/2022/Movimentacoes_Aeroportuarias_2022{m_str}.csv')
    s3_client.put_object(Body=req.text, Bucket='aws-glue-rawdata', Key=f'movimentacoes_2022_{m_str}.csv')

