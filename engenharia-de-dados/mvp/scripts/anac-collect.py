import requests
import boto3

# s3 setup
s3_client = boto3.client('s3', region_name='us-east-1')

# movimentacoes
for m in range(1,13):
    m_str = f"{m:02}"
    req = requests.get(url=f'https://sistemas.anac.gov.br/dadosabertos/Operador%20Aeroportu%C3%A1rio/Dados%20de%20Movimenta%C3%A7%C3%A3o%20Aeroportu%C3%A1rias/2022/Movimentacoes_Aeroportuarias_2022{m_str}.csv')
    
    csv_content = req.text
    csv_lines = csv_content.split('\n')
    
    if (csv_lines[0].startswith("Atualizado")):
        csv_lines.pop(0)  
        csv_content = '\n'.join(csv_lines)
        
    s3_client.put_object(Body=csv_content, Bucket='aws-glue-rawdata', Key=f'movimentacoes/2022_{m_str}.csv')    

