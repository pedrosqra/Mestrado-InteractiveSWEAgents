import pandas as pd
import os

# Configuração de caminhos baseada na estrutura do repositório
input_path = 'data/underspecified.csv'
output_path = 'data/sample_30_underspecified.csv'

def generate_fixed_sample():
    """
    Realiza a amostragem aleatória simples de 30 instâncias do dataset Ambig-SWE.
    Utiliza uma semente fixa (random_state=42) para garantir a reprodutibilidade
    conforme exigido no protocolo experimental da Etapa 1.
    """
    if not os.path.exists(input_path):
        print(f"Erro: Arquivo {input_path} não encontrado! Certifique-se de rodar o script na raiz do projeto.")
        return

    # 1. Carrega o dataset original de problemas subespecificados
    print(f"Lendo dataset original: {input_path}...")
    df = pd.read_csv(input_path)
    
    # 2. Gera a amostra de 30 instâncias com semente fixa
    # O valor 42 é mantido para consistência com o planejamento do DoE
    sample_df = df.sample(n=30, random_state=42)
    
    # 3. Salva o arquivo que será o input oficial do loop de 240 rodadas
    sample_df.to_csv(output_path, index=False)
    
    print("-" * 50)
    print(f"✅ Amostra de 30 instâncias gerada com sucesso!")
    print(f"📁 Localização: {output_path}")
    print(f"🔬 Total de instâncias: {len(sample_df)}")
    print("-" * 50)
    print("IDs selecionados para o experimento pareado:")
    for idx, row in sample_df.iterrows():
        print(f" - {row['instance_id']}")

if __name__ == "__main__":
    generate_fixed_sample()
