# Criar dataframe a partir do csv
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/gcmor/OneDrive/Área de Trabalho/datasets IC/Uncovering the causes of emotions in software developer communication using zero-shot llms/github-train.csv")


# ver primeiras linhas do dataframe
print(df.head())


# Selecionar as colunas binárias que representam as emoções
emotion_columns = ['Anger', 'Love', 'Fear', 'Joy', 'Sadness', 'Surprise']

# Contar os valores de 'sim' (1) e 'não' (0) para cada emoção
binary_counts = df[emotion_columns].apply(lambda x: x.value_counts()).T

# Substituir valores NaN por 0, caso haja colunas sem 0s ou 1s
binary_counts.fillna(0, inplace=True)

# Plotar o histograma empilhado
binary_counts.plot(kind='bar', stacked=True)
plt.title('Distribuição de Sim (1) e Não (0) nas Emoções')
plt.xlabel('Emoções')
plt.ylabel('Contagem')
plt.xticks(rotation=0)
plt.legend(title='Valores Binários', labels=['Não (0)', 'Sim (1)'])
plt.show()

# Gerar um boxplot
plt.figure(figsize=(8, 6))
df[emotion_columns].boxplot()

plt.title('Boxplot das Emoções Binárias')
plt.ylabel('Valores Binários (0 = Não, 1 = Sim)')
plt.xticks(rotation=45)
plt.grid(False)
plt.show()