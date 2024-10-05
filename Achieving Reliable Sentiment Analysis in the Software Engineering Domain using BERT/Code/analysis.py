# Criar dataframe a partir do csv
import pandas as pd

df = pd.read_csv("C:/Users/gcmor/OneDrive/Área de Trabalho/datasets IC/Achieving Reliable Sentiment Analysis in the Software Engineering Domain using BERT/Dataset/NewData/NewData.csv")


# ver primeiras linhas do dataframe
print(df.head())


# gerar um histograma
import matplotlib.pyplot as plt
df["oracle"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

# Gerar um boxplot
df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="oracle", grid=False,
showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()