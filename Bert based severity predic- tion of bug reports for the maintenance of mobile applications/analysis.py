# Criar dataframe a partir do csv
import pandas as pd

df = pd.read_parquet("C:/Users/gcmor/OneDrive/Área de Trabalho/datasets IC/Bert based severity predic- tion of bug reports for the maintenance of mobile applications/train-00000-of-00001.parquet")


# ver primeiras linhas do dataframe
print(df.head())


# gerar um histograma
import matplotlib.pyplot as plt
df["star"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

# Gerar um boxplot
df["Words Per Tweet"] = df["review"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="star", grid=False,
showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()