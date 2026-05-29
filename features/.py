import pandas as pd

df = pd.read_csv("data/prompts.csv")
print(df.shape)  # Should print (200, 3)
