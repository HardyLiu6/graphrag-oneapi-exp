import pandas as pd
df = pd.read_parquet("/home/sunlight/Projects/graphrag-oneapi-exp/inputs/artifacts/communities.parquet")
print(df.columns.tolist())
print(df.head())