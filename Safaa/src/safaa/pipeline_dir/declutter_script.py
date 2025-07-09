import pandas as pd
from Safaa.src.safaa.Safaa import SafaaAgent


agent = SafaaAgent()
df = pd.read_csv('data/preprocessed_copyrights.csv')
data = df['original_content']

predictions = ["f"] * len(data)

output = agent.declutter(data, predictions)

new_df = pd.DataFrame({
    'original_content': data,
    'decluttered_content': output
})
new_df.to_csv('data/decluttered_copyrights.csv', index=False)
