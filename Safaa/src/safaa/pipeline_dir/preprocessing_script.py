import re
import os
import spacy
import pandas as pd
from Safaa.src.safaa.Safaa import SafaaAgent


agent = SafaaAgent()

data = pd.read_csv('data/copyrights.csv')
data = data['original_content']

prep = agent.preprocess_data(data)
new_df = pd.DataFrame({
    'original_content': data,
    'preprocessed_content': prep
})
new_df.to_csv('data/preprocessed_copyrights.csv')
