import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Dataset_cleaned.csv')


label = ['hate','non-hate']

def absolute_value(pct, all_values):
    absolute = int(round(pct/100.*sum(all_values))) 
    return f"{absolute:,} ({pct:.1f}%)" 

hate_speach = data[data['label'] == 0]['label'].count()
nonhate_speach = data[data['label'] == 1]['label'].count()
speach = [hate_speach,nonhate_speach]
plt.figure(figsize=(6, 6))
plt.pie(speach, labels=label, autopct=lambda pct: absolute_value(pct, speach), startangle=140, colors=["red", "blue"])
plt.title("hate_comment_ratio")
plt.show()