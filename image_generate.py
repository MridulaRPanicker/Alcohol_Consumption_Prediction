import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df_original = pd.read_csv('beer-servings_forWebApp.csv') 
plt.figure(figsize=(10,6))
sns.barplot(data=df_original, x='continent', y='total_litres_of_pure_alcohol',color='red',edgecolor='black')
plt.title('Average Alcohol Servings per Continent')
plt.savefig('static/images/infographic.png') # Save to static folder for Flask