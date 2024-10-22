import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\nairp\Downloads\titanic_updated_200.csv") 


print(df.info())
print(df.describe())


plt.figure(figsize=(8, 6))
plt.hist(df['Age'].dropna(), bins=20, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(6, 6))
df['Sex'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])
plt.title('Sex Distribution')
plt.ylabel('')  
plt.show()


df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

df_encoded = df_encoded.drop(['Name', 'Ticket', 'Cabin'], axis=1)


corr_matrix = df_encoded.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

