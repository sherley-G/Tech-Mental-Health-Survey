# step 1:Importing Libraries and Dataset
import pandas as pd
df=pd.read_csv("tech_mental_health_with_roles.csv")
df
df.head()
df.info()
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# step 2:Data Preparation
df.isnull().sum()
y = df['sought_treatment']
X = df.drop('sought_treatment', axis=1)
X = pd.get_dummies(X, drop_first=True)

#step 3:filtering
filtered_df = df[df['Country'] == 'Canada']
print(filtered_df.head(2))

#step 4: grouping
result1=df.groupby('Gender')['Hours_Worked_Per_Week'].sum()
print(result1.head(2))


# Plot
import matplotlib.pyplot as plt
# Count how many people sought treatment in each role
treatment_counts = df[df["sought_treatment"] == "Yes"]["Role"].value_counts()
plt.bar(treatment_counts.index, treatment_counts.values, color="skyblue", edgecolor="black")
# Adding labels and title
plt.title("Employees Who Sought Treatment by Role", fontsize=14, fontweight='bold')
plt.xlabel("Role", fontsize=12)
plt.ylabel("Number of Employees (Sought Treatment)", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Compute correlation matrix
import seaborn as sns
corr = df.corr(numeric_only=True)
# Plot heatmap
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Mental Health Factors", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#SCATTER PLOT
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="Age", y="Hours_Worked_Per_Week", hue="Work_Stress_Level", s=100, palette="Set2")
plt.title("Scatter Plot of Age vs Hours Worked Per Week")
plt.xlabel("Age")
plt.ylabel("Hours Worked Per Week")
plt.legend(title="Work Stress Level")
plt.show()
import matplotlib.pyplot as plt

# Histogram of Age
plt.figure(figsize=(8,6))
plt.hist(df['Age'], bins=6, color='skyblue', edgecolor='black')
plt.title("Histogram of Employee Age")
plt.xlabel("Age")
plt.ylabel("Hours_Worked_Per_Week")
plt.show()


#step 5:Train-Test Split and Model Training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
dt = DecisionTreeClassifier(
    criterion='gini',    
    max_depth=6,          
    random_state=42
)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

#step 6:Model Evaluation (Before Tuning)
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Model Accuracy:", round(accuracy * 100, 2), "%")


#step 7:Hyperparameter Tuning with GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [4, 6, 8, 10, 12, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 6],
    'splitter': ['best', 'random']
}
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,                
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1            
)

grid_search.fit(X_train, y_train)
print("Best Parameters Found:", grid_search.best_params_)
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)

#step 6:Final Results after Tuning
accuracy = accuracy_score(y_test, y_pred)
print("Tuned Decision Tree Accuracy:", round(accuracy * 100, 2), "%")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

importances = pd.Series(best_dt.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh', title='Top 10 Important Features')
