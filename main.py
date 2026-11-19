import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

st.set_page_config(page_title="Predictive Modeling | Thiranex", page_icon="🤖", layout="wide")
st.title("🤖 Predictive Modeling Using Machine Learning")
st.caption("Synthetic employee attrition dataset — demonstration project")

df = pd.read_csv("data/employee_attrition_synthetic.csv")
st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)
st.write(f"Dataset shape: **{df.shape[0]} rows × {df.shape[1]} columns**")

target = "Attrition"
X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_choice = st.selectbox("Select a model", ["Logistic Regression", "Decision Tree", "Random Forest"])
if model_choice == "Logistic Regression":
    model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
else:
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)

model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

col1, col2, col3 = st.columns(3)
col1.metric("Test Accuracy", f"{accuracy:.2%}")
col2.metric("Training Samples", len(X_train))
col3.metric("Testing Samples", len(X_test))

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_xticks([0,1], labels=["No Attrition", "Attrition"])
ax.set_yticks([0,1], labels=["No Attrition", "Attrition"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha="center", va="center")
st.pyplot(fig)

st.subheader("Classification Report")
st.code(classification_report(y_test, pred, target_names=["No Attrition", "Attrition"], zero_division=0))

st.subheader("Model Comparison")
models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
}
results = []
for name, m in models.items():
    m.fit(X_train, y_train)
    results.append({"Model": name, "Accuracy": accuracy_score(y_test, m.predict(X_test))})
results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
st.dataframe(results_df, use_container_width=True)
fig2, ax2 = plt.subplots()
ax2.bar(results_df["Model"], results_df["Accuracy"])
ax2.set_ylim(0, 1)
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy Comparison")
plt.xticks(rotation=15)
st.pyplot(fig2)

if hasattr(model, "predict_proba"):
    st.subheader("ROC Curve")
    probabilities = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)
    fig3, ax3 = plt.subplots()
    ax3.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax3.plot([0,1], [0,1], "--")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.legend()
    st.pyplot(fig3)

st.info("This project uses a synthetic dataset for learning and demonstration. Do not interpret its results as real HR predictions.")
