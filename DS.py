import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# Display title
image_path = 'Inno_logo_.png'  # Replace with your actual PNG image file path

# Specify the desired width and height
st.image(image_path, width=400)

# Title and description
st.title("Decision Surface Visualization")
st.write("""
This app visualizes the decision surface of various classifiers.
""")

# Sidebar for user input

st.sidebar.title("Parameters")
dataset_name = st.sidebar.selectbox("Select Dataset", ["make_classification", "make_circles", "make_moons"])
classifier_name = st.sidebar.selectbox("Select Classifier", ["KNN", "Naive Bayes", "Logistic Regression", "Decision Tree"])
algorithm = st.sidebar.selectbox("KNN Algorithm", ["kd_tree", "ball_tree"], disabled=classifier_name != "KNN")
n_neighbors = st.sidebar.slider("Number of neighbors (k)", 1, 15, 5, disabled=classifier_name != "KNN")
weights = st.sidebar.selectbox("KNN Weights", ["uniform", "distance"], disabled=classifier_name != "KNN")
random_state = st.sidebar.number_input("Random state", value=42)
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3)
decision_mode = st.sidebar.radio(
    "Select Decision Mode", 
    ["Single Decision Region", "Multiple Decision Regions", "Best K Values"],
    disabled=classifier_name != "KNN"
)

# Generate synthetic dataset
if dataset_name == "make_classification":
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
elif dataset_name == "make_circles":
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.5)
elif dataset_name == "make_moons":
    X, y = make_moons(n_samples=100, noise=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

def plot_decision_surface(clf, X_train, X_test, y_train, y_test, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold,
                edgecolor='k', s=50, marker='*')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    st.pyplot(plt)
    plt.close()  # Close the plot to avoid re-rendering issues

if classifier_name == "KNN":
    if decision_mode == "Single Decision Region":
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm, weights=weights)
        clf.fit(X_train, y_train)
        plot_decision_surface(clf, X_train, X_test, y_train, y_test, f"KNN: Single Decision Region (k = {n_neighbors}, weights = '{weights}')")

    elif decision_mode == "Multiple Decision Regions":
        for k in range(1, 4):
            clf = KNeighborsClassifier(n_neighbors=k, algorithm=algorithm, weights=weights)
            clf.fit(X_train, y_train)
            plot_decision_surface(clf, X_train, X_test, y_train, y_test, f"KNN: Decision Region (k = {k}, weights = '{weights}')")

    elif decision_mode == "Best K Values":
        best_k = 1
        best_score = 0
        for k in range(1, 16):
            clf = KNeighborsClassifier(n_neighbors=k, algorithm=algorithm, weights=weights)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            if score > best_score:
                best_k = k
                best_score = score

        clf = KNeighborsClassifier(n_neighbors=best_k, algorithm=algorithm, weights=weights)
        clf.fit(X_train, y_train)
        plot_decision_surface(clf, X_train, X_test, y_train, y_test, f"KNN: Best K Value (k = {best_k}, weights = '{weights}')")
        st.write(f"Best K value determined by test set accuracy: {best_k}")

elif classifier_name == "Naive Bayes":
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    plot_decision_surface(clf, X_train, X_test, y_train, y_test, "Naive Bayes")

elif classifier_name == "Logistic Regression":
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    plot_decision_surface(clf, X_train, X_test, y_train, y_test, "Logistic Regression")

elif classifier_name == "Decision Tree":
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    plot_decision_surface(clf, X_train, X_test, y_train, y_test, "Decision Tree")
