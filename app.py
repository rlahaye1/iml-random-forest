from math import ceil
from flask import Flask, render_template, request, send_from_directory
import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from WeightedRandomForest import WeightedRandomForest
from Tree import Tree
import graphviz
from PIL import Image
import io

app = Flask(__name__)
app.config["TREE_IMG_FOLDER"] = "static/trees"

# -------------global state----------------

X_train_global = None
y_train_global = None
X_selected = None
y_selected = None

# ---------------- Utils ------------------

def build_graphviz_png(node, filename):
    dot = graphviz.Digraph()
    def recurse(n, prefix=""):
        node_id = prefix
        if n.value is not None:
            dot.node(node_id, f"Leaf: {n.value}", shape="box")
        else:
            label = f"X[{n.feature_index}] < {n.threshold:.2f}"
            dot.node(node_id, label)
            left_id = prefix + "L"
            right_id = prefix + "R"
            dot.edge(node_id, left_id, label="True")
            dot.edge(node_id, right_id, label="False")
            recurse(n.left, left_id)
            recurse(n.right, right_id)
    recurse(node)

    png_bytes = dot.pipe(format="png")
    image = Image.open(io.BytesIO(png_bytes))
    image.save(os.path.join(app.config["TREE_IMG_FOLDER"], filename))

# ---------------- Load forest ------------------

def load_forest():
    global X_train_global, y_train_global
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

    # Stocker pour l'affichage dans la modale
    X_train_global = X_train
    y_train_global = y_train

    forest = WeightedRandomForest()
    forest.fit(X_train, y_train, n_trees=6)

    return forest, X_test, y_test

forest, X_test, y_test = load_forest()

# ---------------- Routes ------------------

@app.route("/")
def index():
    n_per_page = 2
    page = int(request.args.get("page", 0))
    test_index = int(request.args.get("test_index", 0))
    start = page * n_per_page
    end = start + n_per_page

    # Générer images si pas déjà là
    for idx, tree in enumerate(forest.trees):
        path = os.path.join(app.config["TREE_IMG_FOLDER"], f"tree_{idx}.png")
        if not os.path.exists(path) and hasattr(tree, "root"):
            build_graphviz_png(tree.root, f"tree_{idx}.png")

    predictions = forest.predict(X_test[test_index].reshape(1, -1))
    true_label = y_test[test_index]

    global_accuracy = None
    if len(y_test) > 0:
        y_pred_all = forest.predict(X_test)
        correct = np.sum(y_pred_all == y_test)
        total = len(y_test)
        global_accuracy = round((correct / total) * 100)

    return render_template("index.html",
                           trees=list(enumerate(forest.trees))[start:end],
                           weights=forest.weights[start:end],
                           n_pages=ceil(len(forest.trees)/n_per_page),
                           page=page,
                           current_page=page,
                           test_index=test_index,
                           prediction=predictions[0],
                           true_label=true_label,
                           total_trees=len(forest.trees),
                           per_page=n_per_page,
                           X_test=X_test,
                           X_train_global=X_train_global,
                           y_train_global=y_train_global,
                           global_accuracy=global_accuracy
                           )


@app.route("/select-data", methods=["POST"])
def select_data():
    global X_selected, y_selected, X_train_global, y_train_global

    indices_str = request.form.get("selected_indices", "")
    if indices_str:
        indices = list(map(int, indices_str.split(",")))
        X_selected = X_train_global[indices]
        y_selected = y_train_global[indices]
    return "OK"  # Peut aussi rediriger ou retourner un message de confirmation


@app.route("/add-tree", methods=["POST"])
def add_tree():
    global X_selected, y_selected, forest

    weight_str = request.form.get("weight")
    if not weight_str:
        return "Poids non fourni", 400

    if X_selected is None or y_selected is None:
        return "Aucune donnée sélectionnée pour entraîner un arbre", 400

    try:
        weight = float(weight_str)
    except ValueError:
        return "Poids invalide", 400

    # Créer un nouvel arbre pondéré
    new_tree = Tree()
    forest.add_tree(new_tree, X_selected, y_selected, weight=weight)

    # Générer image pour ce nouvel arbre
    new_index = len(forest.trees) - 1  # l’arbre a été ajouté à la fin
    if hasattr(new_tree, "root"):
        build_graphviz_png(new_tree.root, f"tree_{new_index}.png")

    global_accuracy = None
    if len(y_test) > 0:
        y_pred_all = forest.predict(X_test)
        correct = np.sum(y_pred_all == y_test)
        total = len(y_test)
        global_accuracy = round((correct / total) * 100)

    return render_template("index.html",
                           trees=list(enumerate(forest.trees))[-2:],  # dernière page
                           weights=forest.weights[-2:],
                           n_pages=ceil(len(forest.trees)/2),
                           page=ceil(len(forest.trees)/2) - 1,
                           current_page=ceil(len(forest.trees)/2) - 1,
                           test_index=0,
                           prediction=None,
                           true_label=None,
                           total_trees=len(forest.trees),
                           per_page=2,
                           X_test=X_test,
                           X_train_global=X_train_global,
                           y_train_global=y_train_global,
                           global_accuracy=global_accuracy
                           )


# ---------------- Run ------------------

if __name__ == "__main__":
    os.makedirs(app.config["TREE_IMG_FOLDER"], exist_ok=True)
    app.run(debug=True)
