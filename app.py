from math import ceil
from flask import Flask, render_template, request, send_from_directory
import numpy as np
import os
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
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
current_n_trees = 6
current_train_percent = 70
current_dataset_selected = "iris"

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

def clear_tree_images():
    folder = app.config["TREE_IMG_FOLDER"]
    for filename in os.listdir(folder):
        if filename.startswith("tree_") and filename.endswith(".png"):
            path = os.path.join(folder, filename)
            try:
                os.remove(path)
            except Exception as e:
                print(f"Erreur lors de la suppression de {path} : {e}")

# ---------------- Load forest ------------------

def load_forest(n_trees=6, train_size=0.7, dataset="iris"):
    global X_train_global, y_train_global
   
    # test_size = 1-train_size
    if dataset == "iris":
        data = load_iris()
    elif dataset == "wine":
        data = load_wine()
    elif dataset == "cancer":
        data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, train_size=train_size, random_state=42
    )

    # Stocker pour l'affichage dans la modale
    X_train_global = X_train
    y_train_global = y_train

    forest = WeightedRandomForest()
    forest.fit(X_train, y_train, n_trees=n_trees)

    return forest, X_test, y_test

forest, X_test, y_test = load_forest()

# ---------------- Routes ------------------

@app.route("/")
def index():

    global forest, X_test, y_test, current_n_trees, current_train_percent, current_dataset_selected

    n_per_page = 2

    page = int(request.args.get("page", 0))
    test_index = int(request.args.get("test_index", 0))
    n_trees = int(request.args.get("n_trees", current_n_trees))
    dataset_selected = request.args.get("dataset-select",current_dataset_selected)
    train_percent = int(request.args.get("train_percent", current_train_percent))

    # Si les hyper-paramètres changent, on recharge la forêt
    if current_n_trees != n_trees or current_train_percent != train_percent or current_dataset_selected != dataset_selected:
        train_size = train_percent / 100.0
        # test_size = 1 - train_size

        forest, X_test, y_test = load_forest(n_trees=n_trees, train_size=train_size, dataset=dataset_selected)

        current_n_trees = n_trees
        current_train_percent = train_percent
        current_dataset_selected = dataset_selected
        clear_tree_images()

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


    trees_paginated = [
        {
            "index": i,
            "tree": forest.trees[i],
            "weight": forest.weights[i]
        }
        for i in range(start, min(end, len(forest.trees)))
    ]

    return render_template("index.html",
                            trees_paginated=trees_paginated,
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
                           global_accuracy=global_accuracy,
                           n_trees=n_trees,
                           train_percent=train_percent,
                           dataset_selected=current_dataset_selected
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
    global X_selected, y_selected, forest, current_n_trees, current_train_percent, current_dataset_selected

    weight_str = request.form.get("weight")
    if not weight_str:
        return "Poids non fourni", 400

    if X_selected is None or y_selected is None:
        return "Aucune donnée sélectionnée pour entraîner un arbre", 400

    try:
        weight = float(weight_str)
    except ValueError:
        return "Poids invalide", 400

    new_tree = Tree()
    forest.add_tree(new_tree, X_selected, y_selected, weight=weight)

    new_index = len(forest.trees) - 1
    if hasattr(new_tree, "root"):
        build_graphviz_png(new_tree.root, f"tree_{new_index}.png")

    # recalculer pagination pour afficher dernière page
    n_per_page = 2
    total_trees = len(forest.trees)
    last_page = ceil(total_trees / n_per_page) - 1
    start = last_page * n_per_page
    end = start + n_per_page

    trees_paginated = [
        {
            "index": i,
            "tree": forest.trees[i],
            "weight": forest.weights[i]
        }
        for i in range(start, min(end, total_trees))
    ]

    global_accuracy = None
    if len(y_test) > 0:
        y_pred_all = forest.predict(X_test)
        correct = np.sum(y_pred_all == y_test)
        total = len(y_test)
        global_accuracy = round((correct / total) * 100)

    return render_template("index.html",
                           trees_paginated=trees_paginated,
                           n_pages=ceil(total_trees / n_per_page),
                           page=last_page,
                           current_page=last_page,
                           test_index=0,
                           prediction=None,
                           true_label=None,
                           total_trees=total_trees,
                           per_page=n_per_page,
                           X_test=X_test,
                           X_train_global=X_train_global,
                           y_train_global=y_train_global,
                           global_accuracy=global_accuracy,
                           n_trees=current_n_trees,
                           train_percent=current_train_percent,
                           dataset_selected=current_dataset_selected
                           )


# ---------------- Run ------------------

if __name__ == "__main__":
    os.makedirs(app.config["TREE_IMG_FOLDER"], exist_ok=True)
    app.run(debug=True)
