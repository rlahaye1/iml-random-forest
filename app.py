from math import ceil
from flask import Flask, render_template, request, send_from_directory
import numpy as np
import os
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_california_housing 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from WeightedRandomForest import WeightedRandomForest
from Tree import Tree
import graphviz
from PIL import Image
import io
from collections import defaultdict

app = Flask(__name__)
app.config["TREE_IMG_FOLDER"] = "static/trees"

# -------------global state----------------

X_train_global = None
y_train_global = None
X_selected = None
y_selected = None
current_n_trees = 6
current_train_percent = 70
current_dataset_selected = "california"
selected_features = None
max_accuracy = 100
current_chunk = 0
chunk_list_x_train = None
chunk_list_y_train = None
chunk_predictions_correct = []
feature_filter = None

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

def get_filtered_tree_data(forest, threshold=100, feature_filter=None):
    filtered_data = []

    # Si vide ou invalide → pas de filtre
    try:
        feature_filter = int(feature_filter)
    except (TypeError, ValueError):
        feature_filter = None

    for i, (tree, weight, acc) in enumerate(zip(forest.trees, forest.weights, forest.accuracy)):
        if acc <= threshold:
            if feature_filter is not None and feature_filter not in tree.selected_features:
                continue
            filtered_data.append({
                "index": i,
                "tree": tree,
                "weight": weight,
                "accuracies": acc
            })
    return filtered_data


def calculate_features_ponderate(trees, weights):
    global feature_names

    feature_weight_map = defaultdict(float)

    for tree, weight in zip(trees, weights):
        for f in getattr(tree, "selected_features", []):
            feature_weight_map[f] += weight

    feature_distribution = []
    for i, name in enumerate(feature_names):
        value = round(feature_weight_map.get(i, 0), 4)
        if value > 0:
            feature_distribution.append({"name": name, "value": value})

    feature_distribution.sort(key=lambda x: x["value"], reverse=True)

    return feature_distribution



# ---------------- Load forest ------------------

def load_forest(n_trees=6, train_size=0.7, dataset="california"):
    global X_train_global, y_train_global, feature_names
    global chunk_list_x_train, chunk_list_y_train, current_chunk
   
    # test_size = 1-train_size
    if dataset == "iris":
        data = load_iris()
    elif dataset == "wine":
        data = load_wine()
    elif dataset == "cancer":
        data = load_breast_cancer()
    elif dataset == "california":
        data_raw = fetch_california_housing()
        X = data_raw.data
        y_cont = data_raw.target  
        median = np.median(y_cont)
        y = (y_cont > median).astype(int)
        data = type('obj', (object,), {
            'data': X,
            'target': y,
            'feature_names': data_raw.feature_names
        })()

    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, train_size=train_size, random_state=42
    )

    # Mélanger pour bien répartir
    X_shuffled, y_shuffled = shuffle(X_train, y_train, random_state=42)

    n_total = len(X_shuffled)
    n_main = int(0.8 * n_total)
    n_chunk = int(0.02 * n_total)

    # La partie principale pour X_train / y_train
    X_train = X_shuffled[:n_main]
    y_train = y_shuffled[:n_main]

    # Les 20% restants en 10 chunks
    X_chunk_pool = X_shuffled[n_main:n_main + n_chunk * 10]
    y_chunk_pool = y_shuffled[n_main:n_main + n_chunk * 10]

    chunk_list_x_train = [X_chunk_pool[i * n_chunk:(i + 1) * n_chunk] for i in range(10)]
    chunk_list_y_train = [y_chunk_pool[i * n_chunk:(i + 1) * n_chunk] for i in range(10)]

    # Stockage global pour affichage
    X_train_global = X_train
    y_train_global = y_train

    current_chunk = 0  # reset

    forest = WeightedRandomForest()
    forest.fit(X_train, y_train, n_trees=n_trees)

    return forest, X_test, y_test

forest, X_test, y_test = load_forest()

# ---------------- Render Index ------------

def render_index(page=0, test_index=0, prediction=None, true_label=None):
    global chunk_predictions_correct

    n_per_page = 2
    start = page * n_per_page
    end = start + n_per_page

    # Générer images si elles n'existent pas
    for idx, tree in enumerate(forest.trees):
        path = os.path.join(app.config["TREE_IMG_FOLDER"], f"tree_{idx}.png")
        if not os.path.exists(path) and hasattr(tree, "root"):
            build_graphviz_png(tree.root, f"tree_{idx}.png")

    global_accuracy = None
    if len(y_test) > 0:
        y_pred_all = forest.predict(X_test)
        correct = np.sum(y_pred_all == y_test)
        total = len(y_test)
        global_accuracy = round((correct / total) * 100)

        forest.accuracy = []
        for preds in forest.per_tree_preds:
            correct_tree = np.sum(preds == y_test)
            acc_tree = (correct_tree / total) * 100
            forest.accuracy.append(round(acc_tree, 2))

    feature_distribution = calculate_features_ponderate(forest.trees, forest.weights)

    filtered_trees = get_filtered_tree_data(forest, max_accuracy, feature_filter)
    trees_paginated = filtered_trees[start:end]
    n_pages = ceil(len(filtered_trees) / n_per_page)

    if chunk_list_x_train and chunk_list_y_train:
        X_current = chunk_list_x_train[current_chunk]
        y_current = chunk_list_y_train[current_chunk]
        y_pred = forest.predict(X_current)
        chunk_predictions_correct = (y_pred == y_current).astype(int).tolist()

    

    return render_template("index.html",
                           trees_paginated=trees_paginated,
                           n_pages=n_pages,
                           page=page,
                           current_page=page,
                           test_index=test_index,
                           prediction=prediction,
                           true_label=true_label,
                           total_trees=len(forest.trees),
                           per_page=n_per_page,
                           X_test=X_test,
                           X_train_global=X_train_global,
                           y_train_global=y_train_global,
                           global_accuracy=global_accuracy,
                           n_trees=current_n_trees,
                           train_percent=current_train_percent,
                           dataset_selected=current_dataset_selected,
                           feature_names=feature_names,
                           current_chunk= current_chunk,
                           chunk_list_x_train=chunk_list_x_train,
                           chunk_predictions_correct=chunk_predictions_correct,
                           feature_distribution=feature_distribution
                           )


# ---------------- Routes ------------------

@app.route("/")
def index():
    global forest, X_test, y_test, current_n_trees, current_train_percent, current_dataset_selected, feature_names, max_accuracy, feature_filter

    n_trees = int(request.args.get("n_trees", current_n_trees))
    dataset_selected = request.args.get("dataset-select", current_dataset_selected)
    train_percent = int(request.args.get("train_percent", current_train_percent))
    max_accuracy = int(request.args.get("max_accuracy", 100))
    page = int(request.args.get("page", 0))
    test_index = int(request.args.get("test_index", 0))
    feature_filter = request.args.get("feature_filter")

    # Si les hyper-paramètres changent, on recharge la forêt
    if current_n_trees != n_trees or current_train_percent != train_percent or current_dataset_selected != dataset_selected:
        forest, X_test, y_test = load_forest(n_trees=n_trees, train_size=train_percent / 100.0, dataset=dataset_selected)
        current_n_trees = n_trees
        current_train_percent = train_percent
        current_dataset_selected = dataset_selected
        clear_tree_images()

    return render_index(page=page, test_index=test_index)


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
    global X_selected, y_selected, forest, current_n_trees, current_train_percent
    global current_dataset_selected, selected_features, feature_names, max_accuracy, current_chunk, feature_filter

    weight_str = request.form.get("weight")
    max_accuracy = int(request.args.get("max_accuracy", 100))
    feature_filter = request.args.get("feature_filter")

    if not weight_str:
        return "Poids non fourni", 400

    if X_selected is None or y_selected is None:
        return "Aucune donnée sélectionnée pour entraîner un arbre", 400

    try:
        weight = float(weight_str)
    except ValueError:
        return "Poids invalide", 400

    new_tree = Tree()
    forest.add_tree(new_tree, X_selected, y_selected, weight=weight, forced_features=selected_features)

    new_index = len(forest.trees) - 1
    if hasattr(new_tree, "root"):
        build_graphviz_png(new_tree.root, f"tree_{new_index}.png")

    # Aller à la dernière page
    n_per_page = 2
    last_page = ceil(len(forest.trees) / n_per_page) - 1

    current_chunk += 1

    if current_chunk == 10:
        current_chunk=0
    X_selected = None
    y_selected = None
    selected_features= None


    return render_index(page=last_page)

@app.route("/select-features", methods=["POST"])
def select_features():
    global selected_features
    indices_str = request.form.get("selected_features", "")
    if indices_str:
        selected_features = list(map(int, indices_str.split(",")))
    else:
        selected_features = None
    return "OK"

@app.route("/delete-tree/<int:index>", methods=["POST"])
def delete_tree(index):
    global forest

    if index < 0 or index >= len(forest.trees):
        return f"Index {index} invalide", 400

    # Supprimer l'arbre, son poids, et ses prédictions
    del forest.trees[index]
    del forest.weights[index]
    del forest.per_tree_preds[index]
    if hasattr(forest, "accuracy") and index < len(forest.accuracy):
        del forest.accuracy[index]

    # Supprimer le fichier image associé si présent
    image_path = os.path.join(app.config["TREE_IMG_FOLDER"], f"tree_{index}.png")
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
        except Exception as e:
            print(f"Erreur lors de la suppression de l'image : {e}")

    # Renommer les images restantes pour conserver les bons index
    for i, tree in enumerate(forest.trees):
        expected_path = os.path.join(app.config["TREE_IMG_FOLDER"], f"tree_{i}.png")
        if not os.path.exists(expected_path) and hasattr(tree, "root"):
            build_graphviz_png(tree.root, f"tree_{i}.png")

    return render_index()



# ---------------- Run ------------------

if __name__ == "__main__":
    os.makedirs(app.config["TREE_IMG_FOLDER"], exist_ok=True)
    app.run(debug=True)
