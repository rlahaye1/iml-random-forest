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
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)
    forest = WeightedRandomForest()
    forest.fit(X_train, y_train, n_trees=6)

    X_subset = X_train[:10]
    y_subset = y_train[:10]
    important_tree = Tree()
    forest.add_tree(important_tree, X_subset, y_subset, weight=5.0)

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

    return render_template("index.html",
                           trees=list(enumerate(forest.trees))[start:end],
                           weights=forest.weights[start:end],
                           n_pages=(len(forest.trees) + n_per_page - 1) // n_per_page,
                           page=page,
                           current_page=page,
                           test_index=test_index,
                           prediction=predictions[0],
                           true_label=true_label,
                           total_trees=len(forest.trees),
                           per_page=n_per_page,
                           X_test=X_test
                           )

# ---------------- Run ------------------

if __name__ == "__main__":
    os.makedirs(app.config["TREE_IMG_FOLDER"], exist_ok=True)
    app.run(debug=True)
