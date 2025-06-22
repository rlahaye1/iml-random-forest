class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index de la caractéristique utilisée pour le split
        self.threshold = threshold          # Seuil utilisé
        self.left = left                    # Fils gauche
        self.right = right                  # Fils droit
        self.value = value                  # Classe (si feuille)