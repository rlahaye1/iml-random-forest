class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """
        Représente un nœud dans un arbre de décision.

        Attributs :
        - feature_index : int ou None
            Index de la caractéristique utilisée pour diviser le nœud (None si feuille).
        - threshold : float ou None
            Seuil à appliquer sur la caractéristique pour séparer les données.
        - left : TreeNode ou None
            Sous-arbre gauche (instances où la caractéristique <= threshold).
        - right : TreeNode ou None
            Sous-arbre droit (instances où la caractéristique > threshold).
        - value : any ou None
            Valeur prédite (classe majoritaire) si le nœud est une feuille.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value