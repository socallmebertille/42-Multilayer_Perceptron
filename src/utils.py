import csv

def lire_csv(fichier):
    """
    Read a CSV file and return its content as a list of lists.
    """

    tableau = []
    try:
        with open(fichier, "r", encoding="utf-8") as f:
            lecteur = csv.reader(f, delimiter=",")
            for ligne in lecteur:
                ligne = [float(valeur) if valeur.replace('.', '', 1).isdigit() else valeur for valeur in ligne]
                tableau.append(ligne)
        return tableau
    except FileNotFoundError:
        print(f"Error: the file {fichier} does not exist.")
        return []

def ecrire_csv(fichier, data):
    """
    Write data (list of lists) to a CSV file.
    """

    try:
        with open(str(fichier), "w", newline='', encoding="utf-8") as f:
            ecrivain = csv.writer(f, delimiter=",")
            for ligne in data:
                ecrivain.writerow(ligne)
    except Exception as e:
        print(f"Error while writing into {fichier} : {e}")
    return

def binary_class(tab, class1, class2):
    """
    Convert the class labels in the dataset to binary format.
    """

    for i in range(len(tab)):
        if tab[i] == class1:
            tab[i] = 0
        elif tab[i] == class2:
            tab[i] = 1
    return tab
