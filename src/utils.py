import csv, sys

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
    except FileNotFoundError:
        print(f"Error: the file '{fichier}' does not exist.", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: permission denied for file '{fichier}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error while reading '{fichier}': {e}", file=sys.stderr)
        sys.exit(1)
    return tableau

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
