import sys

def main():

    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        print("Usage: python mlp.py --dataset <dataset_name>")
        print("                     --train")
        print("                     --predict")
        return 1


    return 1

if __name__ == "__main__":
    main()
