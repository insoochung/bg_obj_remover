import datasets

if __name__ == "__main__":
    dataset = datasets.load_dataset("imagenet-1k")
    print(dataset)
