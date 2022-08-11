import os, json

# used to get category names
training_dir = "./dataset/training/"
categories = []

# tf.keras.utils.image_dataset_from_directory orders class directories using
# os.walk. This ensures the model's softmax output has its index match
# list(map(lambda y: " ".join(y[0].split("/")[-1].split("_")[1:]), os.walk("./dataset/training/")))
for folder in os.walk(training_dir):
    folder_path = folder[0]
    
    # skip parent directory
    if folder_path in ["...", training_dir]:
        continue
    
    folder_name = folder_path.split("/")[-1]
    category = " ".join(folder_name.split("_")[1:])
    categories.append(category)
    
categories_str = json.dumps(categories, indent=1)
with open("categories.json", "w") as f:
    f.write(categories_str)