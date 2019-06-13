# Import needed modules
import argparse
import model
import os
import dataloader

# Get parameters
parser = argparse.ArgumentParser(prog = "python predict.py", usage="%(prog)s /path/to/image checkpoint-file")
parser.add_argument("image", type = str, help = "Path to image")
parser.add_argument("checkpoint", type = str, help = "Checkpoint file to use")
parser.add_argument("--top_k", type = int, default=3, help = "Number of most likely classes to return")
parser.add_argument("--category_names", type = str, default="", help = "JSON-file with mapping of categories to real names")
parser.add_argument("--gpu", default = False, action = "store_true", help = "Use GPU for predicting")
args = parser.parse_args()

# Checking parameters
if os.path.isfile(args.image) == False:
    print(f"Image `{args.image}` doesn't exist")
    exit()
if os.path.isfile(args.checkpoint) == False:
    print(f"Checkpoint-file `{args.checkpoint}` doesn't exist")
    exit()
if args.category_names != "" and os.path.isfile(args.category_names) == False:
    print(f"JSON-file `{args.category_names}` doesn't exist")
    exit()

# Setup and load model
mymodel = model.pretrained_model_load(args.checkpoint)
if args.gpu:
    if mymodel.gpu() == False:
        print("No CUDA compatible GPU available. Try without --gpu parameter.")
        exit()

# Open image
img = dataloader.load_and_proces_image(args.image)

# Predict image
probs, classes = mymodel.predict(img, args.top_k)

# Should we use class to names?
cat_to_name = False
if args.category_names != "":
    cat_to_name = model.cat_to_name(args.category_names)

# Print out prediction
print("Class                    Probability")
for cls, prob in zip(classes, probs):
    if cat_to_name != False:
        cls = cat_to_name[cls]

    print(f"{cls:<25}{prob:.3f}")
