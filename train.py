# Import needed modules
import argparse
import model
import os
import dataloader
import time
import datetime

# Get parameters
parser = argparse.ArgumentParser(prog = "python train.py", usage="%(prog)s data_dir [options]")
parser.add_argument("data_dir", type = str, help = "Data-dir with images")
parser.add_argument("--save_dir", type = str, default="./", help = "Directory where checkpoint.pth should be saved")
parser.add_argument("--arch", type = str, default="vgg19", help = "Choosen architecture", choices=model.available_models())
parser.add_argument("--learning_rate", type = float, default=0.0001, help = "Learning-rate")
parser.add_argument("--hidden_units", type = int, default=512, help = "Number of hidden units")
parser.add_argument("--epochs", type = int, default=3, help = "Number of epochs")
parser.add_argument("--gpu", default = False, action = "store_true", help = "Use GPU for training")
args = parser.parse_args()

# Checking parameters
if args.data_dir[-1] != "/":
    args.data_dir += "/"
if os.path.exists(args.data_dir) == False:
    print(f"Directory `{args.data_dir}` doesn't exist")
    exit()
if os.path.exists(args.data_dir + "train") == False:
    print(f"Directory `{args.data_dir}train/` doesn't exist. Should contain training images.")
    exit()
if os.path.exists(args.data_dir + "valid") == False:
    print(f"Directory `{args.data_dir}valid/` doesn't exist. Should contain validation images.")
    exit()
if args.save_dir[-1] != "/":
    args.save_dir += "/"
if os.path.exists(args.save_dir) == False:
    print(f"Directory `{args.save_dir}` doesn't exist")
    exit()

# Image, transforms and dataloader
imagesets = dataloader.imagesets(args.data_dir)
dataloaders = dataloader.dataloaders(imagesets)

# Setup model
mymodel = model.pretrained_model(args.arch, args.hidden_units, args.learning_rate, imagesets["train"].class_to_idx)
if args.gpu:
    if mymodel.gpu() == False:
        print("No CUDA compatible GPU available. Try without --gpu parameter.")
        exit()

# Training
start_time = time.time()
print(f"Starting training, parameters:")
print(f"    data dir:       {args.data_dir}")
print(f"    save dir:       {args.save_dir}")
print(f"    architecture:   {args.arch}")
print(f"    learning rate:  {args.learning_rate}")
print(f"    hidden units:   {args.hidden_units}")
print(f"    epochs:         {args.epochs}")
print(f"    device:         {mymodel.device}")
print(" ")
print(f"Epoch  Batch  Training loss   Validation loss   Validation accuracy")
for e in range(args.epochs):
    i = 0
    running_loss = 0
    for images, labels in dataloaders["train"]:
        # Let model train this batch
        loss = mymodel.train(images, labels)

        # Sum up loss
        running_loss += loss
        
        # Number of batches
        i += 1

        # Calc % and ETA
        pct = (len(dataloaders["train"]) * e + i) / (len(dataloaders["train"]) * args.epochs) * 100
        eta = int((time.time() - start_time) / pct * (100 - pct))

        # Output stats
        print(f"{e+1:<7}{i:<7}{running_loss / i:<55,.3f}{pct:.1f}% ETA {datetime.timedelta(seconds = eta)}     ", end="\r")
            
    else:
        # End of epoch, let's do some validation
        valid_loss = 0
        valid_accuracy = 0
        for images, labels in dataloaders["valid"]:
            # Let the model classify the batch
            loss, accuracy = mymodel.validate(images, labels)

            # Sum up loss and accuracy
            valid_loss += loss
            valid_accuracy += accuracy

        print(f"{e+1:<7}{i:<7}{running_loss / len(dataloaders['train']):<16,.3f}{valid_loss / len(dataloaders['valid']):<18,.3f}{valid_accuracy / len(dataloaders['valid']) * 100:.0f}%{' ':<40}")

print(" ")
print(f"Training complete. Total time: {datetime.timedelta(seconds = time.time() - start_time)}")

# Saving checkpoint
filename = args.save_dir + args.arch + ".pth"
if mymodel.save(filename):
    print(f"Training checkpoint has been saved as {filename}")

# Just a single test
if False:
    # Open image
    img = dataloader.load_and_proces_image("d:/flower_data/test/51/image_01340.jpg")

    # Predict image
    probs, classes = mymodel.predict(img, 5)

    # Print out prediction
    print("Class                    Probability")
    for cls, prob in zip(classes, probs):
        print(f"{cls:<25}{prob:.3f}")
