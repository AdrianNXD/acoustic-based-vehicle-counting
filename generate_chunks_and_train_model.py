import generate_chunks as gc
from chunk_dataset import ChunkDataset
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report


chunk_folder = "chunks"
samplerate = 48000

validation_size = 0.4

t_labels = pd.read_csv("labels.csv", sep=";", index_col=0)




print("Generate the first batch of chunks.")
_ = gc.generate_chunks(t_labels = t_labels, output_folder = chunk_folder, samplerate = samplerate)

print("Generate the second batch of chunks using an one second offset.")
_ = gc.generate_chunks(t_labels = t_labels, output_folder = chunk_folder, samplerate = samplerate, init_offset = 1)

print("Generate the third batch of chunks using an audio file for data augmentation.")
_ = gc.generate_chunks(t_labels = t_labels, output_folder = chunk_folder, samplerate = samplerate, data_augmentation = True)

print("Generate the last batch of chunks with both an one second offset and data augmentation.")
c_labels = gc.generate_chunks(t_labels = t_labels, output_folder = chunk_folder, samplerate = samplerate, data_augmentation = True, init_offset = 1)





# split the available data into train and validation data. Since theres a big imbalance between pass_event 1 and 0
# make sure both datasets contain an equal amount of them.
c_labels_train, c_labels_val = train_test_split(
    c_labels,
    test_size    = validation_size,
    random_state = 42,
    stratify     = c_labels["pass_event"]
)

cd_train = ChunkDataset(
    chunk_folder = chunk_folder, 
    chunk_labels = c_labels_train, 
    samplerate   = samplerate, 
    n_mels       = 80, 
    n_frames     = 80, 
    n_fft        = 1024
)

cd_val = ChunkDataset(
    chunk_folder = chunk_folder, 
    chunk_labels = c_labels_val, 
    samplerate   = samplerate, 
    n_mels       = 80, 
    n_frames     = 80, 
    n_fft        = 1024
)

# create the corresponding data loaders.
dl_train = DataLoader(cd_train, batch_size=64, num_workers=0, shuffle=True)
dl_val   = DataLoader(cd_val, batch_size=64, num_workers=0)




# heres a small convolution network to tackle this challenge.
class Conv2D(nn.Module):
    def __init__(self):
        super().__init__()

        # aonvolutional block 1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(16)

        # convolutional block 2
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)

        # convolutional block 3
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # we use batches of 64 mel spectograms. since we use pooling three times the resolution
        # halves three times. using the default resolution of 80 n_mels by 80 n_frames this results
        # in 80x80 -> 40x40 -> 20x20 -> 10x10. thus the first fully connected layers should have
        # 64 x 10 x 10 input nodes. if you want to experiment with the input resolution make sure
        # to adjust the network at this position.
        self.fc1 = nn.Linear(64 * 10 * 10, 128)
        self.fc2 = nn.Linear(128, 64)

        # this will be the last layers which outputs a tensor with two values. one value for pass_event
        # 0 and one for pass_event 1. the greater value will determine if the model predicted a pass_event
        # or not.
        self.fc3 = nn.Linear(64, 2)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # flatten
        x = torch.flatten(x, start_dim=1)

        # fully Connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x



# now check if CUDA is available for training.
torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training will be on device:", device)



def train_model(model, dl_train, dl_val, device, epochs=20, lr=0.001):
    '''This will be the training function for our model.'''

    # i use the CrossEntropyLoss and a AdamW optimizer with a weight decay of 1e-4. 
    # i also implement a scheduler for an adaptive learning rate. this ensure we 
    # can get as close as possible to the minimum of the loss function.
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = lr,
        weight_decay = 1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode      = "max",      # f1-score should be maximized
        factor    = 0.5,        # if triggered decrease the lr by this factor
        patience  = 3,          # wait at least this much epochs
        threshold = 1e-4,
        min_lr    = 1e-6
    )

    model.to(device)

    # create some empty lists to track the model performance over each training epoch
    train_loss_list = []
    val_loss_list = []

    precision_list = []
    recall_list = []
    f1_list = []

    lr_list = []

    for epoch in range(epochs):

        # -------------------------
        # tranining
        # -------------------------
        model.train()

        running_loss = 0.0

        for images, labels in dl_train:

            images = images.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(dl_train)
        train_loss_list.append(avg_train_loss)

        # -------------------------
        # validation
        # -------------------------
        model.eval()

        val_running_loss = 0.0

        all_predictions = []
        all_labels = []

        with torch.no_grad():

            for images, labels in dl_val:

                images = images.to(device)
                labels = labels.long().to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_running_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # track all metrics
        avg_val_loss = val_running_loss / len(dl_val)
        val_loss_list.append(avg_val_loss)

        precision = precision_score(
            all_labels,
            all_predictions,
            zero_division=0
        )

        recall = recall_score(
            all_labels,
            all_predictions,
            zero_division=0
        )

        f1 = f1_score(
            all_labels,
            all_predictions,
            zero_division=0
        )

        # tell the scheduler the current f1-score so it can decide if the lr should be decreased.
        scheduler.step(f1)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        current_lr = optimizer.param_groups[0]["lr"]
        lr_list.append(current_lr)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

    return (
        {"train_loss":train_loss_list,
        "val_loss":val_loss_list,
        "precision":precision_list,
        "recall":recall_list,
        "f1-score":f1_list,
        "lr":lr}
    )




model = Conv2D()
training_results = train_model(model=model, dl_train=dl_train, dl_val=dl_val, device=device, epochs=20, lr=0.001)
torch.save(model.state_dict(), "trained_model.pth")
print("Model training is DONE.")



# now evaluate the model on the validation data
model.eval()

all_preds = []
all_labels = []

# calculate the predictions
with torch.no_grad():
    for images, labels in dl_val:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# print the classification report
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, digits=4))

# calculate the confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# create the confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
    xticklabels=["no vehicle", "vehicle"],
    yticklabels=["no vehicle", "vehicle"],
)

plt.xlabel("MODEL PREDICTION")
plt.ylabel("LABEL")
plt.title("Confusion Matrix on Validation Data")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()