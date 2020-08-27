#imports
from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from PIL import Image
import logging
import faiss
import matplotlib.pyplot as plt
from cycler import cycler
import record_keeper
import pytorch_metric_learning
logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s"%pytorch_metric_learning.__version__)

from record_keeper import RecordKeeper, RecordWriter

# Set the image transforms
train_transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

val_transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# loading the data, creating three different sets
train_dataset =  datasets.ImageFolder('/Users/Simona/Documents/Studium/MasterThesisFolder/AdditionalCode/Fresno_Area/train', transform=train_transform)
val_dataset =  datasets.ImageFolder('/Users/Simona/Documents/Studium/MasterThesisFolder/AdditionalCode/Fresno_Area/val', transform=val_transform)
test_dataset =  datasets.ImageFolder('/Users/Simona/Documents/Studium/MasterThesisFolder/AdditionalCode/Fresno_Area/test', transform=val_transform)


# Model definition
class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]
        #self.record_these = ["last_linear", "net"]

    def forward(self, x):
        return self.net(x)

# Initialize models, optimizers and image transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set trunk model and replace the softmax layer with an identity function
trunk = torchvision.models.resnet18(pretrained=True)
trunk_output_size = trunk.fc.in_features
trunk.fc = common_functions.Identity()
trunk = torch.nn.DataParallel(trunk.to(device))

# Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
embedder = torch.nn.DataParallel(MLP([trunk_output_size, 64]).to(device))

# Set optimizers (trying out different learnng rates)
#trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00001, weight_decay=0.0001)
#embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.0001, weight_decay=0.0001)
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.0001, weight_decay=0.0001)
embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.001, weight_decay=0.0001)

# Create the loss, miner, sampler, and package them into dictionaries
# Set the loss function
loss = losses.TripletMarginLoss(margin=0.1)

# Set the mining function
miner = miners.TripletMarginMiner(margin=0.1, type_of_triplets="all")
#miner = miners.MultiSimilarityMiner(epsilon=0.1)

# Set the dataloader sampler
# 4 samples each will be returned -> for us m=2 max
sampler = samplers.MPerClassSampler(train_dataset.targets, m=2, length_before_new_iter=len(train_dataset))
#sampler  = samplers.FixedSetOfTriplets(train_dataset.targets, len(train_dataset))

# Set other training parameters
batch_size = 64
num_epochs = 4

# Package the above stuff into dictionaries.
models = {"trunk": trunk, "embedder": embedder}
optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}
loss_funcs = {"metric_loss": loss}
mining_funcs = {"tuple_miner": miner}

# Create the training and testing hooks
record_keeper, _, _ = logging_presets.get_record_keeper("example_logs", "example_tensorboard")
hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"val": val_dataset}
model_folder = "example_saved_models"

# Create the tester
tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook = hooks.end_of_testing_hook,
                                            dataloader_num_workers = 32)

end_of_epoch_hook = hooks.end_of_epoch_hook(tester,
                                            dataset_dict,
                                            model_folder,
                                            test_interval = 1,
                                            patience = 1)

trainer = trainers.MetricLossOnly(models,
                                optimizers,
                                batch_size,
                                loss_funcs,
                                mining_funcs,
                                train_dataset,
                                sampler=sampler,
                                dataloader_num_workers = 32,
                                end_of_iteration_hook = hooks.end_of_iteration_hook,
                                end_of_epoch_hook = end_of_epoch_hook)

#Train the model
trainer.train(num_epochs=num_epochs)

PATH1 = './SentinelNaip_TripletMarginMiner_trunk.pth'
PATH2 = './SentinelNaip_TripletMarginMiner_embed.pth'
torch.save(trunk.state_dict(), PATH1)
torch.save(embedder.state_dict(), PATH2)

# Get a dictionary mapping from loss names to lists
loss_histories = hooks.get_loss_history()

test_dict = {"test": test_dataset}
tester.test(epoch=num_epochs,dataset_dict=test_dict, trunk_model = trunk, embedder_model=embedder)

# extract embeddings
train_emb,train_lab = tester.get_all_embeddings(train_dataset,trunk_model = trunk, embedder_model=embedder)
val_emb, val_lab = tester.get_all_embeddings(val_dataset, trunk_model = trunk, embedder_model=embedder)
test_emb, test_lab = tester.get_all_embeddings(test_dataset, trunk_model = trunk, embedder_model=embedder)

np.savetxt('/Users/Simona/Fresno_Area/train_emb_triplet',train_emb)
np.savetxt('/Users/Simona/Fresno_Area/train_lab_triplet',train_lab)
np.savetxt('/Users/Simona/Fresno_Area/val_emb_triplet',val_emb)
np.savetxt('/Users/Simona/Fresno_Area/val_lab_triplet',val_lab)
np.savetxt('/Users/Simona/Fresno_Area/test_emb_triplet',test_emb)
np.savetxt('/Users/Simona/Fresno_Area/test_lab_triplet',test_lab)
