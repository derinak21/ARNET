import torch as torch
from torchvision.models import mobilenet_v3_small
import numpy as np


# import 2 dataloader modules here one for training, validating and the second for testing
training_dataloader=Dataloader(...)
#training dataset should only have anchor, positive, negative
validation_dataloader=Dataloader(...)
validation_pretrainedclasses_dataloader= Dataloader(...)
testing_dataloader=Dataloader(...)
testing_pretraineddataloader = Dataloader(...)

#validation and testing dataset should have- return pretrained classes (dataframe: output vector, label), new images (dataframe: output vector, label)
model = mobilenet_v3_small(weights='IMAGENET1K_V1')

for name, param in model.named_parameters():
    if 'classifier' not in name:
        param.requires_grad = False

def train_one_epoch():
    running_loss = 0.
    last_loss = 0.
    
    for i, data in enumerate(training_dataloader):
        anchor, positive, negative = data
        optimizer.zero_grad()
        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)
        loss = loss_function(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 
            print(f'batch {i+1} loss: {last_loss}')
    
    return last_loss

def validate_one_epoch():
    accuracy=0
    for i, val_data in enumerate(validation_dataloader):
        X_val, y_val = val_data 
        y_hat_val = model(X_val)
        for i, val_pretrained_data in enumerate(validation_pretrainedclasses_dataloader):
            X_pretrained_val, y_pretrained_val = val_pretrained_data
            y_hat_pretrained_val = model(X_pretrained_val)
            current_difference = np.linalg.norm(y_hat_val - y_hat_pretrained_val)
            if current_difference < best:
                best = current_difference 
                output_label= y_pretrained_val
        accuracy += (output_label==y_val)
    accuracy /= i 
    return accuracy

# train and validation loop
loss_function = torch.nn.TripletMarginLoss()
optimizer = torch.optim.Adam(model.parameters())
EPOCHS = 10

for epoch in range(EPOCHS):
    print(f'EPOCH : {epoch + 1}')
    model.train()
    avg_loss = train_one_epoch()
    model.eval()
    with torch.no_grad():
        accuracy= validate_one_epoch()
    print('LOSS train {} valid {}'.format(avg_loss))
    

# testing loop
model.eval()
with torch.no_grad():
    accuracy=0
    for i, tdata in enumerate(testing_dataloader):
        tinput, tlabel = tdata 
        toutput = model(tinput)
        for i, tpretrained in enumerate(testing_pretraineddataloader):
            tpretrainedinput, tpretrainedlabel = tpretrained
            tpretrainedoutput = model(tpretrainedinput)
            current_similarity= torch.nn.functional.cosine_similarity(toutput, tpretrainedoutput)
            if current_similarity > best:
                best= current_similarity
                output_label= tpretrainedlabel
        accuracy= accuracy+ (output_label==tlabel)

accuracy= accuracy/i 
print(accuracy)