import torch
import numpy as np



def ordinal_convertion(labels, num_classes):
    # conv_labels = np.zeros_like(labels)
    return (labels.unsqueeze(1) >= torch.arange(0, num_classes, device = labels.device).unsqueeze(0)).to(torch.float)


def custom_loss(pred, labels, loss_type="cross",num_classes=5 ):
    loss = torch.nn.MSELoss()
    if loss_type=="ordinal":
        # pred = ordinal_convertion(pred, num_classes)
        labels = ordinal_convertion(labels, num_classes)
    elif loss_type=="mse":
        labels = torch.nn.functional.one_hot(labels,num_classes=num_classes)
    else:
        loss = torch.nn.CrossEntropyLoss()


    return loss(pred, labels)

def calculate_pred(pred, loss_type, num_classes):
    tensor_z = ordinal_convertion(torch.arange(num_classes, device=pred.device), num_classes)

    if loss_type != "cross":
        a = (pred**2).unsqueeze(1) + (tensor_z**2).unsqueeze(0) - 2*pred.unsqueeze(1)*tensor_z.unsqueeze(0)
        return a.sum(-1).min(-1)[-1].cpu().tolist()
    else:
        return pred.argmax(-1).cpu().tolist()

def confussion_matr():
    pass