# Code for evaluating IoU 
# Nov 2017
# Eduardo Romera
#######################

import torch

class iouEval:

    def __init__(self, nClasses, ignoreIndex=19):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses>ignoreIndex else -1 #if ignoreIndex is larger than nClasses, consider no ignoreIndex
        self.reset()

    def reset (self):
        classes = self.nClasses if self.ignoreIndex==-1 else self.nClasses-1
        self.tp = torch.zeros(classes, dtype=torch.float64, device="cpu")
        self.fp = torch.zeros(classes, dtype=torch.float64, device="cpu")
        self.fn = torch.zeros(classes, dtype=torch.float64, device="cpu")

    def addBatch(self, x, y):   #x=preds, y=targets
        #sizes should be "batch_size x nClasses x H x W"

        #if size is "batch_size x 1 x H x W" scatter to onehot
        if x.size(1) == 1:
            x_onehot = torch.zeros(
                x.size(0), self.nClasses, x.size(2), x.size(3),
                device=x.device, dtype=torch.float32
            )
            x_onehot.scatter_(1, x.to(torch.long), 1.0)
        else:
            x_onehot = x.to(dtype=torch.float32)

        if y.size(1) == 1:
            y_onehot = torch.zeros(
                y.size(0), self.nClasses, y.size(2), y.size(3),
                device=y.device, dtype=torch.float32
            )
            y_onehot.scatter_(1, y.to(torch.long), 1.0)
        else:
            y_onehot = y.to(dtype=torch.float32)

        if (self.ignoreIndex != -1):
            ignores = y_onehot[:,self.ignoreIndex].unsqueeze(1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores=0

        #print(type(x_onehot))
        #print(type(y_onehot))
        #print(x_onehot.size())
        #print(y_onehot.size())

        tpmult = x_onehot * y_onehot    #times prediction and gt coincide is 1
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fpmult = x_onehot * (1-y_onehot-ignores) #times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fnmult = (1-x_onehot) * (y_onehot) #times prediction says its not that class and gt says it is
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()

        self.tp += tp.detach().to("cpu").to(torch.float64)
        self.fp += fp.detach().to("cpu").to(torch.float64)
        self.fn += fn.detach().to("cpu").to(torch.float64)

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou), iou     #returns "iou mean", "iou per class"

# Class for colors
class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

# Colored value output if colorized flag is activated.
def getColorEntry(val):
    if not isinstance(val, float):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN

