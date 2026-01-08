# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import os
import sys
import time
import torch
import shutil
import zipfile
import subprocess
from pathlib import Path

from PIL import Image
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),  #ignore label to 19
])

class_names = [
    "Road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle"
]


def select_device(force_cpu: bool) -> torch.device:
    """Select best available device: CUDA > MPS > CPU (unless force_cpu)"""
    if force_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_cityscapes_datadir(datadir: str) -> Path:

    if not (path := Path(datadir)).exists():
        raise FileNotFoundError(f"`{path.expanduser().resolve()}` does not exist")

    # for compatibility reason with the previous version of this script
    if (path / "gtFine").exists() and (path / "leftImg8bit").exists():
        out_root = path
    else:
        # Case Zip FIle: extract next to this script
        left_zip = next(path.glob("leftImg8bit_*.zip"), None)
        gtFine_zip = next(path.glob("gtFine_*.zip"), None)

        if not left_zip or not gtFine_zip:
            raise FileNotFoundError("No Cityscapes dataset found. "
                                    "Provide either a folder containing `gtFine` and `leftImg8bit`"
                                    "or a folder containing `leftImg8bit_*.zip` and `gtFine_*.zip`")

        script_dir = Path(__file__).resolve().parent
        tmp_dir = script_dir / "_cityscapes_tmp"
        out_root = script_dir / "cityscapes"

        with zipfile.ZipFile(left_zip, "r") as zf:
            zf.extractall(tmp_dir)

        with zipfile.ZipFile(gtFine_zip, "r") as zf:
            zf.extractall(tmp_dir)

        # Locate directories named exactly 'gtFine' and 'leftImg8bit'.
        gt_dir = next((p for p in tmp_dir.rglob("gtFine") if p.is_dir()), None)
        left_dir = next((p for p in tmp_dir.rglob("leftImg8bit") if p.is_dir()), None)

        if gt_dir is None or left_dir is None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise FileNotFoundError("Extraction completed but could not locate both 'gtFine' and 'leftImg8bit' directories.")

        # Create the root and copy only the necessary directories
        out_root.mkdir(parents=True, exist_ok=True)
        out_gt = out_root / "gtFine"
        out_left = out_root / "leftImg8bit"

        # Refresh outputs to avoid mixing old/new data
        if out_gt.exists(): shutil.rmtree(out_gt)
        if out_left.exists(): shutil.rmtree(out_left)

        shutil.copytree(gt_dir, out_gt)
        shutil.copytree(left_dir, out_left)

        # Clean up temporary extracted files
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("Creating trainId label images. This may take a while...")
    env = os.environ.copy()
    env["CITYSCAPES_DATASET"] = str(out_root)
    subprocess.run(
        [sys.executable, "-m", "cityscapesscripts.preparation.createTrainIdLabelImgs"], env=env, check=True
    )

    return out_root



def main(args):
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    device = select_device(args.cpu)
    print(f"Device: {device}")

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    # Use DataParallel only on CUDA
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=device))
    model = model.to(device)
    print("Model and weights LOADED successfully")

    model.eval()

    if not Path(args.datadir).expanduser().exists():
        raise FileNotFoundError(f"Data directory `{args.datadir}` does not exist")

    city_root = resolve_cityscapes_datadir(args.datadir)

    loader = DataLoader(
        cityscapes(str(city_root), input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.perf_counter()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)

        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)

        filenameSave = Path(filename[0]).name

        # print(step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i]) + '{:0.2f}'.format(iou_classes[i] * 100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print(f"Took {time.perf_counter() - start:.2f} seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")

    for name, iou_str in zip(class_names, iou_classes_str):
        print(iou_str, name)

    print("=======================================")
    iouStr = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal * 100) + '\033[0m'
    print("MEAN IoU: ", iouStr, "%")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
