import csv
import torch
from torch.utils.data import DataLoader
from medpy.metric.binary import hd95
import numpy as np
from scipy import ndimage
from torch.nn.functional import cross_entropy

from model import UnetMCDoptout
from dataset import ISIC2018DataSet, TwoStreamBatchSampler
from utils import ramp_up, dice_loss, parse_args, set_seed

args = parse_args()
BASE_DIR = args.data_dir

image_training = BASE_DIR / "ISIC2018_Task1-2_Training_Input"
mask_training = BASE_DIR / "ISIC2018_Task1_Training_GroundTruth"
image_validation = BASE_DIR / "ISIC2018_Task1-2_Validation_Input"
mask_validation = BASE_DIR / "ISIC2018_Task1_Validation_GroundTruth"
image_testing = BASE_DIR / "ISIC2018_Task1-2_Test_Input"
mask_testing = BASE_DIR / "ISIC2018_Task1_Test_GroundTruth"

# hyperparameters
# training batch = LABELED_BS labeled + UNLABELED_BS unlabeled images per step
LABELED_BS = 8
UNLABELED_BS = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 0.01
MAX_ITERATIONS = 6000
EMA_ALPHA = 0.99
RAMP_UP = 40
MC_DROPOUT = 8
LABELED_FRACTION = 0.1
BASE_CHANNELS = 16
SEED = 42

set_seed(SEED)

# set up right device for machine
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


def keep_largest_cc(mask):
    """
    Post-processing: keep only the largest connected foreground component.
    """
    if mask.sum() == 0:
        return mask
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    # component 0 is background; find the largest of labels 1..n
    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return labeled == largest


def evaluate(model, loader, names=None, log_path=None):
    """
    Single-pass inference over a loader; returns (mean_dice, mean_iou, mean_hd95).
    """
    dice_scores, iou_scores, hd95_scores = [], [], []
    log_rows = []
    sample_idx = 0
    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            images = images.float().to(DEVICE)
            masks = masks.float().to(DEVICE)

            probs = torch.softmax(model(images), dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(len(preds)):
                pred_i = preds[i].cpu().numpy().astype(bool)
                pred_i = keep_largest_cc(pred_i)
                mask_i = masks[i].cpu().numpy().astype(bool)

                intersection = (pred_i & mask_i).sum()
                union = pred_i.sum() + mask_i.sum() - intersection
                dice = (2 * intersection + 1e-6) / (pred_i.sum() + mask_i.sum() + 1e-6)
                IoU = (intersection + 1e-6) / (union + 1e-6)

                dice_scores.append(dice)
                iou_scores.append(IoU)

                # SSL4MIS HD95: empty prediction -> 0 (counted in mean)
                if pred_i.sum() > 0 and mask_i.sum() > 0:
                    hd95_score = hd95(pred_i, mask_i)
                    hd95_scores.append(hd95_score)
                elif pred_i.sum() == 0:
                    hd95_score = 0.0
                    hd95_scores.append(hd95_score)
                else:
                    hd95_score = float("nan")  # empty GT (medpy undefined): exclude

                if names is not None:
                    log_rows.append([names[sample_idx], dice, IoU, hd95_score])
                sample_idx += 1

    if log_path is not None and log_rows:
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(log_rows)

    mean_dice = float(np.mean(dice_scores))
    mean_iou = float(np.mean(iou_scores))
    mean_hd95 = float(np.mean(hd95_scores)) if hd95_scores else float("nan")
    return mean_dice, mean_iou, mean_hd95


# create datasets — one training dataset holds both streams (labeled first,
# then unlabeled); the sampler decides the per-batch mix.
train_dataset = ISIC2018DataSet(image_training, mask_training, 'train', LABELED_FRACTION, aug_seed=SEED)
val_dataset = ISIC2018DataSet(image_validation, mask_validation, 'val', LABELED_FRACTION)
test_dataset = ISIC2018DataSet(image_testing, mask_testing, 'test', LABELED_FRACTION)

# two-stream sampler: every batch is LABELED_BS labeled + UNLABELED_BS unlabeled
# indices. Reshuffles per epoch and re-augments every access (no cycle() cache).
batch_sampler = TwoStreamBatchSampler(
    train_dataset.labeled_indices,
    train_dataset.unlabeled_indices,
    batch_size=LABELED_BS + UNLABELED_BS,
    labeled_batch_size=LABELED_BS,
)
train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)


def infinite_batches(loader):
    """Yield batches forever, restarting the loader each epoch so the sampler
    reshuffles and the dataset re-augments without caching any batch."""
    while True:
        for batch in loader:
            yield batch

# create teacher and student model
teacher = UnetMCDoptout(n_channels=3, n_classes=2, base_ch=BASE_CHANNELS).to(DEVICE)
student = UnetMCDoptout(n_channels=3, n_classes=2, base_ch=BASE_CHANNELS).to(DEVICE)

# copy students weights into the teacher
teacher.load_state_dict(student.state_dict())

# freeze the teacher's gradients
for p in teacher.parameters():
    p.requires_grad = False

# optimizer
optimizer = torch.optim.SGD(
    student.parameters(),
    lr = LEARNING_RATE,
    momentum=0.9,
    weight_decay=1e-4,
)


train_iter = infinite_batches(train_loader)
best_dice = 0.0

iters_per_epoch = len(batch_sampler)

VAL_LOG_PATH = "validation_log.csv"
with open(VAL_LOG_PATH, mode="w", newline="") as f:
    csv.writer(f).writerow(["iteration", "dice", "iou", "hd95"])


for iteration in range(1, 1 if args.eval_only else MAX_ITERATIONS + 1):
    print(f"iteration [{iteration}/{MAX_ITERATIONS}]")

    lambda_t, rampup_fraction = ramp_up(iteration // iters_per_epoch, RAMP_UP)

    student.train()
    teacher.eval()

    # set dropout layers to train keep batchnorm layers in eval
    for module in teacher.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
            module.train()

    # batch is [labeled | unlabeled] along dim 0  per TwoStreamBatchSampler.
    combined_images, combined_masks = next(train_iter)
    combined_images = combined_images.to(DEVICE)
    combined_masks = combined_masks.to(DEVICE)

    labeled_bs = LABELED_BS
    masks = combined_masks[:labeled_bs]
    u_images = combined_images[labeled_bs:]

    # zero out gradients
    optimizer.zero_grad()

    # forward pass
    student_outputs = student(combined_images)

    # supervised loss
    ce_loss = cross_entropy(student_outputs[:labeled_bs], masks)
    d_loss = dice_loss(student_outputs[:labeled_bs], masks)
    supervised_loss = 0.5 * (ce_loss + d_loss)

    teacher_preds = []

    # teachers forward pass
    with torch.no_grad():
        for T in range(MC_DROPOUT):
            noisy_images = u_images + torch.clamp(torch.randn_like(u_images) * 0.1, -0.2, 0.2)
            probs = torch.softmax(teacher(noisy_images), dim=1)
            teacher_preds.append(probs)

        teacher_preds = torch.stack(teacher_preds, dim=0)
        teacher_mean = teacher_preds.mean(dim=0)
        entropy = -torch.sum(teacher_mean * torch.log(teacher_mean + 1e-6), dim=1, keepdim=True)
        threshold = (0.75 + 0.25 * rampup_fraction) * np.log(2)
        uncertainty_mask = (entropy < threshold).float()

    # apply softmax to student outputs
    student_probs = torch.softmax(student_outputs[labeled_bs:], dim=1)
    mse = (student_probs - teacher_mean) ** 2

    consistency_loss = lambda_t * (mse * uncertainty_mask).sum() / (uncertainty_mask.sum() * 2 + 1e-6)

    # backward pass
    total_loss = supervised_loss + consistency_loss
    total_loss.backward()
    optimizer.step()

    # polynomial LR decay
    lr = LEARNING_RATE * (1 - iteration / MAX_ITERATIONS) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # compute alpha
    alpha = min(1 - 1/iteration, EMA_ALPHA)

    # EMA teacher update includes BatchNorm buffers via state_dict
    with torch.no_grad():
        for ema_v, student_v in zip(teacher.state_dict().values(), student.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.copy_(alpha * ema_v + (1 - alpha) * student_v)
            else:
                ema_v.copy_(student_v)

    if iteration % 200 == 0:
        mean_dice, mean_iou, mean_hd95 = evaluate(student, val_loader)
        print(f"--- Validation @ iter {iteration} ---")
        print(f"Dice: {mean_dice:.4f}")
        print(f"IoU: {mean_iou:.4f}")
        print(f"HD95: {mean_hd95:.4f}")

        with open(VAL_LOG_PATH, mode="a", newline="") as f:
            csv.writer(f).writerow([iteration, mean_dice, mean_iou, mean_hd95])

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save({
                'iteration': iteration,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "checkpoint_best.pth")
            print(f"  -> New best Dice {best_dice:.4f}, saved checkpoint_best.pth")



checkpoint = torch.load("checkpoint_best.pth", map_location=DEVICE)
student.load_state_dict(checkpoint['student_state_dict'])
if args.eval_only:
    print(f"Loaded best checkpoint from iteration {checkpoint['iteration']}")
else:
    print(f"Loaded best checkpoint from iteration {checkpoint['iteration']} (Dice {best_dice:.4f})")

test_names = test_dataset.samples

TEST_LOG_PATH = "test_per_image_log.csv"
with open(TEST_LOG_PATH, mode="w", newline="") as f:
    csv.writer(f).writerow(["image_name", "dice", "iou", "hd95"])

mean_dice, mean_iou, mean_hd95 = evaluate(student, test_loader, names=test_names, log_path=TEST_LOG_PATH)

print("-------- Test Results ----------")
print(f"Dice: {mean_dice:.4f}")
print(f"IoU: {mean_iou:.4f}")
print(f"HD95: {mean_hd95:.4f}")
print(f"Per-image results written to {TEST_LOG_PATH}")
