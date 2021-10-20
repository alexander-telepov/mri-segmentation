from collections import Counter
from tqdm import tqdm
import torchio
from torchio import DATA
import torch
import numpy as np
import enum
from collections import defaultdict
from torch.cuda.amp import autocast
from .utils import prepare_batch, prepare_aseg, MRI, LABEL


class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'


def predict(model, sample, validation_batch_size=4, patch_size=64, patch_overlap=0,
            device='cuda', num_validation_workers=1):
    sample = sample
    grid_sampler = torchio.inference.GridSampler(
        sample,
        patch_size,
        patch_overlap,
    )
    patch_loader = torch.utils.data.DataLoader(
        grid_sampler, batch_size=validation_batch_size, num_workers=num_validation_workers)

    aggregator = torchio.inference.GridAggregator(grid_sampler)

    model.eval()

    for patches_batch in patch_loader:
        inputs = patches_batch[MRI][DATA].to(device)
        locations = patches_batch['location']
        logits = model(inputs.float())

        aggregator.add_batch(logits, locations)

    prediction = aggregator.get_output_tensor().unsqueeze(0)

    return prediction


@torch.no_grad()
def make_predictions(model, evaluation_set, out_dir, validation_batch_size=4, patch_size=64, patch_overlap=0,
                     device='cuda', num_validation_workers=1, **kwargs):

    out_dir.mkdir(parents=True)
    for i in tqdm(range(len(evaluation_set)), leave=False):
        sample = evaluation_set[i]
        prediction = predict(model, sample, validation_batch_size=validation_batch_size, patch_size=patch_size,
                             patch_overlap=patch_overlap, device=device, num_validation_workers=num_validation_workers)

        uid = sample[MRI]['path'].split('/')[-1][:-7]
        np.save(out_dir / uid, prediction.detach().cpu().numpy())


@torch.no_grad()
def evaluate(model, evaluation_set, metrics, validation_batch_size=4, patch_size=64, patch_overlap=0, device='cuda',
             num_validation_workers=1, **kwargs):

    scores = defaultdict(list)
    for i in tqdm(range(len(evaluation_set)), leave=False):
        sample = evaluation_set[i]
        targets = torch.from_numpy(
            prepare_aseg(sample[LABEL][DATA])
        )

        prediction = predict(model, sample, validation_batch_size=validation_batch_size, patch_size=patch_size,
                             patch_overlap=patch_overlap, device=device, num_validation_workers=num_validation_workers)

        dice_scores = []
        for name, metric in metrics.items():
            scores[name].append(metric(prediction, targets))
            if 'dice' in name:
                dice_scores.append(scores[name][-1])

        scores['dice'].append(np.mean(dice_scores))

    return scores


def train(experiment, num_epochs, training_loader, validation_set, model, optimizer, criterions, metrics,
          scheduler=None, save_path='model.pth', scaler=None, **kwargs):
    scores = evaluate(model, validation_set, metrics, **kwargs)

    for key in scores.keys():
        scores[key] = np.mean(scores[key])
        experiment.log_metric(f"avg_val_{key}", scores[key], step=0, epoch=0)

    best_dice = scores['dice']
    print(f"Validation mean score: DICE {scores['dice']:0.3f}")

    step_counter = Counter()
    torch.save(model.state_dict(), save_path)
    for epoch_idx in range(1, num_epochs + 1):
        print('\nStarting epoch', epoch_idx)
        run_epoch(experiment, epoch_idx, Action.TRAIN, training_loader, model, optimizer, step_counter, criterions,
                  scaler=scaler, scheduler=scheduler)

        scores = evaluate(model, validation_set, metrics, **kwargs)
        for key in scores.keys():
            scores[key] = np.mean(scores[key])
            experiment.log_metric(f"avg_val_{key}", scores[key], step=epoch_idx, epoch=epoch_idx)

        print(f"Validation mean score: DICE {scores['dice']:0.3f}")

        avg_dice = scores['dice']
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), save_path)


def run_epoch(experiment, epoch_idx, action, loader, model, optimizer, step_counter, criterions,
              scheduler=None, device='cuda', scaler=None):
    is_training = action == Action.TRAIN

    epoch_losses = []
    model.train(is_training)

    for batch_idx, batch in enumerate(tqdm(loader, leave=False)):
        inputs, targets, distmaps = prepare_batch(batch, device)
        if is_training and scaler:
            inputs, targets, = inputs.to(torch.float16), targets.to(torch.float16)
            if distmaps:
                distmaps = distmaps.to(torch.float16)

        optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            with autocast(is_training):
                logits = model(inputs.float())
                batch_losses = dict()
                for name, criterion in criterions.items():
                    batch_losses[name] = criterion(logits, targets, distmaps)

            loss = 0.
            for batch_loss in batch_losses.values():
                loss += batch_loss

            if is_training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            epoch_losses.append(loss.item())
            prefix = 'train' if action == Action.TRAIN else 'val'
            for name, value in batch_losses.items():
                experiment.log_metric(f"{prefix}_{name}", value.item(), epoch=epoch_idx, step=step_counter[action])

            step_counter[action] += 1

    epoch_losses = np.array(epoch_losses)
    avg_loss = epoch_losses.mean()

    if action == Action.TRAIN:
        experiment.log_metric("avg_train_loss", avg_loss, step=epoch_idx, epoch=epoch_idx)
        if scheduler:
            scheduler.step(avg_loss)

    elif action == Action.VALIDATE:
        experiment.log_metric("avg_val_dice", 1 - avg_loss, step=epoch_idx, epoch=epoch_idx)

    print(f'{action.value} mean loss: {avg_loss:0.3f}')
    return avg_loss
