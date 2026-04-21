import os
import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, f1_score


def _build_real_node_mask(batch):
    mask = torch.ones(batch.x.size(0), dtype=torch.bool, device=batch.x.device)
    _, counts = batch.batch.unique(return_counts=True)
    offset = 0
    for c in counts:
        mask[offset + c - 1] = False
        offset += c
    return mask


def build_criterion(task):
    if task == 'regression':
        return nn.L1Loss()
    elif task == 'multilabel_classification':
        return nn.BCEWithLogitsLoss()
    elif task == 'node_classification':
        return nn.CrossEntropyLoss()
    elif task == 'graph_classification':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown task: {task}")


def build_scheduler(optimizer, config):
    epochs = config['training']['epochs']
    warmup = config['training']['warmup_epochs']

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup]
    )
    return scheduler


def train_epoch(model, loader, optimizer, criterion, device, task, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch, collect_diagnostics=False)

        if task == 'regression':
            target = batch.y.float()
        elif task == 'multilabel_classification':
            target = batch.y.float()
        elif task == 'node_classification':
            target = batch.y.long()
            if target.dim() > 1:
                target = target.squeeze(-1)
            if hasattr(batch, 'vnode_idx'):
                pred = pred[_build_real_node_mask(batch)]
        elif task == 'graph_classification':
            target = batch.y.long()
            if target.dim() > 1:
                target = target.squeeze(-1)

        loss = criterion(pred, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        n = batch.num_graphs if task not in ('node_classification',) else batch.y.size(0)
        total_loss += loss.item() * n
        total_samples += n

    return total_loss / total_samples


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, task):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch, collect_diagnostics=False)

        if task == 'regression':
            target = batch.y.float()
        elif task == 'multilabel_classification':
            target = batch.y.float()
        elif task == 'node_classification':
            target = batch.y.long()
            if target.dim() > 1:
                target = target.squeeze(-1)
            if hasattr(batch, 'vnode_idx'):
                pred = pred[_build_real_node_mask(batch)]
        elif task == 'graph_classification':
            target = batch.y.long()
            if target.dim() > 1:
                target = target.squeeze(-1)

        loss = criterion(pred, target)

        n = batch.num_graphs if task not in ('node_classification',) else target.size(0)
        total_loss += loss.item() * n
        total_samples += n

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    avg_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if task == 'regression':
        metric_val = avg_loss
        metric_name = 'mae'
    elif task == 'multilabel_classification':
        probs = torch.sigmoid(all_preds).numpy()
        targets_np = all_targets.numpy()
        metric_val = average_precision_score(targets_np, probs, average='macro')
        metric_name = 'ap'
    elif task == 'node_classification':
        pred_labels = all_preds.argmax(dim=-1).numpy()
        targets_np = all_targets.numpy()
        metric_val = f1_score(targets_np, pred_labels, average='macro', zero_division=0)
        metric_name = 'f1'
    elif task == 'graph_classification':
        pred_labels = all_preds.argmax(dim=-1).numpy()
        targets_np = all_targets.numpy()
        metric_val = (pred_labels == targets_np).mean()
        metric_name = 'accuracy'

    return avg_loss, metric_val, metric_name


def is_better(metric_val, best_val, task):
    if task == 'regression':
        return metric_val < best_val
    else:
        return metric_val > best_val


def train_model(model, train_loader, val_loader, config, device, task, save_dir='outputs', use_wandb=True):
    epochs = config['training']['epochs']
    lr = config['training']['lr']
    wd = config['training']['weight_decay']
    grad_clip = config['training']['grad_clip']

    criterion = build_criterion(task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = build_scheduler(optimizer, config)

    os.makedirs(save_dir, exist_ok=True)

    best_val_metric = float('inf') if task == 'regression' else 0.0
    train_losses = []
    val_losses = []

    pbar = tqdm(range(1, epochs + 1), desc='Training')

    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, task, grad_clip)
        val_loss, val_metric, metric_name = eval_epoch(model, val_loader, criterion, device, task)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        if is_better(val_metric, best_val_metric, task):
            best_val_metric = val_metric
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

        pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            f'val_{metric_name}': f'{val_metric:.4f}',
            f'best_{metric_name}': f'{best_val_metric:.4f}',
        })

        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                f'val/{metric_name}': val_metric,
                'lr': current_lr,
                f'best/{metric_name}': best_val_metric,
            })

    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pt'))
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt'), weights_only=True))

    return {
        'best_val_metric': best_val_metric,
        'metric_name': metric_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
