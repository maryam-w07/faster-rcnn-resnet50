import torch
from torchvision.ops import box_iou

def calculate_map(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP) for a batch of images.
    """
    if not pred_boxes or not gt_boxes:
        return 0.0  # Return 0 if no ground truth or predictions

    all_ap = []  # Store AP for each image

    for p_boxes, p_scores, g_boxes in zip(pred_boxes, pred_scores, gt_boxes):
        if len(g_boxes) == 0:
            all_ap.append(0.0)
            continue

        # Sort predicted boxes by confidence scores in descending order
        sorted_indices = torch.argsort(p_scores, descending=True)
        p_boxes = p_boxes[sorted_indices]

        # Compute IoU matrix
        ious = box_iou(p_boxes, g_boxes)

        # Track assignments to avoid duplicate true positives
        assigned_gt = torch.zeros(len(g_boxes), dtype=torch.bool)

        true_positive = torch.zeros(len(p_boxes))
        false_positive = torch.zeros(len(p_boxes))

        for i, iou_row in enumerate(ious):
            max_iou, max_idx = torch.max(iou_row, dim=0)
            if max_iou > iou_threshold and not assigned_gt[max_idx]:
                true_positive[i] = 1  # Correct detection
                assigned_gt[max_idx] = True  # Mark this GT as assigned
            else:
                false_positive[i] = 1  # False detection

        # Compute cumulative sums for precision-recall curve
        accumulated_tp = torch.cumsum(true_positive, dim=0)
        accumulated_fp = torch.cumsum(false_positive, dim=0)

        precision = accumulated_tp / (accumulated_tp + accumulated_fp + 1e-6)  # Avoid division by zero
        recall = accumulated_tp / len(g_boxes)

        # Add interpolation to precision-recall curve
        precision = torch.cat((torch.tensor([1.0]), precision))
        recall = torch.cat((torch.tensor([0.0]), recall))

        # Compute AP using area under curve
        ap = torch.trapz(precision, recall)
        all_ap.append(ap.item())

    return sum(all_ap) / len(all_ap) if all_ap else 0.0
    def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    total_map = 0.0
    num_batches = 0

    for images, targets in data_loader:
        # Check the batch size
        # print(f"Processing batch of size: {len(images)}")

        images = [img.to(device) for img in images]

        if targets is None or not targets:
            continue  # Skip empty targets

        processed_targets = []
        valid_images = []
        for i, target in enumerate(targets):
            if isinstance(target, dict):
                boxes = target['boxes'].to(device)
                labels = target['labels'].to(device)
                if len(boxes) > 0 and len(labels) > 0:
                    processed_targets.append({'boxes': boxes, 'labels': labels})
                    valid_images.append(images[i])

        if not processed_targets:
            continue

        images = valid_images

        # Forward pass
        loss_dict = model(images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Calculate mAP
        with torch.no_grad():
            model.eval()
            predictions = model(images)
            pred_boxes = [p['boxes'].cpu() for p in predictions if 'boxes' in p]
            pred_scores = [p['scores'].cpu() for p in predictions if 'scores' in p]
            gt_boxes = [t['boxes'].cpu() for t in processed_targets]

            # Calculate mAP for the batch
            map_score = calculate_map(pred_boxes, pred_scores, gt_boxes)
            total_map += map_score
            num_batches += 1
            model.train()

    # Calculate average loss and mAP
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_map = total_map / num_batches if num_batches > 0 else 0

    print(f"Epoch [{epoch}] Loss: {avg_loss:.4f}, mAP: {avg_map * 100:.2f}%")

num_epochs = 5
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch)
    lr_scheduler.step()

    # Save the model's state dictionary after every epoch
    model_path = f"fasterrcnn_resnet50_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

