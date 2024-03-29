import torch
import torch.nn as nn

def simCLR_criterion(outputs, temperature=0.1):
    batch_size = outputs.size(0)

    # Calculate cosine similarity
    outputs = outputs / outputs.norm(dim=1)[:, None]
    sim = torch.mm(outputs, outputs.t())

    # Calculate the loss
    mask = torch.eye(batch_size, dtype=torch.bool).to(outputs.device)
    neg_mask = ~mask

    pos = sim[mask].view(batch_size, 1)
    neg = sim[neg_mask].view(batch_size, -1)

    logits = torch.cat([pos, neg], dim=1)
    logits /= temperature

    labels = torch.zeros(batch_size, dtype=torch.long).to(outputs.device)
    return nn.CrossEntropyLoss()(logits, labels)

def simCLR_train(model, train_loader, optimizer, criterion, device, verbose=False):
    model.train()
    total_loss = 0
    total_batches = len(train_loader)
    batches = 0

    for inputs, _ in train_loader:
        inputs = inputs.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

        if verbose:
            print(f'Processed batch... {batches}/{total_batches}', end='\r')

    return total_loss / total_batches
