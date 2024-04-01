import torch

def simCLR_criterion(batch1, batch2, temp=0.1):
    batch_size = batch1.size(0)
    norms = torch.cat([batch1, batch2], dim=0)
    norms = norms / norms.norm(dim=1)[:, None]
    sims = torch.mm(norms, norms.t())
    exps = torch.exp(sims / temp)
    sums = exps.sum(dim=0)
    numerators = torch.cat((torch.diagonal(exps, offset=batch_size, dim1=1, dim2=0), torch.diagonal(exps, offset=batch_size, dim1=0, dim2=1)))
    denominators = sums - numerators
    neg_logs = -torch.log(numerators / denominators)
    return neg_logs.mean()

    
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

if __name__ == '__main__':
    import time
    start = time.time()
    batch_size = 10000
    temp = 0.1
    
    b1 = torch.stack([torch.arange(start=(i*batch_size)+1, end=((i+1)*batch_size)+1) for i in range(batch_size)])
    b2 = (0.5 * b1) + 3

    print(simCLR_criterion(b1, b2, temp))
    # print(simCLR_criterion(torch.tensor([
    #     [1.0, 0.0, 1.0],
    #     [-0.5, 0.866, 0.0],
    #     [-0.5, -0.866, 0.0],
    # ]), torch.tensor([
    #     [1.0, 0.0, 0.0],
    #     [-0.5, 0.866, 0.0],
    #     [-0.5, -0.866, 0.0],
    # ]), temp))

    print('Time:', time.time() - start)
