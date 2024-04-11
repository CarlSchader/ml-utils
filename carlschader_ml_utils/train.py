import torch

def simCLR_train(
    train_loader, 
    val_loader=None,
    model=models.resnet50(),
    projector=nn.Sequential(nn.Linear(1000, 128), nn.ReLU(), nn.Linear(128, 128)),
    optimizer=None, 
    scheduler=None,
    augment=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        get_color_distortion(),
        transforms.GaussianBlur(kernel_size=3),
    ]),
    criterion=simCLR_criterion, 
    num_epochs=100,
    logger=None, 
    device=DETECTED_DEVICE
):
    logger.debug('Starting training')
    model = model.to(device)
    projector = projector.to(device)
    if optimizer is None:
        optimizer = optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=0.001, weight_decay=0.01)
    if scheduler is None:
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.33, end_factor=1.0, total_iters=5),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0)
        ], milestones=[5])
    
    logger.info('epoch,train_loss,val_loss,lr...')

    for epoch in range(num_epochs):
        train_loss = simCLR_train_iteration(model, train_loader, projector, augment, optimizer, scheduler, criterion, logger, device)
        if val_loader is not None:
            val_loss = simCLR_validate_iteration(model, val_loader, projector, augment, criterion, logger, device)
        else:
            val_loss = 'N/A'
        if logger is not None:
            lr_string = ','.join(map(str, scheduler.get_last_lr()))
            logger.info(f'{epoch+1},{train_loss},{val_loss},{lr_string}')

    logger.debug('Training complete')

    return model, projector
