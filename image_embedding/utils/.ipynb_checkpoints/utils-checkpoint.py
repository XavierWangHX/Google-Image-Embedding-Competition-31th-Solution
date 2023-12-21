import torch
import numpy as np

def load_checkpoint(config, model, logger):
    logger.info(f"==============> Load model from {config.MODEL.finetune}")
    checkpoint = torch.load(config.MODEL.finetune, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    torch.cuda.empty_cache()
    import gc
    gc.collect()

def save_checkpoint(model, save_path, save_optim=False, optimizer=None, epoch=None, config=None):
    if save_optim:
        save_state = {'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'epoch': epoch,
                      'config': config}
    else:
        save_state = {'state_dict': model.state_dict(),
                      'optimizer': {},
                      'epoch': {},
                      'config': {}}
    torch.save(save_state, save_path)

def get_train_epoch_lr(c_epoch, max_epoch, init_lr):
    return 0.5 * init_lr * (1.0 + np.cos(np.pi * c_epoch / max_epoch))

def get_warm_up_lr(warm_up_epochs, c_epoch, warm_up_step, init_lr, iters_per_epoch=None):
    c_step = (c_epoch-1)*iters_per_epoch
    t_step = warm_up_epochs*iters_per_epoch
    alpha = (c_step+warm_up_step)/t_step
    factor = 0.1 * (1.0 - alpha) + alpha
    lr = init_lr*factor
    return lr

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr