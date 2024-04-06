import torch.nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time
#import lpips
from torch.utils.tensorboard import SummaryWriter
from .dataset.datasets import DIV2K, BenchmarkDataset
from .models.model_arch import SRModel, param_count
from .models.metrics import *


def train(cfg):
    # Load training/validation dataset
    train_dataset = DIV2K(cfg, "train")
    val_dataset = DIV2K(cfg, "val")

    # Calculate total epochs
    iters_per_epoch = len(train_dataset) // cfg.TRAIN.BATCH
    epochs = int(cfg.TRAIN.MAX_ITER // iters_per_epoch)
    print("Total epochs: {}".format(epochs))

    # Initialize model, optimizer and scheduler
    model = SRModel(cfg.MODEL.CIN,
                    cfg.MODEL.CMID,
                    cfg.MODEL.CUP,
                    cfg.MODEL.COUT,
                    cfg.MODEL.N_BLOCK).to(device=cfg.DEVICE)
    print(param_count(model))
    # model.load_state_dict(torch.load("D:\deeplearning\AMLS2\A\models\pre.pth"))
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg.TRAIN.ETA_MIN)

    # Start training loop
    train_loop(cfg, epochs, train_dataset, val_dataset, model, optimizer, scheduler)


def train_loop(cfg, epochs, train_dataset, val_dataset, model, optimizer, scheduler):
    # Initialize tensorboard writer
    writer = SummaryWriter("./runs/lr_{}_mid_{}_up_{}_seed_{}".format(cfg.TRAIN.LR,
                                                                      cfg.MODEL.CMID,
                                                                      cfg.MODEL.CUP,
                                                                      cfg.RAND_SEED))

    # Initialize training/validation dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH, shuffle=False, drop_last=False)

    # Initialize mse, ssim loss functions
    l1_loss_fn = L1LOSS().to(device=cfg.DEVICE)
    ssim_loss_fn = SSIMLoss(window_size=11, device=cfg.DEVICE)
    psnr_fn = PSNR().to(device=cfg.DEVICE)
    #lpips_loss_fn = lpips.LPIPS(net="vgg", eval_mode=False).to(device=cfg.DEVICE)

    for epoch in range(1, epochs + 1):
        model.train()

        l1_loss_epoch = 0
        ssim_loss_epoch = 0
        ssim_epoch = 0
        psnr_epoch = 0
        lpips_loss_epoch = 0

        t1 = time.time()

        for lr, hr in train_dataloader:
            b = lr.shape[0]
            lr = lr.to(device=cfg.DEVICE)
            hr = hr.to(device=cfg.DEVICE)
            pred_sr = model(lr)

            l1_loss = l1_loss_fn(pred_sr, hr)
            ssim_loss, ssim = ssim_loss_fn(pred_sr, hr)
            #lpips_loss = lpips_loss_fn.forward(pred_sr, hr)
            psnr = psnr_fn(pred_sr.detach(), hr.detach())
            loss = l1_loss + ssim_loss

            l1_loss_epoch += l1_loss.item() * b
            ssim_loss_epoch += ssim_loss.item() * b
            ssim_epoch += ssim.item() * b
            psnr_epoch += psnr.item() * b
            #lpips_loss_epoch += lpips_loss.item() * b
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Calculate losses and metrics for each epoch
        l1_loss_epoch /= len(train_dataset)
        ssim_loss_epoch /= len(train_dataset)
        lpips_loss_epoch /= len(train_dataset)
        total_loss_epoch = l1_loss_epoch + ssim_loss_epoch + lpips_loss_epoch
        ssim_epoch /= len(train_dataset)
        psnr_epoch /= len(train_dataset)

        t2 = time.time()

        print("epoch: {} time spent: {}s total_loss: {} "
              "mse_loss: {} ssim_loss: {} lpips_loss: {} ssim: {} psnr: {}".format(epoch,
                                                                                   t2 - t1,
                                                                                   total_loss_epoch,
                                                                                   l1_loss_epoch,
                                                                                   ssim_loss_epoch,
                                                                                   lpips_loss_epoch,
                                                                                   ssim_epoch,
                                                                                   psnr_epoch))
        # Perform validation

        print("Start validation at epoch {}".format(epoch))
        model.eval()

        l1_loss_epoch_val = 0
        ssim_loss_epoch_val = 0
        lpips_loss_epoch_val = 0
        ssim_epoch_val = 0
        psnr_epoch_val = 0

        for lr, hr in val_dataloader:
            b = lr.shape[0]
            lr = lr.to(device=cfg.DEVICE)
            hr = hr.to(device=cfg.DEVICE)

            pred_sr = model(lr)

            l1_loss_val = l1_loss_fn(pred_sr, hr)
            ssim_loss_val, ssim_val = ssim_loss_fn(pred_sr, hr)
            #lpips_loss_val = lpips_loss_fn.forward(pred_sr, hr)
            psnr_val = psnr_fn(pred_sr.detach(), hr.detach())

            l1_loss_epoch_val += l1_loss_val.item() * b
            ssim_loss_epoch_val += ssim_loss_val.item() * b
            ssim_epoch_val += ssim_val.item() * b
            psnr_epoch_val += psnr_val.item() * b
            #lpips_loss_epoch_val += lpips_loss_val * b
        l1_loss_epoch_val /= len(val_dataset)
        ssim_loss_epoch_val /= len(val_dataset)
        lpips_loss_epoch_val /= len(val_dataset)
        total_loss_epoch_val = l1_loss_epoch_val + ssim_loss_epoch_val + lpips_loss_epoch_val
        ssim_epoch_val /= len(val_dataset)
        psnr_epoch_val /= len(val_dataset)

        print("Validation of epoch {}, total_loss: {} "
              "mse_loss: {} ssim_loss: {} lpips_loss: {} ssim: {} psnr: {}".format(epoch,
                                                                                   total_loss_epoch_val,
                                                                                   l1_loss_epoch_val,
                                                                                   ssim_loss_epoch_val,
                                                                                   lpips_loss_epoch_val,
                                                                                   ssim_epoch_val,
                                                                                   psnr_epoch_val
                                                                                   ))

        # Log data into tensorboard writer
        writer.add_scalars(main_tag="l1_loss",
                           tag_scalar_dict={"train": l1_loss_epoch,
                                            "val": l1_loss_epoch_val},
                           global_step=epoch)
        writer.add_scalars(main_tag="ssim_loss",
                           tag_scalar_dict={"train": ssim_loss_epoch,
                                            "val": ssim_loss_epoch_val},
                           global_step=epoch)
        writer.add_scalars(main_tag="lpips_loss",
                           tag_scalar_dict={"train": lpips_loss_epoch,
                                            "val": lpips_loss_epoch_val})
        writer.add_scalars(main_tag="total_loss",
                           tag_scalar_dict={"train": total_loss_epoch,
                                            "val": total_loss_epoch_val},
                           global_step=epoch)
        writer.add_scalars(main_tag="ssim_epoch",
                           tag_scalar_dict={"train": ssim_epoch,
                                            "val": ssim_epoch_val},
                           global_step=epoch)
        writer.add_scalars(main_tag="psnr_epoch",
                           tag_scalar_dict={"train": psnr_epoch,
                                            "val": psnr_epoch_val},
                           global_step=epoch)

    model_name = "model_lr_{}_mid_{}_up_{}_seed_{}.pth".format(cfg.TRAIN.LR,
                                                               cfg.MODEL.CMID,
                                                               cfg.MODEL.CUP,
                                                               cfg.RAND_SEED)

    torch.save(model, cfg.SAVE_DIR + model_name)

    print("Model is saved at {}".format(cfg.SAVE_DIR + model_name))


def test(cfg, dataset, trial_time):
    # Initialize test dataset
    test_dataset = BenchmarkDataset(cfg, name=dataset, length=1400)

    model_name = "model_lr_{}_mid_{}_up_{}_seed_{}.pth".format(cfg.TRAIN.LR,
                                                               cfg.MODEL.CMID,
                                                               cfg.MODEL.CUP,
                                                               cfg.RAND_SEED)
    # Load trained model
    model = torch.load(cfg.SAVE_DIR + model_name)
    epochs = int(trial_time // len(test_dataset))
    test_loop(cfg, epochs, test_dataset, model)


def test_loop(cfg, epochs, test_dataset, model):
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH, drop_last=False)
    ssim_loss_fn = SSIMLoss(window_size=11, device=cfg.DEVICE)
    psnr_fn = PSNR().to(device=cfg.DEVICE)

    ssim_total = 0
    psnr_total = 0

    for epoch in range(1, epochs + 1):
        ssim_epoch = 0
        psnr_epoch = 0
        model.eval()
        t1 = time.time()
        for lr, hr in test_dataloader:
            b = lr.shape[0]
            lr = lr.to(device=cfg.DEVICE)
            hr = hr.to(device=cfg.DEVICE)
            pred_sr = model(lr)
            psnr = psnr_fn(pred_sr.detach(), hr.detach())
            _, ssim = ssim_loss_fn(pred_sr, hr)

            ssim_epoch += ssim.item() * b
            psnr_epoch += psnr.item() * b

        ssim_total += ssim_epoch
        psnr_total += psnr_epoch

        ssim_epoch /= len(test_dataset)
        psnr_epoch /= len(test_dataset)

        t2 = time.time()

        print("epoch: {} time spent: {}s ssim: {} psnr: {}".format(epoch,
                                                                   t2 - t1,
                                                                   ssim_epoch,
                                                                   psnr_epoch))
    ssim_total /= (epochs * len(test_dataset))
    psnr_total /= (epochs * len(test_dataset))

    print("The average ssim is: {}".format(ssim_total))
    print("The average psnr is: {}".format(psnr_total))


def convert_weight(cfg):
    model = SRModel(cfg.MODEL.CIN,
                    cfg.MODEL.CMID,
                    cfg.MODEL.CUP,
                    cfg.MODEL.COUT,
                    cfg.MODEL.N_BLOCK).to(device=cfg.DEVICE)
