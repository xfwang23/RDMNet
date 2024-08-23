import random
import torch
from tqdm import tqdm
import torch.nn as nn
from utils.utils import get_lr
from torchvision.utils import save_image


def augment(inp_img):
    res = []
    for img in inp_img:
        aug = random.randint(0, 8)  # 包括0和8
        if aug == 1:
            img = img.flip(1)
        elif aug == 2:
            img = img.flip(2)
        elif aug == 3:
            img = torch.rot90(img, dims=(1, 2))
        elif aug == 4:
            img = torch.rot90(img, dims=(1, 2), k=2)
        elif aug == 5:
            img = torch.rot90(img, dims=(1, 2), k=3)
        elif aug == 6:
            img = torch.rot90(img.flip(1), dims=(1, 2))
        elif aug == 7:
            img = torch.rot90(img.flip(2), dims=(1, 2))
        res.append(img)

    return torch.stack(res, dim=0)


def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, save_period):
    Det_loss = 0
    val_loss = 0
    # Dehazy_loss = 0
    # Contrs_loss = 0
    criterion_l1 = nn.L1Loss().cuda()
    contrast_loss = nn.CrossEntropyLoss().cuda()
    wgt = [1.0, 0.9, 0.8, 0.7, 0.6]

    model_train.train()
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets, clearimgs = batch[0], batch[1], batch[2]
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                clearimgs = torch.from_numpy(clearimgs).type(torch.FloatTensor).cuda()
                posimgs = augment(images)
                hazy_and_clear = torch.cat([images, posimgs], dim=0).cuda()

            optimizer.zero_grad()

            # outputs         = model_train(images)
            detected, restored, logits, labels = model_train(hazy_and_clear)

            loss_det = yolo_loss(detected, targets)
            # for l in range(len(outputs) - 1):
            #     loss_item = yolo_loss(outputs[l], targets)
            #     loss_value_all  += loss_item
            loss_l1 = criterion_l1(restored, clearimgs)
            # save_image(restored[0], './results/dehazing.png')
            # save_image(clearimgs[0], './results/clean.png')
            # save_image(images[0], './results/hazy.png')
            loss_contrs = contrast_loss(logits, labels)
            total_loss = 0.2 * loss_det + wgt[epoch // 5] * loss_l1 + 0.1 * loss_contrs

            total_loss.backward()
            optimizer.step()

            Det_loss += loss_det.item()
            # Dehazy_loss += loss_l1.item()
            # Contrs_loss += loss_contrs.item()

            pbar.set_postfix(**{'loss_det': f'{loss_det:.2f}',
                                'loss_l1': f'{loss_l1:.2f}',
                                'loss_contrs': f'{loss_contrs:.2f}',
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    # print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():

                images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]

                optimizer.zero_grad()

                detected, restored = model_train(images)

                det_loss = yolo_loss(detected, targets)
                # for l in range(len(outputs)-1):
                #     loss_item = yolo_loss(outputs[l], targets)
                #     loss_value_all  += loss_item
                # det_loss = loss_value_all

            val_loss += det_loss.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')

    loss_history.append_loss(epoch + 1, Det_loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (Det_loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, Det_loss / epoch_step, val_loss / epoch_step_val))
