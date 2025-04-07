import pathlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from model import CAModel
from Network import CAModel

# Set the parameters directly in the script
image_path = './Carlos.png'
batch_size = 2
device_type = 'cuda'
eval_frequency = 150
eval_iterations = 100
n_batches = 20
n_channels = 16
log_directory = 'logs'
padding_size = 16
pool_size = 1024
image_size = 64


def load_image(path, size=64):
    img = Image.open(path)
    img = img.resize((size, size), Image.LANCZOS)
    img = np.float32(img) / 255.0
    img[..., :3] *= img[..., 3:]
    return torch.from_numpy(img).permute(2, 0, 1)[None, ...]


def to_rgb(img_rgba):
    rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:, ...], 0, 1)
    return torch.clamp(1.0 - a + rgb, 0, 1)


def make_seed(size, n_channels):
    x = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    x[:, 3:, size // 2, size // 2] = 1
    return x


def main():
    # Misc
    device = torch.device(device_type)
    log_path = pathlib.Path(log_directory)
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)

    # Target image
    target_img_ = load_image(image_path, size=image_size)
    p = padding_size
    target_img_ = nn.functional.pad(target_img_, (p, p, p, p), "constant", 0)
    target_img = target_img_.to(device)
    target_img = target_img.repeat(batch_size, 1, 1, 1)

    # Model and optimizer
    model = CAModel(n_channels=n_channels, device=device, n_inpt=n_channels, input_shape=[1, 28, 28],
                    kernel_size=(2, 2), n_filters=n_channels, stride=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    # Pool initialization
    seed = make_seed(image_size, n_channels).to(device)
    seed = nn.functional.pad(seed, (p, p, p, p), "constant", 0)
    pool = seed.clone().repeat(pool_size, 1, 1, 1)

    for it in tqdm(range(n_batches)):
        batch_ixs = np.random.choice(pool_size, batch_size, replace=False).tolist()
        x = pool[batch_ixs]
        for i in range(np.random.randint(64, 96)):
            x = model(x)

        loss_batch = ((target_img - x[:, :4, ...]) ** 2).mean(dim=[1, 2, 3])
        loss = loss_batch.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("train/loss", loss, it)

        argmax_batch = loss_batch.argmax().item()
        argmax_pool = batch_ixs[argmax_batch]
        remaining_batch = [i for i in range(batch_size) if i != argmax_batch]
        remaining_pool = [i for i in batch_ixs if i != argmax_pool]

        pool[argmax_pool] = seed.clone()
        pool[remaining_pool] = x[remaining_batch].detach()

        if it % eval_frequency == 0:
            x_eval = seed.clone()  # (1, n_channels, size, size)
            eval_video = torch.empty(1, eval_iterations, 3, *x_eval.shape[2:])
            for it_eval in range(eval_iterations):
                x_eval = model(x_eval)
                x_eval_out = to_rgb(x_eval[:, :4].detach().cpu())
                eval_video[0, it_eval] = x_eval_out
            writer.add_video("eval", eval_video, it, fps=60)


if __name__ == "__main__":
    main()
