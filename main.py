# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from src.pipeline import SynGenFlow
import uvicorn

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
model = SynGenFlow()


@app.post("/generate")
async def generate_images(num_images: int = 1):
    # Generate synthetic images
    generated = model.generate_samples(num_images)

    # Convert to base64
    images = []
    for img in generated:
        # Convert tensor to PIL Image
        img = transforms.ToPILImage()(img * 0.5 + 0.5)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images.append(img_str)

    return {"images": images}


@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess image
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    image_tensor = transform(image).unsqueeze(0)

    # Train for one step
    g_loss, d_loss = model.train_step(image_tensor, batch_size=1)

    return {"g_loss": float(g_loss), "d_loss": float(d_loss)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# src/pipeline.py (continuation of previous GAN implementation)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List, Tuple


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, img_channels: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SynGenFlow:
    def __init__(self, latent_dim: int = 100, img_channels: int = 3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, img_channels).to(self.device)
        self.discriminator = Discriminator(img_channels).to(self.device)

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

    def train_step(
        self, real_images: torch.Tensor, batch_size: int
    ) -> Tuple[float, float]:
        real_images = real_images.to(self.device)

        # Train Discriminator
        self.d_optimizer.zero_grad()
        label_real = torch.ones(batch_size, 1).to(self.device)
        label_fake = torch.zeros(batch_size, 1).to(self.device)

        output_real = self.discriminator(real_images)
        d_loss_real = nn.BCELoss()(output_real, label_real)

        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_images = self.generator(z)
        output_fake = self.discriminator(fake_images.detach())
        d_loss_fake = nn.BCELoss()(output_fake, label_fake)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()
        output_fake = self.discriminator(fake_images)
        g_loss = nn.BCELoss()(output_fake, label_real)
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item(), d_loss.item()

    def generate_samples(self, num_samples: int) -> List[torch.Tensor]:
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            samples = self.generator(z)
        return samples.cpu()
