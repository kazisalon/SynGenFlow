from src.generator import Generator
from src.discriminator import Discriminator
from src.data_augmentation import DataAugmentation
from config import CONFIG
import torch


def main():
    # Initialize models
    generator = Generator(CONFIG["latent_dim"], CONFIG["img_channels"])
    discriminator = Discriminator(CONFIG["img_channels"])

    # Add your training loop here
    print("Models initialized successfully!")


if __name__ == "__main__":
    main()
