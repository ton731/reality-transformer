import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .metrics import CycleGANMetric


def plot_metric(train_metric: CycleGANMetric, valid_metric: CycleGANMetric, 
                A_name, B_name, checkpoint_dir: Path) -> None:

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Discriminator
    axs[0].plot(train_metric.discriminatorA_predict_realA, label="train_discA_real", color="black", linewidth=2)
    axs[0].plot(train_metric.discriminatorA_predict_fakeA, label="train_discA_fake", color="gray", linewidth=2)
    axs[0].plot(valid_metric.discriminatorA_predict_realA, label="valid_discA_real", color="red", linewidth=2)
    axs[0].plot(valid_metric.discriminatorA_predict_fakeA, label="valid_discA_fake", color="orange", linewidth=2)
    axs[0].set_title(f"Discriminator: {A_name}")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Probability")
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(train_metric.discriminatorB_predict_realB, label="train_discB_real", color="black", linewidth=2)
    axs[1].plot(train_metric.discriminatorB_predict_fakeB, label="train_discB_fake", color="gray", linewidth=2)
    axs[1].plot(valid_metric.discriminatorB_predict_realB, label="valid_discB_real", color="red", linewidth=2)
    axs[1].plot(valid_metric.discriminatorB_predict_fakeB, label="valid_discB_fake", color="orange", linewidth=2)
    axs[1].set_title(f"Discriminator: {B_name}")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Probability")
    axs[1].grid()
    axs[1].legend()

    plt.savefig(checkpoint_dir / "discriminator.png")
    plt.close()

    # generator: Cycle
    plt.figure(figsize=(8, 6))
    plt.plot(train_metric.generator_A2B2A, label="train_A2B2A", color='black', linewidth=2)
    plt.plot(train_metric.generator_B2A2B, label="train_B2A2B", color='gray', linewidth=2)
    plt.plot(valid_metric.generator_A2B2A, label="valid_A2B2A", color='red', linewidth=2)
    plt.plot(valid_metric.generator_B2A2B, label="valid_B2A2B", color='orange', linewidth=2)
    plt.legend()
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("Cycle Consistency")
    plt.savefig(checkpoint_dir / "cycle.png")
    plt.close()

    # Identity
    if train_metric.use_identity:
        plt.figure(figsize=(8, 6))
        plt.plot(train_metric.identity_A2A, label="train_A2A", color='black', linewidth=2)
        plt.plot(train_metric.identity_B2B, label="train_B2B", color='gray', linewidth=2)
        plt.plot(valid_metric.identity_A2A, label="valid_A2A", color='red', linewidth=2)
        plt.plot(valid_metric.identity_B2B, label="valid_B2B", color='orange', linewidth=2)
        plt.legend()
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("SSIM")
        plt.title("Identity Consistency")
        plt.savefig(checkpoint_dir / "identity.png")
        plt.close()