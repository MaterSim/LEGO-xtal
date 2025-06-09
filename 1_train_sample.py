import argparse
import os
import numpy as np
import pandas as pd
import torch

from lego.GAN import GAN
from lego.VAE import VAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table Synthesizer")
    parser.add_argument("--data", help="Input CSV data")
    parser.add_argument(
        "--model", 
        default="GAN", 
        help="Models: supports GAN, VAE"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Number of epochs for training (default: 250)",
    )
    parser.add_argument(
        "--nbatch", 
        type=int, 
        default=500, 
        help="Number of batch size for training")
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Training seeds")
    parser.add_argument(
        "--cutoff", 
        type=int, 
        help="Cutoff number for training samples"
    )
    parser.add_argument(
        "--sample", 
        type=int, 
        default=100000, 
        help="Output sample size")

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        print("cuda is available")
        cuda = True
    else:
        cuda = False
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    # Read data
    df = pd.read_csv(args.data)
    if args.cutoff is not None and len(df) > args.cutoff:
        print("Select only a few samples for quick test")
        df = df[: args.cutoff]

    print(f"Data shape {df.shape} \n")
    print(f"Data Head \n {df.head()} \n")

    # Set up the categorical columns
    dis_cols = ["spg"]
    num_wps = int((len(df.columns) - 7) / 4)
    if abs(df['a'][0] - round(df['a'][0])) < 1e-2: 
        discrete_cell = True
        dis_cols.extend(['a', 'b', 'c', 'alpha', 'beta', 'gamma'])
    else:
        discrete_cell = False

    discrete = df['x0'].max() >= 2.5 + 1e-3

    for i in range(num_wps): 
        dis_cols.append('wp' + str(i))
        if discrete:
            dis_cols.append('x' + str(i))
            dis_cols.append('y' + str(i))
            dis_cols.append('z' + str(i))
    #dis_cols.append('label')

    # Initialize synthesizer with specified parameters
    os.makedirs("models", exist_ok=True)
    model = args.model
    if model == "GAN":
        synthesizer = GAN(
            embedding_dim=128,
            generator_dim=(512, 512),
            discriminator_dim=(512, 512),
            generator_lr=2e-4,
            generator_decay=1e-6,
            discriminator_lr=2e-4,
            discriminator_decay=1e-6,
            batch_size=args.nbatch,
            discriminator_steps=1,
            log_frequency=True,
            verbose=True,
            epochs=args.epochs,
            pac=10,
            cuda=cuda,
            folder="models/GAN",
        )

    elif model == "VAE":
        synthesizer = VAE(
            embedding_dim=128,
            compress_dims=(512, 512),
            decompress_dims=(512, 512),
            l2scale=1e-5,
            loss_factor=2,
            epochs=args.epochs,
            verbose=True,
            cuda=cuda,
            batch_size=args.nbatch,
            folder = "models/VAE",
        )
    else:
        raise RuntimeError("Only supports GAN/VAE, not", model)

    # Train models
    synthesizer.fit(df, discrete_columns=dis_cols)

    # Output is stored in synthetic_data
    if args.sample is None:
        synthetic_data_size = len(df)
    else:
        synthetic_data_size = args.sample
    df_synthetic = synthesizer.sample(samples=synthetic_data_size)

    print(f"(synthetic data sample\n {df_synthetic.head(10)}\n")
    os.makedirs("data/sample", exist_ok=True)
    output_file = f"data/sample/{args.model}-dis{len(dis_cols)}-{args.sample}.csv"
    print(f"Save {synthetic_data_size} samples to {output_file}")
    df_synthetic.columns = df_synthetic.columns.str.replace(" ", "")
    df_synthetic = df_synthetic.map(lambda x: str(x).replace(",", " "))
    df_synthetic.to_csv(output_file, index=False, header=True)
