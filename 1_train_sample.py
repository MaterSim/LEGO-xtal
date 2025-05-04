import os
import pandas as pd
import numpy as np
import argparse
from lego.GAN import CTGAN
from lego.VAE import TVAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Table Synthesizer')
    parser.add_argument('--data', dest='data',
            help='Input CSV data')
    parser.add_argument('--model', dest='model', default='CTGAN',
            help='Input models: supports CTGAN, TVAE')
    parser.add_argument('--epochs', type=int, default=250,
            help='Number of epochs for training (default: 250)')
    parser.add_argument('--nbatch', dest='nbatch', type=int, default=500,
            help='Number of batch size for training')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
            help='Training seeds')
    parser.add_argument('--cutoff', dest='cutoff', type=int,
            help='Cutoff number for training samples, for testing purposes')
    parser.add_argument('--sample', dest='sample', type=int, default=100000,
            help='Output sample size')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    # Read data
    df = pd.read_csv(args.data)
    if args.cutoff is not None and len(df) > args.cutoff:
        print("Select only a few samples for quick test")
        df = df[:args.cutoff]

    print(f' data shape {df.shape} \n')
    print(f' Data Head \n {df.head()} \n')

    # Set up the categorical columns
    discrete_columns = ['spg']
    num_wps = int((len(df.columns)-7)/4)
    for i in range(num_wps):
        discrete_columns.append('wp'+str(i))

    # Initialize synthesizer with specified parameters
    os.makedirs('models', exist_ok=True)
    model = args.model
    if model == 'CTGAN':
        synthesizer = CTGAN(
                        embedding_dim=128,
                        generator_dim=(256, 256),
                        discriminator_dim=(256, 256),
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
                        cuda=True,
                    )

    elif model == 'TVAE':
        synthesizer = TVAE(
                        embedding_dim=128,
                        compress_dims=(1024, 1024),
                        decompress_dims=(1024, 1024),
                        l2scale=1e-5,
                        loss_factor=2,
                        epochs=args.epochs,
                        verbose=True,
                        cuda=True,
                        batch_size=args.nbatch)
    else:
        raise RuntimeError("Only supports CTGAN/TVAE, not", model)

    # Train models
    synthesizer.fit(df, discrete_columns=discrete_columns)

    # Output is stored in synthetic_data
    if args.sample is None:
        synthetic_data_size = len(df)
    else:
        synthetic_data_size = args.sample
    df_synthetic = synthesizer.sample(samples=synthetic_data_size)

    print(f'(synthetic data sample \n {df_synthetic.head(10)} \n')
    os.makedirs('data', exist_ok=True)
    output_file = f"data/sample/{args.model}.csv"
    print(f'Save {synthetic_data_size} samples to {output_file}')
    df_synthetic.columns = df_synthetic.columns.str.replace(' ', '')
    df_synthetic = df_synthetic.applymap(lambda x: str(x).replace(',', ' '))
    df_synthetic.to_csv(output_file_name, index=False, header=False)
