"""
Generative Fourier-based Auto-Encoders:Preliminary Results
Copyright (C) 2018  Alessandro Zonta (a.zonta@vu.nl)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import argparse
import logging
import os

import numpy as np
import torch
import torchaudio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torchaudio.transforms import GriffinLim

from src.data_loader.torch_loader import TorchLoader
from src.ppca.ppca import PPCA


def prepare_parser():
    parser = argparse.ArgumentParser(description="pPCA")
    parser.add_argument("--source_folder", default="data/only_dog/",
                        help="Folder containing dataset to read")
    parser.add_argument("--output_folder", default="output",
                        help="Folder where to save the results")
    parser.add_argument("--name", default="reconstruction", help="Name experiment")
    parser.add_argument("--seed", default=42, help="Seed experiment")

    parser.add_argument("--division", default=8, choices=[1, 2, 4, 8, 10], type=int,
                        help="How many time divide the dataset")
    parser.add_argument("--sample_rate", type=int, default=8000, help="sample rate data")
    parser.add_argument("--components", type=int, default=305, help="-1 to use .95, or number >0")
    parser.add_argument("--samples_to_generate", type=int, default=50, help="how many samples to generate")
    parser.add_argument("--low_resolution", action='store_true', help=".45 instead of .95")
    parser.add_argument("--debug", action='store_true', help="store all the conversion data")
    return parser

if __name__ == '__main__':
    logger = logging.getLogger("runner")
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    parser = prepare_parser()
    args = parser.parse_args()

    # set the seed
    np.random.seed(args.seed)

    path = args.source_folder
    output_folder = "{}/{}_{}_{}_{}/".format(args.output_folder, args.name, args.division, args.sample_rate,
                                             args.components)
    args.output_folder = output_folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # load audio
    logger.info("Loading Audio")
    audio = TorchLoader(path_source=path, log=logger, sample_rate=args.sample_rate)

    x_train, train_y, x_val, val_y, x_test, test_y = audio.load_data(source_path=path).load_files().save_data(
        destination_path=path).get_data()

    x_bigger_train = np.array(np.split(x_train.numpy(), args.division, axis=1))
    x_bigger_train = np.reshape(x_bigger_train, (-1, x_bigger_train.shape[2]))

    train_set = torch.from_numpy(x_bigger_train)

    # convert the entire dataset
    spectrogram_space = torchaudio.transforms.Spectrogram()(train_set)

    if args.debug:
        np.save("spectrogram_space", spectrogram_space)

    original_dimenstion = spectrogram_space.size()[1:]
    spectrogram_space = spectrogram_space.view(spectrogram_space.size()[0], -1).numpy()

    # scale dataset to 0-1
    scaler = MinMaxScaler()
    scaler.fit(spectrogram_space)
    spectrogram_space_scaled = scaler.transform(spectrogram_space)
    spectrogram_space_scaled = spectrogram_space_scaled + np.finfo(float).eps
    spectrogram_space_scaled = np.log(spectrogram_space_scaled)

    logger.info("fitting pPCA")

    if args.components == -1:
        if args.low_resolution:
            comp = .45
        else:
            comp = .95
        pca_for_components = PCA(comp)
        pca_for_components.fit(spectrogram_space_scaled)
        components = pca_for_components.n_components_
    else:
        components = args.components

    # now fitting real ppca
    ppca = PPCA(n_dimension=args.components)
    ppca.fit(spectrogram_space_scaled, method='eig')
    small_dataset = ppca.transform(spectrogram_space_scaled, probabilistic=True)
    back_from_transformed = ppca.inverse_transform(small_dataset, probabilistic=True)
    back_from_transformed = np.exp(back_from_transformed)
    back_from_transformed = back_from_transformed - np.finfo(float).eps

    np.save("reconstructed_spectrogram", back_from_transformed)

    converted_data = torch.from_numpy(back_from_transformed).view(back_from_transformed.shape[0],
                                                                  original_dimenstion[0],
                                                                  original_dimenstion[1])
    reconstructed_raw = GriffinLim()(converted_data.float()).numpy()
    np.save("reconstructed_raw", reconstructed_raw)

    logger.info("Generating Data ")
    data_generated = ppca.generate(n_sample=args.samples_to_generate)
    data_generated = np.exp(data_generated)
    data_generated = data_generated - np.finfo(float).eps

    if args.debug:
        np.save("generated_spectrogram", data_generated)

    converted_data = torch.from_numpy(data_generated).view(args.samples_to_generate, original_dimenstion[0],
                                                           original_dimenstion[1])

    generated_raw = GriffinLim()(converted_data.float()).numpy()
    np.save("{}/generated_audio".format(args.output_folder), generated_raw)
