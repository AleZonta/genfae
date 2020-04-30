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

from sklearn import decomposition
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from src.data_loader.torch_loader import TorchLoader


def prepare_parser():
    parser = argparse.ArgumentParser(description="pPCA Likelihood")
    parser.add_argument("--source_folder", default="data/only_dog/",
                        help="Folder containing dataset to read")
    parser.add_argument("--output_folder", default="output",
                        help="Folder where to save the results")
    parser.add_argument("--name", default="test", help="Name experiment")
    parser.add_argument("--seed", default=42, help="Seed experiment")

    parser.add_argument("--division", default=1, choices=[1, 2, 4, 8, 10],
                        type=int, help="How many time divide the signal")
    parser.add_argument("--sample_rate", type=int, default=8000, help="sample rate data")
    parser.add_argument("--max_number_of_components", type=int, default=2000,
                        help="upperbound number of components to check")
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
    output_folder = "{}/{}_{}_{}/".format(args.output_folder, args.name, args.division, args.sample_rate)
    args.output_folder = output_folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # load audio
    audio = TorchLoader(path_source=path, log=logger, sample_rate=args.sample_rate)

    x_train, train_y, x_val, val_y, x_test, test_y = audio.load_data(source_path=path).load_files().save_data(
        destination_path=path).get_data()

    # divide data
    x_bigger_train = np.array(np.split(x_train.numpy(), args.division, axis=1))
    x_bigger_train = np.reshape(x_bigger_train, (-1, x_bigger_train.shape[2]))
    train_set = torch.from_numpy(x_bigger_train)

    # convert the entire dataset
    spectrogram_space = torchaudio.transforms.Spectrogram()(train_set)
    spectrogram_space = spectrogram_space.view(spectrogram_space.size()[0], -1).numpy()

    # scale dataset to 0-1
    scaler = MinMaxScaler()
    scaler.fit(spectrogram_space)
    spectrogram_space_scaled = scaler.transform(spectrogram_space)
    spectrogram_space_scaled = spectrogram_space_scaled + np.finfo(float).eps
    spectrogram_space_scaled = np.log(spectrogram_space_scaled)

    pca = decomposition.PCA()

    # run likelihood comparison
    n_components = np.arange(0, args.max_number_of_components, 5)
    pca_scores = []
    for n in n_components:
        logger.info("current iteration: {}".format(n))
        pca.n_components = n
        try:
            pca_scores.append(np.mean(cross_val_score(pca, spectrogram_space_scaled)))
        except Exception as e:
            logger.debug(e)

    # this search requires a lot of cores and a lot of time. Better save the data and print it afterwards
    np.save("{}/pca_scores".format(args.output_folder), pca_scores)