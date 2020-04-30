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
import logging
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader.torch_loader import TorchLoader
from src.ppca.experiment_ppca import prepare_parser

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
    path = args.source_folder
    path_to_read = "output/"

    # load audio
    audio = TorchLoader(path_source=path, log=logger, sample_rate=args.sample_rate)

    x_train, train_y, x_val, val_y, x_test, test_y = audio.load_data(source_path=path).load_files().save_data(
        destination_path=path).get_data()

    x_bigger_train = np.array(np.split(x_train.numpy(), args.division, axis=1))
    x_bigger_train = np.reshape(x_bigger_train, (-1, x_bigger_train.shape[2]))

    train_set = torch.from_numpy(x_bigger_train)
    # convert the entire dataset
    spectrogram_space = torchaudio.transforms.Spectrogram()(train_set)
    original_dimensions = spectrogram_space.size()[1:]

    # read reconstructed spectrogram
    generated_data_spectrogram = np.load("{}/generated_spectrogram.npy".format(path_to_read))
    reconstructed_data = torch.from_numpy(generated_data_spectrogram).view(generated_data_spectrogram.shape[0],
                                                                           original_dimensions[0],
                                                                           original_dimensions[1])
    # read reconstructed raw data
    generated_raw_data = np.load("{}/generated_audio.npy".format(path_to_read))

    sns.set()
    plt.figure(figsize=(10, 10))

    idx_image = 0
    number_or_rows = 4
    number_of_columns = 4
    total = (number_of_columns * number_or_rows) - 1
    for i in range(0, total, number_of_columns):
        j = i + 1
        plt.subplot(number_or_rows, number_of_columns, j)
        plt.plot(x_train[idx_image])
        if i == 0:
            plt.title("Original Signal")
        plt.axis('off')

        plt.subplot(number_or_rows, number_of_columns, j + 1)
        fig = plt.imshow(np.log(np.transpose(spectrogram_space[idx_image].numpy())))
        if i == 0:
            plt.title("Spectrogram")
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.subplot(number_or_rows, number_of_columns, j + 2)
        fig_a = plt.imshow(np.log(np.transpose(reconstructed_data[idx_image])))
        if i == 0:
            plt.title("Generated Spectrogram")
        fig_a.axes.get_xaxis().set_visible(False)
        fig_a.axes.get_yaxis().set_visible(False)

        plt.subplot(number_or_rows, number_of_columns, j + 3)
        plt.plot(generated_raw_data[idx_image])
        if i == 0:
            plt.title("Generated Signal")
        plt.axis('off')

        idx_image += 1

    plt.tight_layout()
    sns.despine()
    # plt.savefig("reconstruction.pdf", bbox_inches='tight')
    plt.show()
