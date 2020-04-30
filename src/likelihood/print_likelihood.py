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
import numpy as np

from src.likelihood.run_likelihood import prepare_parser
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # get the number of components from the run_likelihood.py settings
    parser = prepare_parser()
    args = parser.parse_args()

    n_components = np.arange(0, args.max_number_of_components, 5)

    pca_scores_1 = np.load("output/division_1/pca_scores.npy")
    n_components_pca = n_components[np.argmax(pca_scores_1)]
    pca_scores_2 = np.load("output/division_2/pca_scores.npy")
    n_components_pca_2 = n_components[np.argmax(pca_scores_2)]
    pca_scores_4 = np.load("output/division_4/pca_scores.npy")[:-1]
    n_components_pca_4 = n_components[np.argmax(pca_scores_4)]
    pca_scores_8 = np.load("output/division_8/pca_scores.npy")
    n_components_pca_8 = n_components[np.argmax(pca_scores_8)]

    sns.set()
    sns.set_style("whitegrid", {'axes.grid': False})
    plt.figure(figsize=(12, 6))
    plt.plot(n_components[:len(pca_scores_1)], pca_scores_1, 'b', label='pPCA scores D1')
    plt.plot(n_components[:len(pca_scores_2)], pca_scores_2, 'r', label='pPCA scores D2')
    plt.plot(n_components[:len(pca_scores_4)], pca_scores_4, 'c', label='pPCA scores D4')
    plt.plot(n_components[:len(pca_scores_8)], pca_scores_8, 'm', label='pPCA scores D8')
    plt.axvline(n_components_pca, color='b',
                label='pPCA D1 CV: %d' % n_components_pca, linestyle='--')
    plt.axvline(n_components_pca_2, color='r',
                label='pPCA D2 CV: %d' % n_components_pca_2, linestyle='--')
    plt.axvline(n_components_pca_4, color='c',
                label='pPCA D4 CV: %d' % n_components_pca_4, linestyle='--')
    plt.axvline(n_components_pca_8, color='m',
                label='pPCA D8 CV: %d' % n_components_pca_8, linestyle='--')
    plt.xlabel('nb of components')
    plt.ylabel('CV scores')
    plt.legend(loc='lower right')
    # plt.ylim(-300000, 500)
    plt.title("pPCA scores")
    sns.despine()
    # plt.show()
    plt.savefig("pPCA_scores.pdf", bbox_inches='tight')