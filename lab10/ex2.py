import numpy as np
from matplotlib import pyplot as plt

from lab10.ex1 import read_gray, find_max, H, plot_points
from lab10.utils.pm import plot_matches


def describe_points(img, pts, size):
    Y, X = img.shape
    pts = list(filter(lambda pt: size <= pt[0] < Y - size and
                                 size <= pt[1] < X - size, zip(pts[0], pts[1])))
    list_patches = []
    for pt in pts:
        patch = img[pt[0] - size: pt[0] + size + 1, pt[1] - size: pt[1] + size + 1]
        list_patches.append(patch.astype(np.int64).flatten())
    return list(zip(list_patches, pts))


def get_matches(charac1, charac2, n):
    result = []
    for charac1_neigh in charac1:
        result_for_pt = []
        for charac2_neigh in charac2:
            result_for_pt.append((sum(abs(charac1_neigh[0] - charac2_neigh[0])), charac2_neigh[1]))
        result_for_pt.sort(key=lambda x: x[0])
        result.append((result_for_pt[0][0], charac1_neigh[1], result_for_pt[0][1]))
    result.sort(key=lambda x: x[0])
    return result[:n]

def get_matches_afinic(charac1, charac2, n):
    result = []
    for charac1_neigh in charac1:
        result_for_pt = []
        mean_charac1 = np.mean(charac1_neigh[0])
        std_charac1 = np.std(charac1_neigh[0])
        charac1_aff = (charac1_neigh[0] - mean_charac1) / std_charac1
        for charac2_neigh in charac2:
            mean_charac2 = np.mean(charac2_neigh[0])
            std_charac2 = np.std(charac2_neigh[0])
            charac2_aff = (charac2_neigh[0] - mean_charac2) / std_charac2
            result_for_pt.append((sum(abs(charac1_aff - charac2_aff)), charac2_neigh[1]))
        result_for_pt.sort(key=lambda x: x[0])
        result.append((result_for_pt[0][0], charac1_neigh[1], result_for_pt[0][1]))
    result.sort(key=lambda x: x[0])
    return result[:n]

if __name__ == "__main__":

    # Fontanna
    fontanna1 = read_gray('resources/fontanna1.jpg')
    fontanna2 = read_gray('resources/fontanna2.jpg')

    fontanna1_max = find_max(H(fontanna1, 7), 7, 0.4)
    fontanna2_max = find_max(H(fontanna2, 7), 7, 0.4)
    described_points_fontanna1 = describe_points(fontanna1, fontanna1_max, 10)
    described_points_fontanna2 = describe_points(fontanna2, fontanna2_max, 10)
    fontanna_matches = get_matches(described_points_fontanna1, described_points_fontanna2, 20)
    plot_matches(fontanna1, fontanna2, fontanna_matches)
    plt.show()

    # Budynek
    # budynek1 = read_gray('resources/budynek1.jpg')
    # budynek2 = read_gray('resources/budynek2.jpg')
    #
    # budynek1_max = find_max(H(budynek1, 7), 7, 0.5)
    # budynek2_max = find_max(H(budynek2, 7), 7, 0.5)
    #
    # described_points_budynek1 = describe_points(budynek1, budynek1_max, 10)
    # described_points_budynek2 = describe_points(budynek2, budynek2_max, 10)
    # budynek_matches = get_matches(described_points_budynek1, described_points_budynek2, 20)
    # plot_matches(budynek1, budynek2, budynek_matches)
    # plt.show()

    # Fontanna POW
    # fontanna_pow = read_gray('resources/fontanna_pow.jpg')
    #
    # fontanna_pow_max = find_max(H(fontanna_pow, 7), 7, 0.4)
    #
    # described_points_fontanna_pow = describe_points(fontanna_pow, fontanna_pow_max, 15)
    # fontanna_pow_matches = get_matches(described_points_fontanna1, described_points_fontanna_pow, 20)
    # plot_matches(fontanna1, fontanna_pow, fontanna_pow_matches)
    # plt.show()

    # # Eifel
    # eifel1 = read_gray('resources/eiffel1.jpg')
    # eifel2 = read_gray('resources/eiffel2.jpg')
    #
    # eifel1_max = find_max(H(eifel1, 7), 7, 0.5)
    # eifel2_max = find_max(H(eifel2, 7), 7, 0.5)
    #
    # described_points_eifel1 = describe_points(eifel1, eifel1_max, 10)
    # described_points_eifel2 = describe_points(eifel2, eifel2_max, 10)
    # eifel_matches = get_matches(described_points_eifel1, described_points_eifel2, 20)
    # plot_matches(eifel1, eifel2, eifel_matches)
    # plt.show()
    #
    # Afiniczne
    # eifel_afinic_matches = get_matches_afinic(described_points_eifel1, described_points_eifel2, 20)
    # plot_matches(eifel1, eifel2, eifel_afinic_matches)
    # plt.show()
