import math
import uuid

import numpy as np

from src.data.pinecone import insert_row


def bic_score(X, labels):
    n_points = len(labels)
    n_clusters = len(set(labels))
    n_dimensions = X.shape[1]

    n_parameters = (n_clusters - 1) + (n_dimensions * n_clusters) + 1

    loglikelihood = 0
    for label_name in set(labels):
        X_cluster = X[labels == label_name]
        n_points_cluster = len(X_cluster)
        centroid = np.mean(X_cluster, axis=0)
        variance = np.sum((X_cluster - centroid) ** 2) / (len(X_cluster) - 1)
        loglikelihood += (
            n_points_cluster * np.log(n_points_cluster)
            - n_points_cluster * np.log(n_points)
            - n_points_cluster * n_dimensions / 2 * np.log(2 * math.pi * variance)
            - (n_points_cluster - 1) / 2
        )

    bic = loglikelihood - (n_parameters / 2) * np.log(n_points)

    return bic


def num_to_min(n):
    hours = str(n // 3600)
    hours = "0" * (2 - len(hours)) + hours

    minuts = str(n // 60)
    minuts = "0" * (2 - len(minuts)) + minuts

    seconds = str(n % 60)
    seconds = "0" * (2 - len(seconds)) + seconds

    return f"{hours}:{minuts}:{seconds},000"


def generate_subtitles(labels, user, embedings, index, cluster_names):
    counter = 1
    res = ""
    prev = -1
    curr_time = 0
    prev_time = 0
    starts = [0]
    names = cluster_names
    for i, v in enumerate(labels):
        if curr_time != 0:
            if prev != v:
                res += (
                    str(counter)
                    + "\n"
                    + num_to_min(prev_time)
                    + " --> "
                    + num_to_min(curr_time)
                    + f"\nSpeaker {prev}\n\n"
                )
                prev_time = curr_time
                counter += 1
                starts.append(prev_time)
        prev = v
        insert_row(index, embedings[i], user, prev)
        curr_time += 3
    starts.append(len(labels) * 3)
    res += (
        str(counter)
        + "\n"
        + num_to_min(prev_time)
        + " --> "
        + num_to_min(curr_time)
        + f"\nSpeaker {prev}\n\n"
    )
    return res, starts
