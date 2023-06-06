import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from ..data import preprosess
from ..models import LineDolgNet
from .utils import bic_score, generate_subtitles


class Diarizer:
    def __init__(self, model_path, win_size, stride, imb_dim, device, trh):
        self.model = LineDolgNet(model_path, win_size, stride, imb_dim, device).to(
            device
        )
        self.trh = trh
        self.device = device
        pass

    def diarize(self, audio, user, index):
        with torch.inference_mode():
            print("Start preprosses audio")
            image = preprosess(audio).to(self.device)
            print("Start generating embedings")
            embedings = (
                self.model(torch.stack([image, image], 0))[0].T.detach().cpu().numpy()
            )
            print("Start clustering")
            clustering = self.clusterize(embedings)
            print("Start diarize_voices")
            similarity, used, last_used = self.diarize_voices(embedings, clustering)
            return generate_subtitles(last_used, user, embedings, index)

    def clusterize(self, all_embedings):
        norm = all_embedings
        n_clusters = 1
        best_score = -10000000000000
        min_num = min(len(all_embedings), 10)
        K = range(1, min_num)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(norm)
            score = bic_score(norm, kmeanModel.labels_)
            if best_score * 1 < score:
                best_score = score
                n_clusters = k

        # n_clusters = min(n_clusters - 2)
        print("n_clusters: ", n_clusters)
        clustering = KMeans(n_clusters=n_clusters).fit(norm)
        return clustering

    def diarize_voices(self, all_embedings, clustering):
        similarity = []
        used = [[i] for i in clustering.cluster_centers_]
        original_count = len(used) - 1
        last_used = []
        for i in range(len(all_embedings)):
            max_sim = -1
            voice = -1
            for used_voice in range(len(used)):
                current_sim = cosine_similarity([*used[used_voice], all_embedings[i]])[
                    -1
                ][:-1]
                current_sim = np.mean(current_sim)
                if current_sim > max_sim:
                    voice = used_voice
                    max_sim = current_sim

            if max_sim < self.trh:
                used.append([all_embedings[i]])
                last_used.append(len(used) - 1)
                similarity.append(-1)
            else:
                if used_voice > original_count:
                    used[used_voice].append(all_embedings[i])
                last_used.append(voice)
                similarity.append(max_sim)
        return similarity, used, last_used
