import numpy as np
from collections import defaultdict
from typing import List, Dict, Union
import pickle
import os


class StepHashDict:
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        correct_cluster_threshold: float = 0.5,
        rep_mode: str = "all",  # "first" | "centroid" | "medoid" | "all"
    ):
        self.dicts: Dict[int, Dict[int, dict]] = defaultdict(dict)
        self.resp_len_stats: Dict[int, dict] = defaultdict(lambda: {"min_len": float("inf"), "mean_len": 0.0, "cnt": 0})
        self.similarity_threshold = similarity_threshold
        self.correct_cluster_threshold = correct_cluster_threshold
        self.rep_mode = rep_mode.lower()
        assert self.rep_mode in {"first", "centroid", "medoid", "all"}
        # first    : representative vector is fixed to the first member
        # centroid : representative vector is the mean
        # medoid   : representative vector is the member closest to the mean
        # all      : merge only if *average* similarity exceeds threshold; rep stays as first member

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-8)

    def _build_rep_matrix(self, clusters: Dict[int, dict]) -> np.ndarray:
        """Stack all rep_embedding vectors into a (K, D) matrix; returns None if empty."""
        if not clusters:
            return None
        reps = [info["rep_embedding"] for info in clusters.values()]
        reps = np.vstack(reps).copy()
        reps.setflags(write=True)
        return reps

    def update_sample_step_hash_dict(
        self,
        sample_id: int,
        embeddings: np.ndarray,   # (N, D), L2-normalised
        texts: List[str],
        lead_correct_list: List[bool] | None = None
    ):
        assert len(embeddings) == len(texts), "embeddings and texts length mismatch"

        clusters = self.dicts[sample_id]
        rep_matrix = self._build_rep_matrix(clusters)
        correctness = []

        for idx, (emb, txt) in enumerate(zip(embeddings, texts)):
            lead_to_correct = lead_correct_list[idx] if lead_correct_list else None

            if rep_matrix is None:
                clusters[0] = dict(
                    rep_embedding=emb,
                    rep_text=txt,
                    members_texts=[txt],
                    members_idx=[idx],
                    member_embeddings=[emb],
                    correct_cnt=1 if lead_to_correct else 0
                )
                rep_matrix = emb[None, :].copy()
                rep_matrix.setflags(write=True)
                correctness.append(True if lead_to_correct else False)
                continue

            # Find the best matching cluster
            insert_cid = None
            if self.rep_mode == "all":
                sims_rep = rep_matrix @ emb
                cand_cids = np.where(sims_rep > self.similarity_threshold)[0]

                if cand_cids.size:
                    best_avg, insert_cid = -1.0, None
                    for cid in cand_cids:
                        cinfo = clusters[cid]
                        member_embs = cinfo["member_embeddings"]
                        sims = member_embs @ emb

                        if np.all(sims > self.similarity_threshold):
                            avg_sim = sims.mean()
                            if avg_sim > best_avg:
                                insert_cid, best_avg = cid, avg_sim
            else:
                sims = np.dot(rep_matrix, emb)
                best_row = int(np.argmax(sims))
                if float(sims[best_row]) > self.similarity_threshold:
                    insert_cid = best_row

            # Insert into existing cluster or create a new one
            if insert_cid is not None:
                cinfo = clusters[insert_cid]
                cinfo["members_texts"].append(txt)
                cinfo["members_idx"].append(idx)
                cinfo["correct_cnt"] += 1 if lead_to_correct else 0
                correctness.append(True if cinfo["correct_cnt"] / len(cinfo["members_texts"]) > self.correct_cluster_threshold else False)
                cinfo["member_embeddings"] = np.concatenate(
                    (cinfo["member_embeddings"], emb[None, :]), axis=0
                )

                if self.rep_mode == "centroid":
                    new_rep = self._normalize(np.mean(cinfo["member_embeddings"], 0))
                    cinfo["rep_embedding"] = new_rep
                    rep_matrix[insert_cid] = new_rep
                elif self.rep_mode == "medoid":
                    centroid = np.mean(cinfo["member_embeddings"], 0)
                    sims_centroid = np.dot(cinfo["member_embeddings"], centroid)
                    best_idx = int(np.argmax(sims_centroid))
                    new_rep = cinfo["member_embeddings"][best_idx]
                    cinfo["rep_embedding"] = new_rep
                    cinfo["rep_text"] = cinfo["members_texts"][best_idx]
                    rep_matrix[insert_cid] = new_rep
            else:
                new_cid = len(clusters)
                clusters[new_cid] = dict(
                    rep_embedding=emb,
                    rep_text=txt,
                    members_texts=[txt],
                    members_idx=[idx],
                    member_embeddings=emb[None, :].copy(),
                    correct_cnt=1 if lead_to_correct else 0
                )
                rep_matrix = np.vstack([rep_matrix, emb[None, :]]).copy()
                rep_matrix.setflags(write=True)
                correctness.append(True if lead_to_correct else False)
        return correctness

    def update_min_mean_correct_resp_len(self, sample_id: int, resp_len: int):
        self.resp_len_stats[sample_id]["min_len"] = min(
            self.resp_len_stats[sample_id]["min_len"], resp_len
        )
        self.resp_len_stats[sample_id]["mean_len"] = (
            self.resp_len_stats[sample_id]["mean_len"] * self.resp_len_stats[sample_id]["cnt"] + resp_len
        ) / (self.resp_len_stats[sample_id]["cnt"] + 1)
        self.resp_len_stats[sample_id]["cnt"] += 1

    def look_up_min_mean_correct_resp_len(self, sample_id: int) -> int:
        return self.resp_len_stats.get(sample_id, {"min_len": float("inf"), "mean_len": 0.0})["min_len"], \
               self.resp_len_stats.get(sample_id, {"min_len": float("inf"), "mean_len": 0.0})["mean_len"]

    def look_up_step_correctness(
        self,
        sample_id: int,
        texts: Union[str, List[str]]
    ) -> List[bool]:
        """
        Look up correctness by exact string match against members_texts.

        For each query string, iterate over all clusters for the given sample.
        If the string is found in a cluster's members_texts, return that
        cluster's correctness based on correct_cluster_threshold.
        Raises ValueError if not found.
        """
        if isinstance(texts, str):
            texts = [texts]

        clusters = self.dicts.get(sample_id, {})
        if not clusters:
            raise KeyError(f"No clusters found for sample_id {sample_id}")

        results: List[bool] = []

        for query in texts:
            found = False
            for cinfo in clusters.values():
                if query in cinfo["members_texts"]:
                    results.append(True if cinfo['correct_cnt'] / len(cinfo["members_texts"]) > self.correct_cluster_threshold else False)
                    found = True
                    break

            if not found:
                raise ValueError(
                    f'Text "{query}" not found in any cluster for sample_id {sample_id}'
                )

        return results

    def get_step_dict_info(self, verbose_info: bool = False, print_info: bool = False):
        """Return statistics about the current dictionary."""
        info_dict = defaultdict(dict)
        if print_info:
            print(f"Total samples: {len(self.dicts)}")
        for sample_id, clusters in self.dicts.items():
            avg_member_len = np.mean([len(cinfo["members_texts"]) for cinfo in clusters.values()])
            if print_info:
                print(f"Sample ID: {sample_id}, Clusters: {len(clusters)}, Avg Members: {avg_member_len:.2f}")
            info_dict[sample_id]["overall_info"] = {
                "clusters_cnt": len(clusters),
                "avg_member_len": avg_member_len
            }
            if verbose_info:
                info_dict[sample_id]["verbose_info"] = []
                for cid, cinfo in clusters.items():
                    if print_info:
                        print(f"  Cluster ID: {cid}, Rep text: {cinfo['rep_text'][:80]}, Members: {len(cinfo['members_texts'])}, Acc: {cinfo['correct_cnt'] / len(cinfo['members_texts']) if cinfo['members_texts'] else 0}")
                    info_dict[sample_id]["verbose_info"].append(
                        {
                            "cluster_id": cid,
                            "rep_text": cinfo["rep_text"][:80],
                            "members_count": len(cinfo["members_texts"]),
                            "sampled_member_texts": cinfo["members_texts"],
                            "lead_to_correct": cinfo["correct_cnt"],
                            "accuracy": cinfo["correct_cnt"] / len(cinfo["members_texts"]) if cinfo["members_texts"] else 0,
                        }
                    )

        return info_dict

    def save_info(self, filepath: str, overwrite: bool = True) -> None:
        """Serialize the current dicts to *filepath* directory."""
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(f"{filepath} already exists. Set overwrite=True to overwrite.")
        dicts_to_dump = dict(self.dicts)
        resp_len_stats_to_dump = dict(self.resp_len_stats)
        with open(os.path.join(filepath, 'step_hash_dict.pkl'), "wb") as f:
            pickle.dump(dicts_to_dump, f)
        with open(os.path.join(filepath, 'resp_len_stats.pkl'), "wb") as f:
            pickle.dump(resp_len_stats_to_dump, f)
        print(f"[StepHashDict] and [RespLenStats] saved to folder {filepath}")

    def load_info(self, filepath: str) -> None:
        """Load dicts from *filepath* directory, replacing the current state."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        with open(os.path.join(filepath, 'step_hash_dict.pkl'), "rb") as f:
            dicts_loaded = pickle.load(f)
        with open(os.path.join(filepath, 'resp_len_stats.pkl'), "rb") as f:
            resp_len_stats_loaded = pickle.load(f)
        self.dicts = defaultdict(dict, dicts_loaded)
        self.resp_len_stats = defaultdict(lambda: {"min_len": float("inf"), "mean_len": 0.0, "cnt": 0}, resp_len_stats_loaded)
        print(f"[StepHashDict] and [RespLenStats] loaded dicts from folder {filepath}")


class SampleHashDict:
    """A lightweight per-sample info store.

    self.dicts[sample_id] holds:
        - 'corret_answered': bool  -- whether this sample has been answered correctly (sticky True)
        - 'min_len': float         -- shortest observed response length for this sample
    """

    def __init__(self):
        self.dicts: Dict[int, dict] = defaultdict(lambda: {"corret_answered": False, "min_len": float("inf")})
        self.resp_len_stats: Dict[int, dict] = defaultdict(lambda: {"min_len": float("inf"), "mean_len": 0.0, "cnt": 0})

    def set_correct_answered(self, sample_id: int, value: bool) -> None:
        info = self.dicts[sample_id]
        info["corret_answered"] = bool(info.get("corret_answered", False) or value)

    def get_info(self, sample_id: int) -> dict:
        info = self.dicts[sample_id]
        return dict(info)

    def update_min_mean_correct_resp_len(self, sample_id: int, resp_len: int):
        stats = self.resp_len_stats[sample_id]
        stats["min_len"] = min(stats["min_len"], resp_len)
        stats["mean_len"] = (stats["mean_len"] * stats["cnt"] + resp_len) / (stats["cnt"] + 1)
        stats["cnt"] += 1
        info = self.dicts[sample_id]
        info["min_len"] = min(info.get("min_len", float("inf")), resp_len)
        return None

    def look_up_min_mean_correct_resp_len(self, sample_id: int) -> int:
        stats = self.resp_len_stats.get(sample_id, {"min_len": float("inf"), "mean_len": 0.0})
        return stats["min_len"], stats["mean_len"]

    def save_info(self, filepath: str, overwrite: bool = True) -> None:
        """Serialize to *filepath* directory (two pickle files)."""
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(f"{filepath} already exists. Set overwrite=True to overwrite.")
        dicts_to_dump = dict(self.dicts)
        resp_len_stats_to_dump = dict(self.resp_len_stats)
        with open(os.path.join(filepath, 'sample_hash_dict.pkl'), 'wb') as f:
            pickle.dump(dicts_to_dump, f)
        with open(os.path.join(filepath, 'sample_resp_len_stats.pkl'), 'wb') as f:
            pickle.dump(resp_len_stats_to_dump, f)
        print(f"[SampleHashDict] and [SampleRespLenStats] saved to folder {filepath}")

    def load_info(self, filepath: str) -> None:
        """Load from *filepath* directory, replacing the current state."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        with open(os.path.join(filepath, 'sample_hash_dict.pkl'), 'rb') as f:
            dicts_loaded = pickle.load(f)
        with open(os.path.join(filepath, 'sample_resp_len_stats.pkl'), 'rb') as f:
            resp_len_stats_loaded = pickle.load(f)
        self.dicts = defaultdict(lambda: {"corret_answered": False, "min_len": float("inf")}, dicts_loaded)
        self.resp_len_stats = defaultdict(lambda: {"min_len": float("inf"), "mean_len": 0.0, "cnt": 0}, resp_len_stats_loaded)
        print(f"[SampleHashDict] and [SampleRespLenStats] loaded dicts from folder {filepath}")
