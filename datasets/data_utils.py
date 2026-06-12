import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd


DATASET_ALIASES = {
    "PEMS03": ["PEMS03", "PeMS03", "pems03"],
    "PEMS04": ["PEMS04", "PeMS04", "pems04"],
    "PEMS07": ["PEMS07", "PeMS07", "pems07"],
    "PEMS08": ["PEMS08", "PeMS08", "pems08"],
    "METR-LA": ["METR-LA", "metr-la", "METRLA", "metr_la"],
    "PEMS-BAY": ["PEMS-BAY", "PeMS-BAY", "pems-bay", "PEMSBAY", "pems_bay"],
}


def canonical_name(name: str) -> str:
    u = name.upper().replace("_", "-")
    if u in {"PEMS03", "PEMS04", "PEMS07", "PEMS08"}:
        return u
    if u in {"METR-LA", "METRLA"}:
        return "METR-LA"
    if u in {"PEMS-BAY", "PEMSBAY"}:
        return "PEMS-BAY"
    return name


def find_dataset_dir(root_path: str, dataset_name: str) -> str:
    canonical = canonical_name(dataset_name)
    candidates = DATASET_ALIASES.get(canonical, [dataset_name])
    for cand in candidates:
        path = os.path.join(root_path, cand)
        if os.path.isdir(path):
            return path
    if os.path.isdir(os.path.join(root_path, dataset_name)):
        return os.path.join(root_path, dataset_name)
    raise FileNotFoundError(f"Cannot find dataset folder for {dataset_name} under {root_path}")


def _find_file(folder: str, extensions, keywords=None):
    keywords = keywords or []
    files = []
    for fn in os.listdir(folder):
        low = fn.lower()
        if any(low.endswith(ext) for ext in extensions):
            score = 0
            for kw in keywords:
                if kw.lower() in low:
                    score += 1
            files.append((score, fn))
    if not files:
        return None
    files.sort(key=lambda x: (-x[0], x[1]))
    return os.path.join(folder, files[0][1])


def load_npz_data(path: str) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)
    if "data" in obj:
        data = obj["data"]
    elif "x" in obj:
        data = obj["x"]
    else:
        data = obj[obj.files[0]]
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[..., None]
    if data.ndim == 3:
        return data
    if data.ndim == 4:
        return data.squeeze(-1)
    raise ValueError(f"Unsupported npz data shape {data.shape} from {path}")


def load_h5_data(path: str) -> np.ndarray:
    """Load METR-LA / PEMS-BAY h5 safely.

    These files are usually pandas HDFStore objects whose real data live in
    /df/block0_values.  Without pytables, reading the first HDF5 key may return
    axis labels such as 'axis0', causing the string-to-float error.  This reader
    first tries pandas, then uses h5py to explicitly search numeric datasets.
    """
    try:
        df = pd.read_hdf(path)
        arr = df.values.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr
    except Exception as e:
        print(f"[WARN] pandas.read_hdf failed for {path}: {e}; try h5py fallback.")

    try:
        import h5py
    except ImportError as e:
        raise ImportError("Please install pytables or h5py to read .h5 traffic files.") from e

    with h5py.File(path, "r") as f:
        if "df" in f:
            g = f["df"]
            for key in ["block0_values", "values"]:
                if key in g:
                    arr = np.asarray(g[key], dtype=np.float32)
                    if arr.ndim == 2:
                        arr = arr[..., None]
                    return arr
            value_keys = [k for k in g.keys() if k.startswith("block") and k.endswith("_values")]
            blocks = []
            for k in sorted(value_keys):
                a = np.asarray(g[k])
                if np.issubdtype(a.dtype, np.number) and a.ndim >= 2:
                    blocks.append(a.astype(np.float32))
            if len(blocks) == 1:
                arr = blocks[0]
                return arr[..., None] if arr.ndim == 2 else arr
            if len(blocks) > 1:
                arr = np.concatenate(blocks, axis=1).astype(np.float32)
                return arr[..., None] if arr.ndim == 2 else arr

        candidates = []

        def visit_func(name, obj):
            import h5py as _h5py
            if isinstance(obj, _h5py.Dataset):
                try:
                    a = np.asarray(obj)
                    if np.issubdtype(a.dtype, np.number) and a.ndim >= 2:
                        candidates.append((name, a))
                except Exception:
                    pass

        f.visititems(visit_func)
        if not candidates:
            keys = []
            f.visit(lambda name: keys.append(name))
            raise ValueError(f"Could not find numeric traffic matrix in {path}; keys={keys[:50]}")
        name, arr = max(candidates, key=lambda x: x[1].size)
        print(f"[INFO] h5py fallback loaded '{name}' from {path}, shape={arr.shape}")
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr


def load_traffic_data(root_path: str, dataset_name: str) -> Tuple[np.ndarray, str, str]:
    folder = find_dataset_dir(root_path, dataset_name)
    canonical = canonical_name(dataset_name)
    if canonical in {"METR-LA", "PEMS-BAY"}:
        h5 = _find_file(folder, [".h5"], ["metr", "pems", "bay", "la"])
        if h5 is not None:
            return load_h5_data(h5), folder, h5
    npz = _find_file(folder, [".npz"], [canonical.replace("-", ""), canonical, "pems"])
    if npz is not None:
        return load_npz_data(npz), folder, npz
    h5 = _find_file(folder, [".h5"], ["metr", "pems", "bay", "la"])
    if h5 is not None:
        return load_h5_data(h5), folder, h5
    raise FileNotFoundError(f"No npz/h5 data file found in {folder}")


def _normalize_adj(adj: np.ndarray) -> np.ndarray:
    adj = np.nan_to_num(adj.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    adj[adj < 0] = 0
    if adj.max() > 0:
        adj = adj / adj.max()
    np.fill_diagonal(adj, 1.0)
    return adj


def _adj_from_distance_df(df: pd.DataFrame, num_nodes: int) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    numeric_matrix = df.apply(pd.to_numeric, errors="coerce")
    if numeric_matrix.shape[0] == num_nodes and numeric_matrix.shape[1] == num_nodes:
        mat = numeric_matrix.values.astype(np.float32)
        if np.isfinite(mat).any():
            return _normalize_adj(np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0))

    if df.shape[1] >= 3:
        numeric = df.iloc[:, :3].apply(pd.to_numeric, errors="coerce").dropna()
        if len(numeric) == 0:
            return np.eye(num_nodes, dtype=np.float32)
        arr = numeric.values
        src = arr[:, 0].astype(np.int64)
        dst = arr[:, 1].astype(np.int64)
        dist = arr[:, 2].astype(np.float32)

        observed = np.unique(np.concatenate([src, dst]))
        if src.min() >= 1 and dst.min() >= 1 and src.max() <= num_nodes and dst.max() <= num_nodes:
            src = src - 1
            dst = dst - 1
        elif src.max() >= num_nodes or dst.max() >= num_nodes or src.min() < 0 or dst.min() < 0:
            id_to_idx = {int(v): i for i, v in enumerate(observed[:num_nodes])}
            keep = np.array([(int(i) in id_to_idx and int(j) in id_to_idx) for i, j in zip(src, dst)])
            src = np.array([id_to_idx[int(i)] for i in src[keep]], dtype=np.int64)
            dst = np.array([id_to_idx[int(j)] for j in dst[keep]], dtype=np.int64)
            dist = dist[keep]

        finite = np.isfinite(dist)
        if finite.any():
            sigma = np.std(dist[finite])
            sigma = sigma if sigma > 1e-6 else np.mean(dist[finite]) + 1e-6
            weight = np.exp(-np.square(dist / sigma)).astype(np.float32)
        else:
            weight = np.ones_like(dist, dtype=np.float32)

        for i, j, w in zip(src, dst, weight):
            if 0 <= i < num_nodes and 0 <= j < num_nodes:
                adj[i, j] = max(adj[i, j], float(w))
                adj[j, i] = max(adj[j, i], float(w))
    else:
        adj = np.eye(num_nodes, dtype=np.float32)
    return _normalize_adj(adj)


def _try_to_numeric_square_matrix(obj, num_nodes: int):
    """Return a numeric adjacency-like matrix from one pickle field, or None.

    Standard METR-LA / PeMS-BAY DCRNN pickle files are often stored as
    [sensor_ids, sensor_id_to_ind, adj_mx] rather than a tuple.  The old
    loader only handled tuple, so it tried to cast the whole heterogeneous
    list to float and triggered:
        setting an array element with a sequence ... shape was (3,) ...
    This helper safely scans candidates and only accepts true 2-D matrices.
    """
    if obj is None:
        return None

    # scipy sparse matrix
    if hasattr(obj, "toarray"):
        try:
            obj = obj.toarray()
        except Exception:
            return None

    try:
        arr = np.asarray(obj)
    except Exception:
        return None

    if arr.ndim != 2:
        return None
    if arr.shape[0] != arr.shape[1]:
        return None
    if arr.shape[0] != num_nodes:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(np.float32)
        except Exception:
            return None
    return arr.astype(np.float32)


def _extract_adj_from_pickle(obj, num_nodes: int):
    """Robustly extract adjacency matrix from common traffic pickle formats."""
    # Direct matrix / scipy sparse.
    mat = _try_to_numeric_square_matrix(obj, num_nodes)
    if mat is not None:
        return mat

    # DCRNN-style list/tuple: [sensor_ids, sensor_id_to_ind, adj_mx]
    # Some repositories save it as tuple, others as list.
    if isinstance(obj, (tuple, list)):
        for item in reversed(obj):
            mat = _try_to_numeric_square_matrix(item, num_nodes)
            if mat is not None:
                return mat
        # One more level of nesting, just in case.
        for item in reversed(obj):
            if isinstance(item, (tuple, list, dict)):
                mat = _extract_adj_from_pickle(item, num_nodes)
                if mat is not None:
                    return mat

    # Dict-style formats.
    if isinstance(obj, dict):
        preferred = [
            "adj_mx", "adj", "adjacency", "adj_mat", "A",
            "distance_mx", "dist_mx", "matrix",
        ]
        for key in preferred:
            if key in obj:
                mat = _try_to_numeric_square_matrix(obj[key], num_nodes)
                if mat is not None:
                    return mat
        for value in obj.values():
            mat = _try_to_numeric_square_matrix(value, num_nodes)
            if mat is not None:
                return mat
    return None


def load_adjacency(root_path: str, dataset_name: str, num_nodes: int) -> np.ndarray:
    folder = find_dataset_dir(root_path, dataset_name)
    canonical = canonical_name(dataset_name)
    pkl = _find_file(folder, [".pkl"], ["adj", "adj_mx", "adj_mat"])
    if pkl is not None:
        try:
            with open(pkl, "rb") as f:
                obj = pickle.load(f, encoding="latin1")
            adj = _extract_adj_from_pickle(obj, num_nodes)
            if adj is not None:
                print(f"[INFO] Loaded adjacency pkl {pkl}, shape={adj.shape}")
                return _normalize_adj(adj)
            print(f"[WARN] No {num_nodes}x{num_nodes} numeric matrix found in adjacency pkl {pkl}; try distance file fallback.")
        except Exception as e:
            print(f"[WARN] Failed loading adjacency pkl {pkl}: {e}; try distance file fallback.")

    xls = _find_file(folder, [".xls", ".xlsx", ".csv"], ["distance", "dist", canonical])
    if xls is not None:
        try:
            if xls.lower().endswith(".csv"):
                df = pd.read_csv(xls, header=None, low_memory=False)
            else:
                df = pd.read_excel(xls, header=None)
            print(f"[INFO] Build adjacency from distance file {xls}")
            return _adj_from_distance_df(df, num_nodes)
        except Exception as e:
            print(f"[WARN] Failed loading distance file {xls}: {e}")
    print(f"[WARN] Use identity adjacency for {dataset_name}. This may hurt performance.")
    return np.eye(num_nodes, dtype=np.float32)


def build_time_features(total_steps: int, points_per_hour: int = 12) -> np.ndarray:
    idx = np.arange(total_steps, dtype=np.float32)
    steps_per_day = 24 * points_per_hour
    minute_of_day = (idx % steps_per_day) / steps_per_day
    day_of_week = ((idx // steps_per_day) % 7) / 7.0
    feats = np.stack([
        np.sin(2 * np.pi * minute_of_day),
        np.cos(2 * np.pi * minute_of_day),
        np.sin(2 * np.pi * day_of_week),
        np.cos(2 * np.pi * day_of_week),
    ], axis=-1)
    return feats.astype(np.float32)
