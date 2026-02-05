import json
from itertools import combinations
from Levenshtein import distance as lev_distance


def emotion_stats_from_segmentation(seg1: str, seg2: str):
    if not isinstance(seg1, str) or not isinstance(seg2, str):
        return None, None, None
    n = min(len(seg1), len(seg2))
    if n == 0:
        return None, None, None
    labels = sorted(set(c for c in (seg1[:n] + seg2[:n]) if c.isdigit()))
    if not labels:
        return None, None, None
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    counts = [0] * k
    matrix = [[0] * k for _ in range(k)]
    for c1, c2 in zip(seg1[:n], seg2[:n]):
        if not (c1.isdigit() and c2.isdigit()):
            continue
        if c1 == c2:
            continue
        i = label_to_idx[c1]
        j = label_to_idx[c2]
        counts[i] += 1
        counts[j] += 1
        matrix[i][j] += 1
        matrix[j][i] += 1
    counts = [c / n for c in counts]
    matrix = [[val / n for val in row] for row in matrix]
    emo_error = {lab: counts[label_to_idx[lab]] for lab in labels}
    return labels, emo_error, matrix
def iter_norm_dists(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "$values" in data:
        dialogs = data["$values"]
    elif isinstance(data, list):
        dialogs = data
    else:
        dialogs = [data]
    three = 7
    for dlg_idx, dlg in enumerate(dialogs, start=1):
        three -= 1
        if three == 0:
            break
        markup = dlg.get("Markup")
        if not markup:
            continue
        if isinstance(markup, dict) and "$values" in markup:
            markup_list = markup["$values"]
        elif isinstance(markup, list):
            markup_list = markup
        else:
            continue
        annots = []
        for m in markup_list:
            seg_str = m.get("Segmentation")
            mid = m.get("Id")
            if isinstance(seg_str, str) and mid is not None:
                annots.append((mid, seg_str))
        if len(annots) < 2:
            continue
        for (id1, s1), (id2, s2) in combinations(annots, 2):
            dist = lev_distance(s1, s2)
            norm = dist / max(len(s1), len(s2))
            labels, emo_error, emo_matrix = emotion_stats_from_segmentation(s1, s2)
            yield dlg_idx, id1, id2, norm, labels, emo_error, emo_matrix


if __name__ == "__main__":
    json_path = "output_markup.json"
    for dlg_idx, id1, id2, norm, labels, emo_error, emo_matrix in iter_norm_dists(json_path):
        print(f"Dialog {dlg_idx}, annotators {id1} vs {id2}")
        print(f"  Segmentation Levenshtein (normalized): {norm:.6f}")
        if not labels:
            print("  No digit labels in Segmentation.\n")
            continue
        print("  Disagreement per label (from Segmentation):")
        print("    " + "  ".join(f"{lab}={emo_error[lab]:.4f}" for lab in labels))
        print("  Confusion matrix of disagreements (normalized) over Segmentation:")
        header = "       " + " ".join(f"{lab:>8}" for lab in labels)
        print(header)
        for lab_row, row in zip(labels, emo_matrix):
            row_str = " ".join(f"{val:8.4f}" for val in row)
            print(f"    {lab_row:<3} {row_str}")
        print()
