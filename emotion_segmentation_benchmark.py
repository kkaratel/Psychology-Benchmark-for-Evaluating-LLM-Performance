import os
import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import evaluate
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import glob
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz


class PremarkedEmotionBenchmark:
    def __init__(self):
        self.emotion_map = {
            "JOY": 1,
            "SADNESS": 2,
            "ANGER": 3,
            "DISGUST": 4,
            "FEAR": 5,
            "SURPRISE": 6,
            "CONTEMPT": 7
        }
        self.emotion_names = [
            "NEUTRAL",
            "JOY",
            "SADNESS",
            "ANGER",
            "DISGUST",
            "FEAR",
            "SURPRISE",
            "CONTEMPT"
        ]

        self.psych_question_names = [
            "эмпатичный ответ",
            "этичный ответ",
            "продуктивный вопрос",
            "полезная рекомендация"
        ]

        self.general_question_names = [
            "депрессивное состояние",
            "психологическая или психиатрическая помощь",
            "растерянность",
            "спутанность мышления",
            "риск суицида"
        ]

        try:
            self.accuracy_metric = evaluate.load("accuracy")
            self.precision_metric = evaluate.load("precision")
            self.recall_metric = evaluate.load("recall")
            self.f1_metric = evaluate.load("f1")
        except Exception as e:
            print(f"Error loading evaluate metrics: {str(e)}")
            self.accuracy_metric = None
            self.precision_metric = None
            self.recall_metric = None
            self.f1_metric = None

    def load_json_file(self, filepath: str) -> Optional[Dict]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading file {filepath}: {str(e)}")
            return None

    def load_predictions_from_dir(self, dirpath: str) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        files = glob.glob(os.path.join(dirpath, "*.json"))
        for fp in files:
            data = self.load_json_file(fp)
            if not data or not isinstance(data, dict):
                continue
            for link, info in data.items():
                if isinstance(info, dict):
                    result[link] = info
        return result

    def load_ground_truth_extended(self, filepath: str) -> Optional[Dict[str, Dict[str, Any]]]:
        try:
            data = self.load_json_file(filepath)
            if not data:
                return None

            gt: Dict[str, Dict[str, Any]] = {}
            if isinstance(data, dict) and "$values" in data:
                for item in data["$values"]:
                    link = item.get("Link")
                    if not link:
                        continue

                    seg = None
                    msg_mark = None
                    result = None
                    tags = None

                    markup = item.get("Markup", {})
                    markups = markup.get("$values", []) if isinstance(markup, dict) else []
                    for m in markups:
                        if not isinstance(m, dict):
                            continue
                        if seg is None and "Segmentation" in m:
                            seg = m.get("Segmentation")
                        if msg_mark is None and "MessageMark" in m:
                            msg_mark = m.get("MessageMark")
                        if result is None and "Result" in m:
                            result = m.get("Result")
                        if tags is None and "Tags" in m:
                            tags = m.get("Tags")

                    gt[link] = {
                        "Segmentation": seg,
                        "MessageMark": msg_mark,
                        "Result": result,
                        "Tags": tags
                    }
            else:
                print("Key $values not found in ground truth data")
                return None

            if len(gt) == 0:
                return None
            return gt

        except Exception as e:
            print(f"Error loading ground truth annotations: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _strip_think_blocks(self, text: str) -> str:
        while True:
            start = text.find("<think>")
            if start == -1:
                break
            end = text.find("</think>", start)
            if end == -1:
                text = text[:start]
                break
            text = text[:start] + text[end + len("</think>"):]
        return text

    def extract_client_text_and_indices(self, text: str) -> Tuple[str, List[int]]:
        client_chars: List[str] = []
        client_indices: List[int] = []
        i = 0
        n = len(text)
        speaker = None

        client_prefix = "Клиент:"
        psych_prefix = "Психолог:"
        while i < n:
            if text.startswith(client_prefix, i):
                speaker = "client"
                i += len(client_prefix)
                if i < n and text[i] == " ":
                    i += 1
                continue
            if text.startswith(psych_prefix, i):
                speaker = "psych"
                i += len(psych_prefix)
                if i < n and text[i] == " ":
                    i += 1
                continue
            ch = text[i]
            if speaker == "client":
                client_chars.append(ch)
                client_indices.append(i)
            i += 1
        client_text = "".join(client_chars)
        return client_text, client_indices

    def clean_model_markup(self, annotated_dialogue: str) -> Tuple[str, str]:
        text = self._strip_think_blocks(annotated_dialogue)
        pattern = r"(^|\n)\s*(Клиент|Психолог)\s*:\s*"
        text = re.sub(pattern, r"\1", text)
        cleaned_for_decode = text
        plain_client_text = self.remove_emotion_tags(cleaned_for_decode)
        return cleaned_for_decode, plain_client_text

    def normalize_for_comparison(self, text: str) -> str:
        text = text.replace("\r", " ").replace("\n", " ")
        text = text.lower()
        allowed = []
        for ch in text:
            if ch.isalnum() or ch.isspace():
                allowed.append(ch)
        s = "".join(allowed)
        s = "".join(s.split())
        return s

    def remove_emotion_tags(self, text: str) -> str:
        result_chars = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] == "[":
                i += 1
                while i < n and text[i] != "]":
                    i += 1
                if i < n and text[i] == "]":
                    i += 1
            else:
                result_chars.append(text[i])
                i += 1
        return "".join(result_chars)

    def validate_client_texts(self, original_client_text: str, model_plain_text: str) -> Tuple[bool, str]:
        orig_norm = self.normalize_for_comparison(original_client_text)
        model_norm = self.normalize_for_comparison(model_plain_text)
        if orig_norm == model_norm:
            return True, "Client texts match exactly"
        min_len = min(len(orig_norm), len(model_norm))
        for i in range(min_len):
            if orig_norm[i] != model_norm[i]:
                return False, (
                    f"Mismatch at position {i}: "
                    f"original='{orig_norm[i:i+30]}' , model='{model_norm[i:i+30]}'"
                )
        if len(orig_norm) > len(model_norm):
            return False, f"Original client text is longer by {len(orig_norm) - len(model_norm)} characters"
        return False, f"Model text is longer by {len(model_norm) - len(orig_norm)} characters"

    def align_client_texts(self, original_client_text: str, model_plain_text: str) -> Tuple[bool, List[int], str]:
        matcher = SequenceMatcher(None, original_client_text, model_plain_text, autojunk=False)
        mapping = [-1] * len(original_client_text)
        for match in matcher.get_matching_blocks():
            i, j, n = match
            for k in range(n):
                if i + k < len(original_client_text):
                    mapping[i + k] = j + k
        matched = sum(1 for x in mapping if x != -1)
        msg = f"Aligned {matched} out of {len(original_client_text)} characters"
        return True, mapping, msg

    def test_word_alignment(self, original_text: str, model_text: str) -> Dict[str, Any]:
        orig_words = original_text.split()
        model_words = model_text.split()
        results = {
            "total_orig_words": len(orig_words),
            "total_model_words": len(model_words),
            "matched_words": 0,
            "match_examples": [],
            "mismatch_examples": []
        }
        for orig_word in orig_words[:20]:
            best_score = 0
            best_model_word = ""
            for model_word in model_words:
                score = fuzz.ratio(orig_word, model_word)
                if score > best_score:
                    best_score = score
                    best_model_word = model_word
            if best_score > 80:
                results["matched_words"] += 1
                if len(results["match_examples"]) < 3:
                    results["match_examples"].append(f"'{orig_word}' -> '{best_model_word}' ({best_score}%)")
            else:
                if len(results["mismatch_examples"]) < 3:
                    results["mismatch_examples"].append(f"'{orig_word}' -> not found (best match score: {best_score}%)")
        return results

    def decode_emotions_to_sequence(self, text_with_tags: str) -> List[int]:
        result: List[int] = []
        i = 0
        n = len(text_with_tags)
        current_emotion = 0
        while i < n:
            ch = text_with_tags[i]
            if ch == "\n":
                result.append(current_emotion)
                i += 1
                continue
            if ch == "[":
                if i + 1 < n and text_with_tags[i + 1] == "/":
                    end = text_with_tags.find("]", i)
                    if end != -1:
                        current_emotion = 0
                        i = end + 1
                    else:
                        i += 1
                else:
                    end = text_with_tags.find("]", i)
                    if end != -1:
                        emotion = text_with_tags[i + 1:end]
                        current_emotion = self.emotion_map.get(emotion, 0)
                        i = end + 1
                    else:
                        i += 1
            else:
                result.append(current_emotion)
                i += 1
        return result

    def _list_to_labeled_dict(self, values: Any, labels: List[str]) -> Any:
        if not isinstance(values, list):
            return values
        out = {}
        m = min(len(values), len(labels))
        for i in range(m):
            out[labels[i]] = values[i]
        if len(values) > len(labels):
            for i in range(len(labels), len(values)):
                out[str(i)] = values[i]
        return out

    def calculate_base_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        unique_true = set(y_true)
        unique_pred = set(y_pred)
        all_labels = list(range(8))
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        accuracy_per_class = []
        for i in range(len(all_labels)):
            row_sum = cm[i].sum()
            if row_sum > 0:
                accuracy_per_class.append(cm[i, i] / row_sum)
            else:
                accuracy_per_class.append(0.0)
        try:
            if self.accuracy_metric and self.precision_metric and self.recall_metric and self.f1_metric:
                accuracy = self.accuracy_metric.compute(predictions=y_pred, references=y_true)
                if len(unique_true) > 1 or len(unique_pred) > 1:
                    precision = self.precision_metric.compute(
                        predictions=y_pred, references=y_true,
                        average="weighted", zero_division=0
                    )
                    recall = self.recall_metric.compute(
                        predictions=y_pred, references=y_true,
                        average="weighted", zero_division=0
                    )
                    f1 = self.f1_metric.compute(
                        predictions=y_pred, references=y_true,
                        average="weighted", zero_division=0
                    )
                    f1_macro = self.f1_metric.compute(
                        predictions=y_pred, references=y_true,
                        average="macro", zero_division=0
                    )
                    f1_micro = self.f1_metric.compute(
                        predictions=y_pred, references=y_true,
                        average="micro", zero_division=0
                    )

                    precision_per_class = self.precision_metric.compute(
                        predictions=y_pred, references=y_true,
                        average=None, zero_division=0
                    )
                    recall_per_class = self.recall_metric.compute(
                        predictions=y_pred, references=y_true,
                        average=None, zero_division=0
                    )
                    f1_per_class = self.f1_metric.compute(
                        predictions=y_pred, references=y_true,
                        average=None, zero_division=0
                    )
                else:
                    correct_rate = 1.0 if y_true == y_pred else 0.0
                    precision = {"precision": correct_rate}
                    recall = {"recall": correct_rate}
                    f1 = {"f1": correct_rate}
                    f1_macro = {"f1": correct_rate}
                    f1_micro = {"f1": correct_rate}
                    precision_per_class = {"precision": [correct_rate]}
                    recall_per_class = {"recall": [correct_rate]}
                    f1_per_class = {"f1": [correct_rate]}
            else:
                accuracy = {"accuracy": accuracy_score(y_true, y_pred)}
                if len(unique_true) > 1 or len(unique_pred) > 1:
                    precision = {"precision": precision_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )}
                    recall = {"recall": recall_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )}
                    f1 = {"f1": f1_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )}
                    f1_macro = {"f1": f1_score(
                        y_true, y_pred, average="macro", zero_division=0
                    )}
                    f1_micro = {"f1": f1_score(
                        y_true, y_pred, average="micro", zero_division=0
                    )}
                    precision_per_class = {
                        "precision": precision_score(
                            y_true, y_pred, average=None, zero_division=0
                        ).tolist()
                    }
                    recall_per_class = {
                        "recall": recall_score(
                            y_true, y_pred, average=None, zero_division=0
                        ).tolist()
                    }
                    f1_per_class = {
                        "f1": f1_score(
                            y_true, y_pred, average=None, zero_division=0
                        ).tolist()
                    }
                else:
                    correct_rate = accuracy["accuracy"]
                    precision = {"precision": correct_rate}
                    recall = {"recall": correct_rate}
                    f1 = {"f1": correct_rate}
                    f1_macro = {"f1": correct_rate}
                    f1_micro = {"f1": correct_rate}
                    precision_per_class = {"precision": [correct_rate]}
                    recall_per_class = {"recall": [correct_rate]}
                    f1_per_class = {"f1": [correct_rate]}

            precision_per_class_labeled = self._list_to_labeled_dict(precision_per_class["precision"], self.emotion_names)
            recall_per_class_labeled = self._list_to_labeled_dict(recall_per_class["recall"], self.emotion_names)
            f1_per_class_labeled = self._list_to_labeled_dict(f1_per_class["f1"], self.emotion_names)
            accuracy_per_class_labeled = self._list_to_labeled_dict(accuracy_per_class, self.emotion_names)

            return {
                "accuracy": float(accuracy["accuracy"]),
                "precision_weighted": float(precision["precision"]),
                "recall_weighted": float(recall["recall"]),
                "f1_weighted": float(f1["f1"]),
                "f1_macro": float(f1_macro["f1"]),
                "f1_micro": float(f1_micro["f1"]),
                "precision_per_class": precision_per_class_labeled,
                "recall_per_class": recall_per_class_labeled,
                "f1_per_class": f1_per_class_labeled,
                "accuracy_per_class": accuracy_per_class_labeled,
                "unique_true_labels": sorted(list(unique_true)),
                "unique_pred_labels": sorted(list(unique_pred))
            }

        except Exception as e:
            print(f"Error calculating base metrics: {str(e)}")
            correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            accuracy_manual = correct / len(y_true) if len(y_true) > 0 else 0.0
            return {
                "accuracy": float(accuracy_manual),
                "precision_weighted": float(accuracy_manual),
                "recall_weighted": float(accuracy_manual),
                "f1_weighted": float(accuracy_manual),
                "f1_macro": float(accuracy_manual),
                "f1_micro": float(accuracy_manual),
                "precision_per_class": self._list_to_labeled_dict([float(accuracy_manual)], self.emotion_names),
                "recall_per_class": self._list_to_labeled_dict([float(accuracy_manual)], self.emotion_names),
                "f1_per_class": self._list_to_labeled_dict([float(accuracy_manual)], self.emotion_names),
                "accuracy_per_class": self._list_to_labeled_dict([float(accuracy_manual)], self.emotion_names),
                "unique_true_labels": sorted(list(set(y_true))),
                "unique_pred_labels": sorted(list(set(y_pred))),
                "error_base": str(e)
            }

    def calculate_iou_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        num_classes = 8
        iou_per_class = []
        tp_per_class = []
        fp_per_class = []
        fn_per_class = []
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        for emotion in range(num_classes):
            true_mask = y_true_np == emotion
            pred_mask = y_pred_np == emotion
            tp = int(np.sum(true_mask & pred_mask))
            fp = int(np.sum(pred_mask & ~true_mask))
            fn = int(np.sum(true_mask & ~pred_mask))
            union = tp + fp + fn
            iou = tp / union if union > 0 else 0.0
            tp_per_class.append(tp)
            fp_per_class.append(fp)
            fn_per_class.append(fn)
            iou_per_class.append(float(iou))
        mean_iou = float(np.mean(iou_per_class))
        mean_iou_no_neutral = float(np.mean(iou_per_class[1:])) if len(iou_per_class) > 1 else 0.0
        total_chars = int(len(y_true))
        class_weights = [float(np.sum(y_true_np == e) / total_chars) if total_chars > 0 else 0.0 for e in range(num_classes)]
        weighted_iou = float(np.sum([iou * w for iou, w in zip(iou_per_class, class_weights)]))
        neutral_correct = tp_per_class[0]
        if total_chars > neutral_correct:
            non_neutral_total = total_chars - neutral_correct
            non_neutral_weights = [tp_per_class[e] / non_neutral_total for e in range(1, num_classes)]
            weighted_iou_no_neutral = float(np.sum([
                iou * w for iou, w in zip(iou_per_class[1:], non_neutral_weights)
            ])) if non_neutral_total > 0 else 0.0
        else:
            weighted_iou_no_neutral = 0.0
        return {
            "iou_per_class": self._list_to_labeled_dict(iou_per_class, self.emotion_names),
            "tp_per_class": self._list_to_labeled_dict(tp_per_class, self.emotion_names),
            "fp_per_class": self._list_to_labeled_dict(fp_per_class, self.emotion_names),
            "fn_per_class": self._list_to_labeled_dict(fn_per_class, self.emotion_names),
            "mean_iou": mean_iou,
            "mean_iou_no_neutral": mean_iou_no_neutral,
            "weighted_iou": weighted_iou,
            "weighted_iou_no_neutral": weighted_iou_no_neutral,
            "total_chars": total_chars,
            "neutral_chars": neutral_correct
        }

    def calculate_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        min_len = min(len(y_true), len(y_pred))
        y_true = [int(x) for x in y_true[:min_len]]
        y_pred = [int(x) for x in y_pred[:min_len]]
        base_metrics = self.calculate_base_metrics(y_true, y_pred)
        iou_metrics = self.calculate_iou_metrics(y_true, y_pred)
        return {**base_metrics, **iou_metrics, "total_samples": int(min_len)}

    def create_confusion_matrix(self, y_true: List[int], y_pred: List[int], save_path: str) -> List[List[int]]:
        min_len = min(len(y_true), len(y_pred))
        y_true = [int(x) for x in y_true[:min_len]]
        y_pred = [int(x) for x in y_pred[:min_len]]
        all_labels = list(range(8))
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.emotion_names,
            yticklabels=self.emotion_names,
            cbar_kws={"label": "Count"}
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        finally:
            plt.close()
        return cm.tolist()

    def parse_tags(self, s: Any) -> List[str]:
        if s is None:
            return []
        if not isinstance(s, str):
            s = str(s)
        s = re.sub(r"\s+", "", s)
        parts = [p for p in s.split(",") if p]
        return parts

    def parse_scale_answers(self, s: Any, expected_len: int) -> Optional[List[int]]:
        if s is None:
            return None
        if not isinstance(s, str):
            s = str(s)
        nums = re.findall(r"\d+", s)
        vals = []
        for x in nums:
            try:
                v = int(x)
                if 1 <= v <= 6:
                    vals.append(v)
            except Exception:
                continue
        if len(vals) != expected_len:
            return None
        return vals

    def parse_digits_1_to_6(self, s: Any) -> List[int]:
        if s is None:
            return []
        if not isinstance(s, str):
            s = str(s)
        digits = re.findall(r"[1-6]", s)
        return [int(d) for d in digits]

    def multilabel_metrics(self, true_sets: List[set], pred_sets: List[set]) -> Dict[str, Any]:
        tp = 0
        fp = 0
        fn = 0
        exact = 0
        jaccards = []
        per_tag = {}
        all_tags = set()
        for t in true_sets:
            all_tags |= t
        for p in pred_sets:
            all_tags |= p
        for tag in all_tags:
            per_tag[tag] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for tset, pset in zip(true_sets, pred_sets):
            if tset == pset:
                exact += 1
            inter = tset & pset
            tp += len(inter)
            fp += len(pset - tset)
            fn += len(tset - pset)
            union = len(tset | pset)
            jaccards.append(float(len(inter) / union) if union > 0 else 1.0)
            for tag in tset:
                per_tag[tag]["support"] += 1
            for tag in inter:
                per_tag[tag]["tp"] += 1
            for tag in (pset - tset):
                per_tag[tag]["fp"] += 1
            for tag in (tset - pset):
                per_tag[tag]["fn"] += 1

        micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
        mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0
        exact_match_rate = exact / len(true_sets) if true_sets else 0.0
        per_tag_scores = []
        for tag, c in per_tag.items():
            ttp, tfp, tfn = c["tp"], c["fp"], c["fn"]
            p = ttp / (ttp + tfp) if (ttp + tfp) > 0 else 0.0
            r = ttp / (ttp + tfn) if (ttp + tfn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            per_tag_scores.append({
                "tag": tag,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "support": int(c["support"])
            })
        per_tag_scores.sort(key=lambda x: (-x["support"], -x["f1"], x["tag"]))

        return {
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1),
            "exact_match_rate": float(exact_match_rate),
            "mean_jaccard": float(mean_jaccard),
            "num_samples": int(len(true_sets)),
            "per_tag_top": per_tag_scores[:30]
        }

    def ordinal_metrics(self, true_vals: List[int], pred_vals: List[int]) -> Dict[str, Any]:
        true_np = np.array(true_vals, dtype=np.float32)
        pred_np = np.array(pred_vals, dtype=np.float32)
        if len(true_np) == 0:
            return {
                "accuracy": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
                "within_1": 0.0,
                "num": 0
            }
        acc = float(np.mean(true_np == pred_np))
        mae = float(np.mean(np.abs(true_np - pred_np)))
        rmse = float(np.sqrt(np.mean((true_np - pred_np) ** 2)))
        within_1 = float(np.mean(np.abs(true_np - pred_np) <= 1.0))
        return {
            "accuracy": acc,
            "mae": mae,
            "rmse": rmse,
            "within_1": within_1,
            "num": int(len(true_np))
        }

    def run_full_benchmark(
        self,
        segmentation_dir: str,
        tags_dir: str,
        questions_dir: str,
        psych_dir: str,
        ground_truth_file: str,
        output_dir: str
    ) -> Dict[str, Any]:

        os.makedirs(output_dir, exist_ok=True)
        gt = self.load_ground_truth_extended(ground_truth_file)
        if not gt:
            raise ValueError("Failed to load ground truth annotations")
        pred_seg = self.load_predictions_from_dir(segmentation_dir)
        pred_tags = self.load_predictions_from_dir(tags_dir)
        pred_questions = self.load_predictions_from_dir(questions_dir)
        pred_psych = self.load_predictions_from_dir(psych_dir)
        all_predictions: List[int] = []
        all_ground_truth: List[int] = []
        seg_processed = 0
        seg_skipped = 0
        for dialogue_link, dialogue_info in pred_seg.items():
            original_text = (dialogue_info.get("text", "") or "")
            annotated_dialogue = (dialogue_info.get("annotated_dialogue", "") or "")
            title = dialogue_info.get("title", "")
            if dialogue_link not in gt or not gt[dialogue_link].get("Segmentation"):
                seg_skipped += 1
                continue
            if not annotated_dialogue.strip():
                seg_skipped += 1
                continue

            original_client_text, client_indices = self.extract_client_text_and_indices(original_text)
            cleaned_for_decode, model_plain_client_text = self.clean_model_markup(annotated_dialogue)
            ok_align, mapping, _ = self.align_client_texts(original_client_text, model_plain_client_text)
            if len(original_client_text) > 100 and len(model_plain_client_text) > 100:
                word_alignment_stats = self.test_word_alignment(original_client_text, model_plain_client_text)
                match_rate = word_alignment_stats["matched_words"] / min(20, max(1, word_alignment_stats["total_orig_words"]))
                if match_rate < 0.7:
                    print(f"WORD ALIGNMENT ISSUE: {dialogue_link} | {title} | only {match_rate:.1%} matched")

            if not ok_align:
                seg_skipped += 1
                continue
            full_segmentation = gt[dialogue_link]["Segmentation"]
            client_segmentation_chars = [
                full_segmentation[idx]
                for idx in client_indices
                if idx < len(full_segmentation)
            ]
            true_sequence = [int(ch) for ch in client_segmentation_chars]
            if len(true_sequence) == 0:
                seg_skipped += 1
                continue
            pred_sequence_raw = self.decode_emotions_to_sequence(cleaned_for_decode)
            if len(pred_sequence_raw) != len(model_plain_client_text):
                seg_skipped += 1
                continue

            effective_len = min(len(true_sequence), len(original_client_text))
            aligned_true = []
            aligned_pred = []
            for idx in range(effective_len):
                aligned_true.append(int(true_sequence[idx]))
                model_pos = mapping[idx]
                if model_pos == -1:
                    aligned_pred.append(0)
                else:
                    aligned_pred.append(int(pred_sequence_raw[model_pos]) if model_pos < len(pred_sequence_raw) else 0)

            all_predictions.extend(aligned_pred)
            all_ground_truth.extend(aligned_true)
            seg_processed += 1

        segmentation_metrics = None
        if all_predictions and all_ground_truth:
            segmentation_metrics = self.calculate_metrics(all_ground_truth, all_predictions)
            cm_list = self.create_confusion_matrix(
                all_ground_truth,
                all_predictions,
                os.path.join(output_dir, "confusion_matrix.png")
            )
            segmentation_metrics["confusion_matrix"] = cm_list
            segmentation_metrics["confusion_matrix_labels"] = self.emotion_names

        tags_true_sets = []
        tags_pred_sets = []
        tags_used = 0
        for link, g in gt.items():
            if link not in pred_tags:
                continue
            true_tags = set(self.parse_tags(g.get("Tags")))
            pred_tags_list = set(self.parse_tags(pred_tags[link].get("annotated_dialogue")))
            tags_true_sets.append(true_tags)
            tags_pred_sets.append(pred_tags_list)
            tags_used += 1
        tags_metrics = self.multilabel_metrics(tags_true_sets, tags_pred_sets) if tags_used > 0 else None
        q_expected = 5
        q_true_flat = [[] for _ in range(q_expected)]
        q_pred_flat = [[] for _ in range(q_expected)]
        q_exact = 0
        q_used = 0
        for link, g in gt.items():
            if link not in pred_questions:
                continue
            true_list = self.parse_scale_answers(g.get("Result"), expected_len=q_expected)
            pred_list = self.parse_scale_answers(pred_questions[link].get("annotated_dialogue"), expected_len=q_expected)
            if true_list is None or pred_list is None:
                continue
            if true_list == pred_list:
                q_exact += 1
            for k in range(q_expected):
                q_true_flat[k].append(true_list[k])
                q_pred_flat[k].append(pred_list[k])
            q_used += 1

        questions_metrics = None
        if q_used > 0:
            per_question = {}
            for k in range(q_expected):
                name = self.general_question_names[k] if k < len(self.general_question_names) else str(k)
                per_question[name] = self.ordinal_metrics(q_true_flat[k], q_pred_flat[k])
            overall_true = [v for arr in q_true_flat for v in arr]
            overall_pred = [v for arr in q_pred_flat for v in arr]
            questions_metrics = {
                "num_dialogues": int(q_used),
                "exact_match_rate": float(q_exact / q_used),
                "overall": self.ordinal_metrics(overall_true, overall_pred),
                "per_question": per_question
            }
        m_q = 4
        msg_true_flat = [[] for _ in range(m_q)]
        msg_pred_flat = [[] for _ in range(m_q)]
        msg_exact = 0
        msg_used_dialogues = 0
        msg_used_messages = 0
        for link, g in gt.items():
            if link not in pred_psych:
                continue
            true_digits = self.parse_digits_1_to_6(g.get("MessageMark"))
            pred_digits = self.parse_digits_1_to_6(pred_psych[link].get("annotated_dialogue"))
            true_msgs = len(true_digits) // m_q
            pred_msgs = len(pred_digits) // m_q
            common = min(true_msgs, pred_msgs)
            if common <= 0:
                continue
            for i_msg in range(common):
                t_chunk = true_digits[i_msg * m_q:(i_msg + 1) * m_q]
                p_chunk = pred_digits[i_msg * m_q:(i_msg + 1) * m_q]
                if t_chunk == p_chunk:
                    msg_exact += 1
                for k in range(m_q):
                    msg_true_flat[k].append(t_chunk[k])
                    msg_pred_flat[k].append(p_chunk[k])
            msg_used_dialogues += 1
            msg_used_messages += common

        psych_metrics = None
        if msg_used_messages > 0:
            per_q = {}
            for k in range(m_q):
                name = self.psych_question_names[k] if k < len(self.psych_question_names) else str(k)
                per_q[name] = self.ordinal_metrics(msg_true_flat[k], msg_pred_flat[k])
            overall_true = [v for arr in msg_true_flat for v in arr]
            overall_pred = [v for arr in msg_pred_flat for v in arr]
            psych_metrics = {
                "num_dialogues": int(msg_used_dialogues),
                "num_messages": int(msg_used_messages),
                "per_message_exact_match_rate": float(msg_exact / msg_used_messages),
                "overall": self.ordinal_metrics(overall_true, overall_pred),
                "per_question": per_q
            }

        final_metrics = {
            "counts": {
                "gt_dialogues": int(len(gt)),
                "pred_seg_dialogues": int(len(pred_seg)),
                "pred_tags_dialogues": int(len(pred_tags)),
                "pred_questions_dialogues": int(len(pred_questions)),
                "pred_psych_dialogues": int(len(pred_psych)),
                "seg_processed": int(seg_processed),
                "seg_skipped": int(seg_skipped),
                "tags_used": int(tags_used),
                "questions_used": int(q_used),
                "psych_used_dialogues": int(msg_used_dialogues),
                "psych_used_messages": int(msg_used_messages)
            },
            "segmentation": segmentation_metrics,
            "tags": tags_metrics,
            "questions": questions_metrics,
            "psych_messages": psych_metrics
        }
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=2)
        return final_metrics


def main():
    CONFIG = {
        "segmentation_dir": "qwen32b_final",
        "tags_dir": "tags",
        "questions_dir": "quest",
        "psych_dir": "psych",
        "ground_truth_file": "output_markup.json",
        "output_dir": "benchmark_results"
    }
    benchmark = PremarkedEmotionBenchmark()
    try:
        benchmark.run_full_benchmark(
            segmentation_dir=CONFIG["segmentation_dir"],
            tags_dir=CONFIG["tags_dir"],
            questions_dir=CONFIG["questions_dir"],
            psych_dir=CONFIG["psych_dir"],
            ground_truth_file=CONFIG["ground_truth_file"],
            output_dir=CONFIG["output_dir"]
        )
    except Exception as e:
        print(f"\nError during benchmark execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
