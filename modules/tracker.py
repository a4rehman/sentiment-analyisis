# ============================================================
#  modules/tracker.py – Simple Centroid / IoU Face Tracker
# ============================================================
"""
Assigns persistent track IDs to faces across frames so that:
  - Emotions are attributed to the same person over time
  - Crowd statistics can be computed per individual
"""

from __future__ import annotations
import logging
from collections import OrderedDict
from typing import List, Tuple, Dict

import numpy as np

from modules.detector import FaceResult

logger = logging.getLogger(__name__)


def _iou(boxA: tuple, boxB: tuple) -> float:
    """Compute Intersection-over-Union of two (x,y,w,h) boxes."""
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    xA = max(ax, bx);  yA = max(ay, by)
    xB = min(ax + aw, bx + bw);  yB = min(ay + ah, by + bh)
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _centroid(box: tuple) -> Tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


class FaceTracker:
    """
    Greedy IoU tracker.

    Maintains a registry of active tracks.  Each call to `update()`
    assigns track IDs to the current detections and drops stale tracks.
    """

    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.3):
        self._next_id       = 0
        self._max_disapp    = max_disappeared
        self._iou_thresh    = iou_threshold

        # OrderedDict: track_id → {"bbox": ..., "disappeared": int, "emotion": ..., "history": [...]}
        self._tracks: Dict[int, dict] = OrderedDict()

    # ------------------------------------------------------------------
    def update(self, results: List[FaceResult]) -> List[FaceResult]:
        """
        Match detections to existing tracks via IoU, assign IDs,
        handle disappearances, and return updated FaceResult list.
        """
        if not results:
            # Increment disappeared counter for all active tracks
            for tid in list(self._tracks.keys()):
                self._tracks[tid]["disappeared"] += 1
                if self._tracks[tid]["disappeared"] > self._max_disapp:
                    del self._tracks[tid]
            return results

        current_ids  = list(self._tracks.keys())
        det_bboxes   = [r.bbox for r in results]

        if not current_ids:
            # No existing tracks – register all detections
            for r in results:
                self._register(r)
        else:
            track_bboxes = [self._tracks[tid]["bbox"] for tid in current_ids]

            # Build IoU matrix  (tracks × detections)
            iou_mat = np.zeros((len(current_ids), len(det_bboxes)))
            for i, tb in enumerate(track_bboxes):
                for j, db in enumerate(det_bboxes):
                    iou_mat[i, j] = _iou(tb, db)

            # Greedy assignment: best IoU first
            matched_tracks = set()
            matched_dets   = set()

            flat_order = np.argsort(iou_mat.ravel())[::-1]
            for idx in flat_order:
                i, j = divmod(int(idx), len(det_bboxes))
                if iou_mat[i, j] < self._iou_thresh:
                    break
                if i in matched_tracks or j in matched_dets:
                    continue
                tid = current_ids[i]
                self._update_track(tid, results[j])
                matched_tracks.add(i)
                matched_dets.add(j)

            # Handle unmatched tracks
            for i, tid in enumerate(current_ids):
                if i not in matched_tracks:
                    self._tracks[tid]["disappeared"] += 1
                    if self._tracks[tid]["disappeared"] > self._max_disapp:
                        del self._tracks[tid]

            # Register new detections
            for j, r in enumerate(results):
                if j not in matched_dets:
                    self._register(r)

        # Stamp track IDs back onto results
        # We rebuild an output list based on current track state
        out: List[FaceResult] = []
        for tid, info in self._tracks.items():
            r = info.get("result")
            if r is not None and info["disappeared"] == 0:
                r.track_id = tid
                out.append(r)

        return out

    # ------------------------------------------------------------------
    def get_track_history(self, track_id: int) -> List[str]:
        """Return list of recent emotion labels for a track."""
        return self._tracks.get(track_id, {}).get("history", [])

    def all_track_ids(self) -> List[int]:
        return list(self._tracks.keys())

    # ------------------------------------------------------------------
    def _register(self, result: FaceResult):
        tid = self._next_id
        self._tracks[tid] = {
            "bbox":        result.bbox,
            "disappeared": 0,
            "result":      result,
            "history":     [result.emotion],
        }
        result.track_id = tid
        self._next_id  += 1

    def _update_track(self, tid: int, result: FaceResult):
        hist = self._tracks[tid]["history"]
        hist.append(result.emotion)
        if len(hist) > 100:
            hist.pop(0)
        self._tracks[tid].update({
            "bbox":        result.bbox,
            "disappeared": 0,
            "result":      result,
            "history":     hist,
        })
        result.track_id = tid
