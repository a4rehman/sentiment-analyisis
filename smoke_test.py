import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.detector import FaceResult
from modules.tracker import FaceTracker
from modules.alerts import AlertSystem
from modules.analytics import SessionStats, build_bar_chart, build_timeline_chart, build_face_count_chart, fig_to_bytes, export_summary_txt, export_csv_report
from modules.logger import EmotionLogger

# Two stable faces, boxes identical across frames (high IoU => persistent ids)
box_a = (10, 10, 50, 50)
box_b = (200, 100, 60, 60)
r1 = FaceResult(box_a, "happy", 0.9, {"happy": 0.9, "neutral": 0.1})
r3 = FaceResult(box_b, "angry", 0.95, {"angry": 0.95, "neutral": 0.05})
r2 = FaceResult(box_a, "happy", 0.88, {"happy": 0.88, "neutral": 0.12})
r4 = FaceResult(box_b, "angry", 0.93, {"angry": 0.93, "neutral": 0.07})

t = FaceTracker()
out = t.update([r1, r3])
ids_1 = {r.track_id for r in out}
assert len(ids_1) == 2, ids_1
out = t.update([r2, r4])
ids_2 = {r.track_id for r in out}
assert ids_2 == ids_1, f"track ids should persist across frames: {ids_1} -> {ids_2}"
print("tracker OK: ids persist", ids_2)

a = AlertSystem(anger_conf_thresh=0.5, anger_count_thresh=2)
alerts = a.check([r3, r4])
assert any(x.level == "CRITICAL" for x in alerts), alerts
print("alerts OK:", [repr(x) for x in alerts])

s = SessionStats()
s.update([r1, r3])
s.update([r2, r4])
assert s.total_events() == 4
assert s.dominant_emotion() is not None
assert len(fig_to_bytes(build_bar_chart(s))) > 0
assert len(fig_to_bytes(build_timeline_chart(s))) > 0
assert len(fig_to_bytes(build_face_count_chart(s))) > 0
print("charts OK")
print("stats OK: total=%d dom=%s dist=%s" % (s.total_events(), s.dominant_emotion(), s.emotion_distribution()))

lg = EmotionLogger(session_name="smoke")
lg.log([r1, r3], source="video_file")
lg.flush()
assert Path(lg.log_path).exists()
import pandas as pd
df = pd.read_csv(lg.log_path)
assert len(df) == 2
print("logger OK:", lg.log_path)

p = export_summary_txt(s, session_name="smoke")
assert Path(p).exists()
print("summary OK:", p)
print("ALL SMOKE TESTS PASSED")