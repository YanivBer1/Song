"""Export the avatar's hand/finger rest skeleton for asl_v2.handfit (run with Blender's Python):
  .venv-blender/bin/python asl_v2/export_rig.py   -> asl_v2/models/rig_hand.json"""
import json
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mh_avatar  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rig, _ = mh_avatar.build_character()
out = {}
for b in rig.data.bones:
    base = b.name.rsplit("_", 1)[0]
    if base.split("_")[0] in ("hand", "thumb", "index", "middle", "ring", "pinky", "lowerarm"):
        out[b.name] = dict(rest=np.array(b.matrix_local)[:3, :3].tolist(), head=list(b.head_local),
                           tail=list(b.tail_local), parent=b.parent.name if b.parent else None)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rig_hand.json")
json.dump(out, open(path, "w"))
print("exported", len(out), "bones")
