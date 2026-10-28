"""
USING THE LOGGED DATA FROM ANOTHER ADD-ON
-----------------------------------------
While this add-on is enabled, a ready-made API object is published in Blender's shared
Python namespace, so other add-ons do not need to import this file (its module name
changes with how it was installed):

    import bpy
    api = bpy.app.driver_namespace.get("motion_data_logger")
    if api is not None and api.has_data():
        arr  = api.get_array("Cube")              # numpy (N, 8): time, frame, loc xyz, rot xyz
        locz = api.get_column("Cube", "loc_z")    # one column as a numpy array
        now  = api.sample_at_frame("Cube", 12.5)  # interpolated dict, e.g. now["loc_x"]

        vv = api.get_vertex_array("Cube")         # numpy (N, K, 6): sample, vertex, loc xyz rot xyz
        k  = api.get_vertex_count("Cube")         # K = vertices logged for THAT object
        v3z = api.get_vertex_column("Cube", 2, "loc_z")   # vertex 3 (index 2), one column
        v = api.vertex_sample_at_frame("Cube", 12.5)      # v[2]["loc_z"], v[2]["rot_x"], ...

        def on_event(event, info):                # "started" "progress" "finished"
            print(event, info)                    # "cancelled" "cleared" "loaded"
        api.subscribe(on_event)                   # call api.unsubscribe(on_event) when done
        api.store_per                             # "SHUTTER" (sample interval) or "FRAME" (one per frame)

The whole-object location (Object view) is the point where the straight paths from every
vertex to the vertex directly across from it converge; for a symmetric mesh such as a UV
sphere they all cross at one point, for any other mesh it is the point closest to all of them.

FILM REELS (made by FilmReel_Generator)
---------------------------------------
Add a reel's strip object to the tracked list and choose what to log for it (Reel Mode):
  Reel + Frames   the whole reel AND every frame, each as its own logged item
  Frames only     one item per frame
  Whole reel only one item for the whole strip (centre + rotation, no vertex columns)
  As plain mesh   old behaviour: the strip is one object with every vertex of every frame
Items are named  "<strip> (Reel)"  and  "<strip> (Frame 007)"  (frame = position on the strip,
1 = first). A frame logs its own vertices, in the same face order, with its own wireframe icon,
exactly like an object. api.reel_key("Film Reel") and api.frame_key("Film Reel", 7) return the
names to pass to every other api call; api.related("Film Reel") lists all of them.

Units: time = seconds, frame = scene frames, loc = meters, rot = XYZ Euler degrees.
Vertices are numbered in a fixed order, taken from the world-space positions at the first
sample: vertex 1 is the highest-Z vertex (ties: highest X, then closest to Y = 0). Faces are
taken in rows from +Z down to -Z (one row at a time); inside a row they go once around the
vertical axis from the +X side toward +Y. Each face is walked from its top corner down (-Z),
on toward +Y, back up (+Z) and back to the start, closing the face; a vertex shared by several
faces is numbered where it is first reached. Vertices that belong to no face come last.
Every object logs ALL of its own vertices (no limit) and gets its own wireframe icon with that vertex marked in yellow.
All values handed out are copies, so other add-ons cannot corrupt the log.
"""

bl_info = {
    "name": "Motion Data Logger",
    "author": "Claude",
    "version": (1, 10, 0),
    "blender": (3, 6, 0),
    "location": "3D Viewport > Sidebar (N) > Motion Log",
    "description": "Logs location and rotation of every vertex (corner) of any number of "
                   "objects, and of a Film Reel's frames and the whole reel, every 0.0021 s, with a wireframe icon of the object above each vertex "
                   "column, a log display driven by the chosen frame rate and a public API "
                   "other add-ons can read",
    "category": "Animation",
}

import base64
import io
import json
import math
import os
import shutil
import struct
import tempfile
import time
import traceback
import zlib

import bpy
import bpy.utils.previews
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
    IntProperty, PointerProperty, StringProperty,
)
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup, UIList
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix

# ----------------------------------------------------------------------------
# In-memory log (lives inside the add-on; use "Export Log to Text" to write a text file)
#
# _LOG = {
#   "interval": float seconds between samples,
#   "data": { object_name: [ row, row, ... ] }
# }
# row = (t, frame, lx, ly, lz, rx, ry, rz)
#   t            time in seconds from the start of the recorded range
#   frame        (fractional) scene frame
#   l*           location  = convergence point of the paths from each vertex to the vertex
#                directly across from it (the object's centre found from its vertices), meters
#   r*           rotation  = best-fit (Kabsch) rotation of the vertices, XYZ Euler, degrees
# ----------------------------------------------------------------------------
# _LOG["verts"] = { object_name: [ vrow, vrow, ... ] }  (same length/order as "data")
# vrow = float64 array of K vertices x (lx, ly, lz, rx, ry, rz) flattened to 6*K values, where
#        K depends on the object (its own vertex count, no limit; 0 for objects
#        without geometry). NaN where a vertex disappears during the recording.
#        Vertex n = n-th vertex in face order: rows of faces from +Z down to -Z, once round the axis (+X toward +Y) inside
#        a row, each face walked around its perimeter (see _vertex_order), using world
#        positions at the first sample.
# _LOG["vmeta"] = { object_name: {"total": mesh vertex count, "co": (K,3) rest positions,
#                                 "edges": (M,2) edges between logged vertices} }
#        (what the wireframe icons are drawn from)
_LOG = {"interval": 0.0021, "data": {}, "verts": {}, "vmeta": {}, "fps": None,
        "start_frame": None, "store_per": "SHUTTER"}
# "store_per": how the last recording was stored. "SHUTTER" = one sample every Sample Interval
# seconds; "FRAME" = one sample per frame at the project's frame rate (interval = 1 / fps).

VERT_FIELDS = ("loc_x", "loc_y", "loc_z", "rot_x", "rot_y", "rot_z")
NF = len(VERT_FIELDS)


def _clear_data():
    _LOG["data"] = {}
    _LOG["verts"] = {}
    _LOG["vmeta"] = {}
    _reset_icons()


STORE_ITEMS = [
    ("SHUTTER", "Per Shutter",
     "Store a sample every Sample Interval seconds (the shutter-based sampling)"),
    ("FRAME", "Per Frame",
     "Store one sample for every frame, at the project's frame rate. Sample Interval is not used"),
]


def _record_interval(scene):
    """Seconds between stored samples for the chosen storage mode: the Sample Interval (per
    shutter; Blender float properties are 32-bit, so 0.0021 becomes 0.002099999925... and the
    shortest decimal that round-trips to that float32 is the value the user typed) or exactly
    one frame at the project's frame rate (per frame)."""
    s = scene.motion_log
    if s.store_per == "FRAME":
        return 1.0 / (scene.render.fps / scene.render.fps_base)
    return float(str(np.float32(s.sample_interval)))


def _store_label():
    return "per frame" if _LOG.get("store_per") == "FRAME" else "per shutter"


def _fmt_seconds(x):
    return f"{x:.6g}"


FPS_ITEMS = [(str(f), f"{f} fps", f"Display one log row per frame at {f} fps")
             for f in (12, 24, 25, 30, 48, 50, 60, 100, 120, 240)]

HEADERS = ("time_s", "frame_scene_frames", "loc_x_m", "loc_y_m", "loc_z_m", "rot_x_deg",
           "rot_y_deg", "rot_z_deg")


def _vert_headers(k):
    return tuple(f"v{v + 1}_{f}_{'m' if f.startswith('loc') else 'deg'}"
                 for v in range(k) for f in VERT_FIELDS)


# ----------------------------------------------------------------------------
# Data capture
# ----------------------------------------------------------------------------
def _eval_full(obj_eval, state):
    """Return (local Nx3, world Nx3, normals Nx3, first-neighbour index per vertex) from the
    evaluated object, or None. K (state["k"]) = number of vertices logged individually: all of
    them, fixed at the first sample so every row of the log has the same length."""
    try:
        mesh = obj_eval.to_mesh()
    except RuntimeError:
        return None
    if mesh is None or len(mesh.vertices) == 0:
        obj_eval.to_mesh_clear()
        return None
    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    try:
        nor = np.empty(n * 3, dtype=np.float64)
        mesh.vertices.foreach_get("normal", nor)
        normals = nor.reshape(n, 3)
    except Exception:
        normals = np.array([tuple(mesh.vertices[i].normal) for i in range(n)], dtype=np.float64)

    # One connected neighbour per vertex gives each vertex a stable tangent direction.
    key = (n, len(mesh.edges))
    if state.get("nbr_key") != key:
        e = np.empty(len(mesh.edges) * 2, dtype=np.int32)
        mesh.edges.foreach_get("vertices", e)
        e = e.reshape(-1, 2)
        nbr = np.full(n, -1, dtype=np.int64)
        if len(e):  # earliest edge wins (assigned last)
            nbr[e.reshape(-1)[::-1]] = e[:, ::-1].reshape(-1)[::-1]
        state["nbr_key"], state["nbr"], state["edges"] = key, nbr, e
    if "faces" not in state:  # face corner lists, only needed once to fix the vertex order
        npoly = len(mesh.polygons)
        if npoly:
            lv = np.empty(len(mesh.loops), dtype=np.int32)
            mesh.loops.foreach_get("vertex_index", lv)
            ls = np.empty(npoly, dtype=np.int32)
            mesh.polygons.foreach_get("loop_start", ls)
            state["faces"] = (lv.astype(np.int64), ls.astype(np.int64))
        else:
            state["faces"] = None
    obj_eval.to_mesh_clear()

    local = co.reshape(n, 3)
    mw = np.array(obj_eval.matrix_world, dtype=np.float64)
    world = local @ mw[:3, :3].T + mw[:3, 3]
    return local, world, normals, state["nbr"]


def _eval_vertices(obj_eval, state):
    """Vertices an item logs. A plain object logs its whole mesh. A Film Reel item shares ONE
    evaluation of the strip per sample (state["shared"]) and either logs all of it as a summary
    only (state["whole"]: K = 0 vertex columns) or just the vertices / faces of one frame
    (state["frame"] = (k, n_frames); frame k owns vertices k*nv .. (k+1)*nv - 1, because the strip
    is n identical copies of the frame mesh)."""
    sh = state.get("shared")
    if sh is None:
        full = _eval_full(obj_eval, state)
    else:
        if sh["stamp"] is not None and sh["stamp"] == sh["cur"]:
            full = sh["full"]
        else:
            full = _eval_full(obj_eval, sh["static"])
            sh["full"], sh["stamp"] = full, sh["cur"]
    if full is None:
        return None
    local, world, normals, nbr = full
    n = len(local)
    if sh is None:
        if "k" not in state:
            state["k"] = n
            state["total"] = n
        return full
    static = sh["static"]
    if state.get("whole"):
        state.setdefault("k", 0)
        state.setdefault("total", n)
        return full
    k, nfr = state["frame"]
    if n % nfr:
        raise ValueError("the reel's mesh no longer divides into its frames")
    nv = n // nfr
    a, b = k * nv, (k + 1) * nv
    if "k" not in state:
        state["k"], state["total"] = nv, nv
        e = static.get("edges")
        state["edges"] = (None if e is None else e[(e >= a).all(axis=1) & (e < b).all(axis=1)] - a)
        f = static.get("faces")
        if f is None:
            state["faces"] = None
        else:
            lv, ls = f
            npoly = len(ls)
            pk = npoly // nfr
            if npoly % nfr:
                raise ValueError("the reel's faces no longer divide into its frames")
            p0, p1 = k * pk, (k + 1) * pk
            l0 = int(ls[p0])
            l1 = int(ls[p1]) if p1 < npoly else len(lv)
            state["faces"] = (lv[l0:l1] - a, ls[p0:p1] - l0)
        nb = nbr[a:b].copy()
        nb = np.where((nb >= a) & (nb < b), nb - a, -1)
        state["nbr_slice"] = nb
    return local[a:b], world[a:b], normals[a:b], state["nbr_slice"]


def _legacy_vertex_order(world):
    """Order by position alone: from -Y to +Y; vertices on the same Y (a row) from +Z to -Z;
    ties from -X to +X. Positions are rounded so tiny float noise does not split a row."""
    ext = float(np.ptp(world, axis=0).max())
    tol = max(ext * 1e-6, 1e-12)
    x, y, z = (np.round(world[:, i] / tol) for i in range(3))
    return np.lexsort((x, -z, y))  # last key is the primary one


def _vertex_order(world, faces=None):
    """Order in which vertices are logged, from the world positions at the first sample.

    1. First vertex: the highest Z of the mesh; if several share it, the highest X, then the
       one closest to Y = 0 (then the lowest Y).
    2. Faces are taken in rows: faces whose centres have (about) the same Z form a row; the
       top row (+Z) comes first, then one row down (-Z) at a time. Inside a row the faces go
       once around the vertical axis through the middle of the mesh, starting at the +X side
       and heading toward +Y (counter-clockwise seen from above). The row holding the first
       vertex starts with the first face that contains it.
    3. Each face is walked like this: start at its top corner (highest Z; if several, the
       left-most one); go to the neighbouring corner in the -Z direction; then on toward +Y
       (to the right); then back up in the +Z direction; then back to the start, which closes
       the face (square, triangle or any n-gon). A triangle whose top is a single corner (the
       pole of a sphere) goes to its left lower corner first. A perfectly flat face starts at
       its highest-X corner (closest to Y = 0) and goes round counter-clockwise from above.
       "Left / right" means toward -Y / +Y at the +X start; further round the ring it means
       against / along the counter-clockwise direction, so every face is walked the same way.
    4. A vertex is logged the first time a face reaches it; corners already logged by an
       earlier face are skipped, so the next face only adds its new corners.
    5. Vertices that belong to no face come last, ordered by position.
    Without faces the order is by position only.
    Returns the original vertex index for each logged position."""
    legacy = _legacy_vertex_order(world)
    if faces is None:
        return legacy
    loop_vert, start = faces
    n, nf = len(world), len(start)
    ext = float(np.ptp(world, axis=0).max())
    tol = max(ext * 1e-6, 1e-12)         # "same position"
    ztol = max(ext * 1e-5, 1e-12)        # "same height"

    counts = np.diff(np.append(start, len(loop_vert)))
    face_of = np.repeat(np.arange(nf), counts)
    cen = np.add.reduceat(world[loop_vert], start, axis=0) / counts[:, None]

    # --- rows of faces: face centres sorted by Z (top first); a gap bigger than row_tol starts a row
    row_tol = max(ext * 1e-4, 1e-12)
    by_z = np.argsort(-cen[:, 2], kind="stable")
    zs = cen[by_z, 2]
    row_sorted = np.concatenate(([0], np.cumsum((zs[:-1] - zs[1:]) > row_tol)))
    row = np.empty(nf, dtype=np.int64)
    row[by_z] = row_sorted

    # --- inside a row: once round the vertical axis, from +X toward +Y
    mid = (world.min(axis=0) + world.max(axis=0)) / 2.0
    rel = cen[:, :2] - mid[:2]
    rn = np.hypot(rel[:, 0], rel[:, 1])
    on_axis = rn < max(ext * 1e-4, 1e-12)
    ang = np.where(on_axis, 0.0, np.arctan2(rel[:, 1], rel[:, 0]))
    ang = np.where(ang < -1e-6, ang + 2.0 * math.pi, ang)
    safe = np.maximum(rn, 1e-300)
    rx = np.where(on_axis, 0.0, -rel[:, 1] / safe)      # "to the right" at each face (toward +Y at +X)
    ry = np.where(on_axis, 1.0, rel[:, 0] / safe)
    face_order = np.lexsort((np.round(rn / tol), np.round(ang * 1e6), row))

    # --- the first vertex of the whole mesh
    on_face = np.zeros(n, dtype=bool)
    on_face[loop_vert] = True
    cand = np.flatnonzero(on_face)
    top = cand[world[cand, 2] >= world[cand, 2].max() - ztol]
    top = top[world[top, 0] >= world[top, 0].max() - ztol]
    ys = world[top, 1]
    g = int(top[np.lexsort((ys, np.round(np.abs(ys) / tol)))[0]])

    rows = {}
    for f in face_order.tolist():
        rows.setdefault(int(row[f]), []).append(f)
    row_list = [rows[k] for k in sorted(rows)]
    g_faces = set(face_of[loop_vert == g].tolist())
    for ri, rl in enumerate(row_list):
        p = next((i for i, f in enumerate(rl) if f in g_faces), None)
        if p is not None:           # that row goes first and starts at the first face holding g
            row_list.pop(ri)
            row_list.insert(0, rl[p:] + rl[:p])
            break
    first_face = row_list[0][0]

    # --- per-corner numbers for the walk (plain lists: the loop below runs once per face)
    cu = ((world[loop_vert, 0] - cen[face_of, 0]) * rx[face_of]
          + (world[loop_vert, 1] - cen[face_of, 1]) * ry[face_of])
    ul, zl = cu.tolist(), world[loop_vert, 2].tolist()
    xl, yl = world[loop_vert, 0].tolist(), world[loop_vert, 1].tolist()
    lvl, st, cn = loop_vert.tolist(), start.tolist(), counts.tolist()

    def left_first(i):      # left-most corner first, then lowest Y, then lowest X
        return (round(ul[i] / tol), round(yl[i] / tol), round(xl[i] / tol))

    def flat_first(i):      # highest X first, then closest to Y = 0, then lowest Y
        return (-round(xl[i] / tol), round(abs(yl[i]) / tol), round(yl[i] / tol))

    def walk(f, force=None):
        a, m = st[f], cn[f]
        loops = range(a, a + m)
        zmax, zmin = max(zl[a:a + m]), min(zl[a:a + m])
        flat = zmax - zmin <= ztol
        if force is not None and force in lvl[a:a + m]:
            s0 = a + lvl[a:a + m].index(force)
        elif flat:
            s0 = min(loops, key=flat_first)
        else:
            s0 = min((i for i in loops if zl[i] >= zmax - ztol), key=left_first)
        s = s0 - a
        nxt, prv = a + (s + 1) % m, a + (s - 1) % m
        d = 0
        if not flat:
            down_n = zl[nxt] < zl[s0] - ztol
            down_p = zl[prv] < zl[s0] - ztol
            if down_n != down_p:               # the one lower corner: that is the -Z step
                d = 1 if down_n else -1
            elif down_n and down_p:            # both lower (top of a fan): left one first
                d = 1 if left_first(nxt) <= left_first(prv) else -1
        if d == 0:                             # flat face: counter-clockwise seen from above
            area2 = sum(xl[a + i] * yl[a + (i + 1) % m] - xl[a + (i + 1) % m] * yl[a + i]
                        for i in range(m))
            d = 1 if area2 >= 0.0 else -1
        return [lvl[a + (s + d * i) % m] for i in range(m)]

    seen = bytearray(n)
    out = []
    for rl in row_list:
        for f in rl:
            for v in walk(f, g if f == first_face else None):
                if not seen[v]:
                    seen[v] = 1
                    out.append(v)
    seen_np = np.frombuffer(bytes(seen), dtype=np.uint8).astype(bool)
    rest = legacy[~seen_np[legacy]]
    return np.concatenate((np.array(out, dtype=np.int64), rest.astype(np.int64)))


def _converge_point(pts, ia, ib, fallback=None):
    """Point where the paths vertex -> opposite vertex converge: the point with the smallest
    total squared distance to all those straight paths. If every path crosses at one point
    (a UV sphere, a cube, any symmetric object) that is exactly the crossing point. Paths
    of zero length are ignored; a nearly flat or one-directional set of paths is steadied
    by a tiny pull toward the average path midpoint."""
    a, b = pts[ia], pts[ib]
    d = b - a
    length = np.linalg.norm(d, axis=1)
    ok = length > 1e-12
    if not ok.any():
        return pts.mean(axis=0) if fallback is None else fallback
    d = d[ok] / length[ok, None]
    mid = (a[ok] + b[ok]) * 0.5
    cnt = len(d)
    avg = mid.mean(axis=0)
    eps = 1e-10 * cnt
    lhs = (cnt + eps) * np.eye(3) - d.T @ d
    rel = mid - avg                      # solved relative to the average midpoint (steadier)
    rhs = rel.sum(axis=0) - (d * np.sum(d * rel, axis=1, keepdims=True)).sum(axis=0)
    try:
        return avg + np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return avg


def _opposite_of(pts, centre, block=1024):
    """For every vertex, the index of the vertex lying most directly on the other side of
    `centre` (the direction from the centre that points most nearly opposite). Vertices
    sitting on the centre have no direction and are paired with themselves (ignored)."""
    r = pts - centre
    nr = np.linalg.norm(r, axis=1)
    live = nr > 1e-12
    u = np.zeros_like(r)
    u[live] = r[live] / nr[live, None]
    out = np.arange(len(pts))
    for s in range(0, len(pts), block):
        score = -(u[s:s + block] @ u.T)                  # 1 = exactly opposite
        score[:, ~live] = -2.0
        # exact ties (a ring of equal choices) go to the one farthest from the centre
        score = score + 1e-12 * nr[None, :] / max(float(nr.max()), 1e-300)
        out[s:s + block] = np.argmax(score, axis=1)
    out[~live] = np.flatnonzero(~live)
    return out


def _opposite_pairs(world, max_points=4000, passes=5):
    """Pair every vertex with the vertex directly across from it, found from the first
    sample and kept for the whole recording. Starts from the middle of the bounding box,
    pairs, moves to where the paths converge, and pairs again, so objects that are not
    centred on their origin still pair correctly. Very dense meshes use an evenly spread
    subset of at most `max_points` vertices to keep the search fast.
    Returns (index of vertex, index of its opposite) as two arrays, or None."""
    n = len(world)
    if n < 2:
        return None
    sub = np.arange(n)
    if n > max_points:
        sub = sub[:: int(math.ceil(n / max_points))]
    p = world[sub]
    centre = (p.min(axis=0) + p.max(axis=0)) * 0.5
    for _ in range(passes):
        opp = _opposite_of(p, centre)
        centre = _converge_point(p, np.arange(len(p)), opp, centre)
    opp = _opposite_of(p, centre)
    keep = opp != np.arange(len(p))
    return sub[keep], sub[opp[keep]]


def _kabsch(ref_local, world):
    """Rotation matrix that best maps the (centred) reference vertices onto the world vertices."""
    p = ref_local - ref_local.mean(axis=0)
    q = world - world.mean(axis=0)
    h = p.T @ q
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    if d == 0:
        d = 1.0
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def _vertex_frames(pos, normals, nb_pos, has_nb):
    """Orthonormal frame (columns x, y, z) of each vertex in the mesh's local space, shape
    (K, 3, 3): z = vertex normal, x = direction to a connected neighbour (made perpendicular
    to z). Loose vertices and degenerate edges fall back to the axis least aligned with z."""
    nz = np.linalg.norm(normals, axis=1, keepdims=True)
    z = np.where(nz > 1e-12, normals / np.maximum(nz, 1e-12), np.array([0.0, 0.0, 1.0]))
    t = np.where(has_nb[:, None], nb_pos - pos, 0.0)
    t = t - z * np.sum(t * z, axis=1, keepdims=True)
    tn = np.linalg.norm(t, axis=1, keepdims=True)
    bad = tn[:, 0] < 1e-12
    if bad.any():
        ax = np.eye(3)[np.argmin(np.abs(z[bad]), axis=1)]
        alt = ax - z[bad] * np.sum(ax * z[bad], axis=1, keepdims=True)
        t[bad] = alt
        tn[bad] = np.linalg.norm(alt, axis=1, keepdims=True)
    x = t / tn
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=2)


def _euler_xyz_batch(m, prev):
    """XYZ Euler angles (radians, shape (K, 3)) for K rotation matrices at once, the same
    convention as Matrix.to_euler("XYZ"). Of the two equivalent solutions the one closest to
    `prev` (K, 3) is returned, unwrapped by whole turns, so angles stay continuous."""
    cy = np.hypot(m[:, 0, 0], m[:, 1, 0])
    gimbal = cy < 1e-6
    y1 = np.arctan2(-m[:, 2, 0], cy)
    x1 = np.where(gimbal, np.arctan2(-m[:, 1, 2], m[:, 1, 1]), np.arctan2(m[:, 2, 1], m[:, 2, 2]))
    z1 = np.where(gimbal, 0.0, np.arctan2(m[:, 1, 0], m[:, 0, 0]))
    a = np.stack((x1, y1, z1), axis=1)
    if prev is None:
        return a
    y2 = np.arctan2(-m[:, 2, 0], -cy)
    x2 = np.where(gimbal, x1, np.arctan2(-m[:, 2, 1], -m[:, 2, 2]))
    z2 = np.where(gimbal, z1, np.arctan2(-m[:, 1, 0], -m[:, 0, 0]))
    b = np.stack((x2, y2, z2), axis=1)
    tau = 2.0 * math.pi
    a = a + tau * np.round((prev - a) / tau)
    b = b + tau * np.round((prev - b) / tau)
    use_b = np.abs(b - prev).sum(axis=1) < np.abs(a - prev).sum(axis=1)
    return np.where(use_b[:, None], b, a)


def _pure_rotation(m3):
    """Closest pure rotation to a 3x3 matrix (removes scale/shear)."""
    u, _s, vt = np.linalg.svd(m3)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, 2] *= -1.0
        r = u @ vt
    return r


def sample_object(obj, depsgraph, state, stride, unit_scale=1.0):
    """Capture one sample for obj. state is a per-object dict kept across samples.
    Returns (object_values[6], vertex_values[6 * K]); location in meters (Blender units x
    scene unit scale), rotation as XYZ Euler degrees. K is this object's own vertex count."""
    obj_eval = obj.evaluated_get(depsgraph)
    verts = _eval_vertices(obj_eval, state)

    if verts is not None:
        local_all, world_all, normals, nbr_all = verts
        local = local_all[::stride]
        world = world_all[::stride]
        # Object location: the point where the paths from every vertex to the vertex directly
        # across from it converge (pairs found once, at the first sample, and kept).
        pr = state.get("pairs")
        if pr is None or pr[0] != len(world):
            state["pairs"] = pr = (len(world), _opposite_pairs(world))
        loc = (world.mean(axis=0) if pr[1] is None
               else _converge_point(world, pr[1][0], pr[1][1], world.mean(axis=0)))
        ref = state.get("ref")
        if ref is None or len(ref) != len(local):
            state["ref"] = ref = local.copy()
        rot = _kabsch(ref, world) if len(local) >= 3 else np.array(
            obj_eval.matrix_world.to_3x3().normalized())

        # --- per-vertex location and rotation (every logged vertex, in the fixed order) ---
        kk = state["k"]
        if kk == 0:           # whole-reel summary: no individual vertices
            vert_vals = np.empty(0)
        else:
            perm = state.get("perm")
            if perm is None:
                state["perm"] = perm = _vertex_order(world_all, state.get("faces"))
            ok = perm < len(local_all)          # False only if the mesh lost vertices mid-recording
            sel = perm[ok]
            pos = local_all[sel]
            nb = nbr_all[sel]
            has = nb >= 0
            nb_pos = local_all[np.where(has, nb, 0)]
            frames = _vertex_frames(pos, normals[sel], nb_pos, has)
            r_obj = _pure_rotation(np.array(obj_eval.matrix_world, dtype=np.float64)[:3, :3])
            rest = state.get("vrest")
            if rest is None:
                state["vrest"] = rest = frames  # rest frames = first sample
                state["vprev"] = None
                ed = state.get("edges")
                if ed is not None and len(ed):
                    inv = np.empty(kk, dtype=np.int64)
                    inv[perm] = np.arange(kk)    # original index -> position in the logged order
                    ed = inv[ed].astype(np.int32)
                else:
                    ed = np.empty((0, 2), np.int32)
                state["wire"] = {"total": state["total"], "co": local_all[perm].copy(), "edges": ed}
            m = np.matmul(r_obj, np.matmul(frames, np.transpose(rest[ok], (0, 2, 1))))
            out = np.full((kk, NF), np.nan)
            out[ok, :3] = world_all[sel] * unit_scale
            prev = state["vprev"]
            eul = _euler_xyz_batch(m, None if prev is None else prev[ok])
            if prev is None:
                state["vprev"] = prev = np.zeros((kk, 3))
            prev[ok] = eul
            out[ok, 3:] = np.degrees(eul)
            vert_vals = out.reshape(-1)
    else:
        # Objects without geometry (empties, cameras, lights): fall back to the matrix.
        mw = obj_eval.matrix_world
        loc = np.array(mw.translation)
        rot = np.array(mw.to_3x3().normalized())
        state.setdefault("k", 0)
        vert_vals = np.full(state["k"] * NF, np.nan)

    loc = np.asarray(loc, dtype=np.float64) * unit_scale
    euler = Matrix(rot.tolist()).to_euler("XYZ", state.get("prev"))
    state["prev"] = euler
    return (*loc.tolist(), *(math.degrees(a) for a in euler)), vert_vals


# ----------------------------------------------------------------------------
# Film Reel support (reels made by FilmReel_Generator)
# ----------------------------------------------------------------------------
REEL_MARK = "film_reel_addon"      # custom property the generator puts on a reel's strip object


def _reel_frames(ob):
    """Number of frames of a Film Reel strip object, or 0 if ob is not one (also 0 when the
    FilmReel_Generator add-on is not enabled, so the strip is then logged as a plain mesh)."""
    try:
        if ob is None or ob.type != "MESH" or not ob.get(REEL_MARK):
            return 0
        n = int(ob.film_reel_unwind.n_frames)
    except Exception:
        return 0
    return n if n > 0 and ob.data is not None else 0


def _reel_key(name):
    return f"{name} (Reel)"


def _frame_key(name, k, n):
    return f"{name} (Frame {k:0{len(str(max(n, 1)))}d})"


def _group_keys(name):
    """Logged items that belong to the tracked object called name, in logged order."""
    return [k for k in _LOG["data"]
            if k == name or k.startswith(name + " (Reel)") or k.startswith(name + " (Frame ")]


def _build_targets(s):
    """What a recording logs: a list of dicts {key, obj, kind PLAIN/WHOLE/FRAME, k, n, nverts}
    built from the tracked list, and an error text (or None)."""
    targets = []
    for t in s.objects:
        ob = t.obj
        if not ob:
            continue
        n = _reel_frames(ob)
        mode = t.reel_mode if n else "MESH"
        if mode == "MESH":
            targets.append({"key": ob.name, "obj": ob, "kind": "PLAIN",
                            "nverts": len(getattr(ob.data, "vertices", ()) or ())})
            continue
        nv_all, npoly = len(ob.data.vertices), len(ob.data.polygons)
        if nv_all % n or (npoly and npoly % n):
            return [], f"'{ob.name}' does not divide evenly into {n} frames (was it edited?)"
        if mode in ("BOTH", "WHOLE"):
            targets.append({"key": _reel_key(ob.name), "obj": ob, "kind": "WHOLE", "n": n,
                            "nverts": 0})
        if mode in ("BOTH", "FRAMES"):
            first = max(1, t.reel_first)
            last = n if t.reel_last <= 0 else min(t.reel_last, n)
            if first > last:
                return [], f"'{ob.name}': First Frame {first} is after Last Frame {last}"
            for k in range(first - 1, last):
                targets.append({"key": _frame_key(ob.name, k + 1, n), "obj": ob, "kind": "FRAME",
                                "k": k, "n": n, "nverts": nv_all // n})
    return targets, None


# ----------------------------------------------------------------------------
# Public API for other add-ons (see the usage notes at the top of this file)
# ----------------------------------------------------------------------------
API_KEY = "motion_data_logger"
API_VERSION = (1, 4, 0)
FIELDS = ("time", "frame", "loc_x", "loc_y", "loc_z", "rot_x", "rot_y", "rot_z")
UNITS = {"time": "s", "frame": "scene frames", "loc_x": "m", "loc_y": "m", "loc_z": "m",
         "rot_x": "deg", "rot_y": "deg", "rot_z": "deg"}

_SUBSCRIBERS = []
_STATE = {"recording": False, "stop_requested": False, "clear_after": False,
          "last_run": None}   # last_run: {"sec_per_sample", "sig"} of the last recording


def _notify(event, **info):
    """Tell every subscriber what happened. A failing subscriber never breaks recording."""
    for cb in list(_SUBSCRIBERS):
        try:
            cb(event, info)
        except Exception:
            traceback.print_exc()


def _interp(rows, f):
    """Linearly interpolate between rows at fractional row index f (clamped to the log)."""
    f = min(max(f, 0.0), len(rows) - 1)
    i0 = int(math.floor(f))
    i1 = min(i0 + 1, len(rows) - 1)
    w = f - i0
    a = np.asarray(rows[i0], dtype=np.float64)
    b = np.asarray(rows[i1], dtype=np.float64)
    return tuple((a + (b - a) * w).tolist())


class MotionLogAPI:
    """Read-only access to the Motion Data Logger's data. Every value returned is a copy."""

    version = API_VERSION
    fields = FIELDS
    units = dict(UNITS)

    # -- state ---------------------------------------------------------------
    @property
    def interval(self):
        """Seconds between raw samples."""
        return _LOG["interval"]

    @property
    def store_per(self):
        """How the last recording was stored: "SHUTTER" (a sample every `interval` seconds) or
        "FRAME" (one sample per frame at the project's frame rate, interval = 1 / fps)."""
        return _LOG.get("store_per", "SHUTTER")

    @property
    def fps(self):
        """Scene frame rate the last recording was made at (None before any recording)."""
        return _LOG["fps"]

    @property
    def start_frame(self):
        """Scene frame at which the last recording started (None before any recording)."""
        return _LOG["start_frame"]

    @property
    def is_recording(self):
        return _STATE["recording"]

    def object_names(self):
        return list(_LOG["data"])

    def related(self, name):
        """All logged items of a tracked object: itself, or its reel and frame items."""
        return _group_keys(name)

    def reel_key(self, name):
        """Name of the whole-reel item of Film Reel strip `name`, or None if not logged."""
        k = _reel_key(name)
        return k if k in _LOG["data"] else None

    def frame_key(self, name, frame):
        """Name of the item for frame number `frame` (1 = first) of Film Reel strip `name`."""
        for k in _group_keys(name):
            if k.startswith(name + " (Frame ") and k.endswith(")"):
                try:
                    if int(k[len(name) + 8:-1]) == int(frame):
                        return k
                except ValueError:
                    pass
        return None

    def has_data(self, name=None):
        if name is None:
            return any(len(r) for r in _LOG["data"].values())
        return len(_LOG["data"].get(name, ())) > 0

    def sample_count(self, name):
        return len(_LOG["data"].get(name, ()))

    # -- raw data --------------------------------------------------------------
    def get_array(self, name):
        """All samples of one object as a float64 numpy array of shape (N, 8),
        columns in the order of `fields`. Empty (0, 8) array if there is no data."""
        rows = _LOG["data"].get(name)
        if not rows:
            return np.empty((0, len(FIELDS)), dtype=np.float64)
        return np.array(rows, dtype=np.float64)

    def get_all(self):
        """{object_name: array} for every logged object."""
        return {n: self.get_array(n) for n in _LOG["data"]}

    def get_column(self, name, field):
        """One field (see `fields`) of one object as a 1-D numpy array."""
        return self.get_array(name)[:, FIELDS.index(field)]

    def get_sample(self, name, index):
        """One raw sample as a dict keyed by `fields`. Negative indices count from the end."""
        return dict(zip(FIELDS, (float(v) for v in _LOG["data"][name][index])))

    # -- per-vertex data ---------------------------------------------------------
    vertex_fields = VERT_FIELDS

    def get_vertex_count(self, name):
        """Number of vertices logged for this object (all of its vertices)."""
        rows = _LOG["verts"].get(name)
        return len(rows[0]) // NF if rows else 0

    def get_vertex_array(self, name):
        """Per-vertex samples of one object as a float64 array of shape (N, K, 6): sample,
        vertex (0 = vertex 1), field (see `vertex_fields`). K = get_vertex_count(name)."""
        rows = _LOG["verts"].get(name)
        if not rows:
            return np.empty((0, 0, NF), dtype=np.float64)
        return np.array(rows, dtype=np.float64).reshape(len(rows), -1, NF)

    def get_vertex_column(self, name, vertex, field):
        """One field of one vertex (index 0 = vertex 1) as a 1-D numpy array."""
        return self.get_vertex_array(name)[:, vertex, VERT_FIELDS.index(field)]

    def _vertex_dict(self, row):
        a = np.array(row, dtype=np.float64).reshape(-1, NF)
        return {v: dict(zip(VERT_FIELDS, (float(x) for x in a[v]))) for v in range(len(a))}

    def vertex_sample_at_time(self, name, t):
        """{vertex_index: {field: value}} interpolated at t seconds (clamped)."""
        rows = _LOG["verts"].get(name)
        if not rows:
            raise KeyError(f"No logged vertex data for '{name}'")
        return self._vertex_dict(_interp(rows, t / _LOG["interval"]))

    def vertex_sample_at_frame(self, name, frame):
        """Same as vertex_sample_at_time, for a (fractional) scene frame."""
        if _LOG["fps"] is None:
            raise KeyError("Nothing has been recorded yet")
        return self.vertex_sample_at_time(name, (frame - _LOG["start_frame"]) / _LOG["fps"])

    # -- lookups ---------------------------------------------------------------
    def sample_at_time(self, name, t):
        """Interpolated sample at t seconds from the start of the recording (clamped)."""
        rows = _LOG["data"].get(name)
        if not rows:
            raise KeyError(f"No logged data for '{name}'")
        return dict(zip(FIELDS, _interp(rows, t / _LOG["interval"])))

    def sample_at_frame(self, name, frame):
        """Interpolated sample at a (fractional) scene frame (clamped to the recording)."""
        if _LOG["fps"] is None:
            raise KeyError("Nothing has been recorded yet")
        return self.sample_at_time(name, (frame - _LOG["start_frame"]) / _LOG["fps"])

    # -- events ----------------------------------------------------------------
    def subscribe(self, callback):
        """callback(event, info_dict). Events: "started", "progress", "finished",
        "cancelled", "cleared", "loaded". Unsubscribe when your add-on is disabled."""
        if callback not in _SUBSCRIBERS:
            _SUBSCRIBERS.append(callback)

    def unsubscribe(self, callback):
        if callback in _SUBSCRIBERS:
            _SUBSCRIBERS.remove(callback)


API = MotionLogAPI()

# ----------------------------------------------------------------------------
# Saving the log inside the .blend file
# The log is packed (compressed) into a Text datablock, which Blender stores in the .blend.
# ----------------------------------------------------------------------------
TEXT_NAME = "MotionLog_Saved"
_MAGIC = "MOTION_LOG_V1"


def _pack_log():
    names = list(_LOG["data"])
    arrays = {f"a{i}": np.array(_LOG["data"][n], dtype=np.float64) for i, n in enumerate(names)}
    totals = {}
    for i, n in enumerate(names):
        if _LOG["verts"].get(n):
            arrays[f"v{i}"] = np.array(_LOG["verts"][n], dtype=np.float64)
        wm = _LOG["vmeta"].get(n)
        if wm:
            arrays[f"mc{i}"] = np.asarray(wm["co"], dtype=np.float64)
            arrays[f"me{i}"] = np.asarray(wm["edges"], dtype=np.int32).reshape(-1, 2)
            totals[n] = int(wm["total"])
    meta = json.dumps({"names": names, "interval": _LOG["interval"], "fps": _LOG["fps"],
                       "start_frame": _LOG["start_frame"], "fields": FIELDS,
                       "store_per": _LOG.get("store_per", "SHUTTER"),
                       "vtotals": totals})
    buf = io.BytesIO()
    np.savez_compressed(buf, meta=np.array(meta), **arrays)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "\n".join([_MAGIC] + [b64[i:i + 120] for i in range(0, len(b64), 120)])


def _unpack_log(text):
    lines = text.split("\n")
    if not lines or lines[0].strip() != _MAGIC:
        raise ValueError("Not a saved Motion Log")
    raw = base64.b64decode("".join(l.strip() for l in lines[1:]))
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        data = {n: [tuple(r) for r in z[f"a{i}"].tolist()] for i, n in enumerate(meta["names"])}
        verts = {n: list(z[f"v{i}"])
                 for i, n in enumerate(meta["names"]) if f"v{i}" in z.files}  # absent in old saves
        vmeta = {n: {"total": meta.get("vtotals", {}).get(n, len(z[f"mc{i}"])),
                     "co": z[f"mc{i}"], "edges": z[f"me{i}"]}
                 for i, n in enumerate(meta["names"]) if f"mc{i}" in z.files}
    return meta, data, verts, vmeta


def _load_saved(notify=True):
    """Restore the log from the Text datablock. Returns the number of objects restored."""
    txt = bpy.data.texts.get(TEXT_NAME)
    if txt is None:
        raise KeyError("No saved Motion Log in this file")
    meta, data, verts, vmeta = _unpack_log(txt.as_string())
    _reset_icons()
    _LOG["data"] = data
    _LOG["verts"] = verts
    _LOG["vmeta"] = vmeta
    _LOG["interval"] = meta["interval"]
    _LOG["store_per"] = meta.get("store_per", "SHUTTER")  # absent in older saves
    _LOG["fps"] = meta["fps"]
    _LOG["start_frame"] = meta["start_frame"]
    if notify:
        _notify("loaded", objects=list(data))
    return len(data)



CSV_PREFIX = "MotionLog_"   # CSV Texts made by older versions: MotionLog_<object>_<fps>fps.csv


def _purge_project_data():
    """Remove everything this add-on has put into the open .blend, plus its in-memory log:
    the saved-log Text, any leftover CSV Texts from older versions, and the per-scene settings
    (tracked objects, display options). Only called from unregister(), and only when the "Remove project
    data" preference is ticked. Each step is guarded so a failure cannot raise."""
    _STATE.update(stop_requested=True, clear_after=False)  # a running recording stops itself
    try:
        _clear_data()  # also frees the icon previews and their temp folder
        _LOG.update(interval=0.0021, fps=None, start_frame=None, store_per="SHUTTER")
    except Exception:
        traceback.print_exc()
    try:
        for txt in list(bpy.data.texts):
            if txt.name == TEXT_NAME or (txt.name.startswith(CSV_PREFIX)
                                         and txt.name.endswith(".csv")):
                bpy.data.texts.remove(txt)
    except Exception:
        traceback.print_exc()
    try:
        for sc in bpy.data.scenes:
            if "motion_log" in sc.keys():
                del sc["motion_log"]  # the stored values behind Scene.motion_log
    except Exception:
        traceback.print_exc()



def _restore_saved_log():
    """Timer callback run shortly after the add-on is enabled (bpy.data cannot be used inside
    register() itself). Brings back a log that was saved in this .blend with "Save Log in
    .blend" before the add-on was disabled. Does nothing if there is no saved log, if a log is
    already in memory, or if a recording is running."""
    try:
        if _STATE["recording"] or _LOG["data"] or TEXT_NAME not in bpy.data.texts:
            return None
        _load_saved()
        wm = bpy.context.window_manager
        for win in wm.windows:
            for area in win.screen.areas:
                area.tag_redraw()
    except Exception:
        traceback.print_exc()
    return None  # run once


def _install_api():
    bpy.app.driver_namespace[API_KEY] = API


def _reset_log_state():
    """Forget the in-memory log completely (data, icons and the recording settings)."""
    _clear_data()
    _LOG.update(interval=0.0021, fps=None, start_frame=None, store_per="SHUTTER")


@persistent
def _reinstall_api(_dummy=None):
    """Runs after every file load. Keeps the API published, and makes sure the log on screen
    belongs to the file that is now open: the previous file's log is dropped, then the log
    saved inside the new .blend (if there is one) is restored."""
    _install_api()
    _STATE.update(recording=False, stop_requested=False, clear_after=False, last_run=None)
    if _LOG["data"] or _LOG["verts"] or _LOG["vmeta"]:
        try:
            _reset_log_state()
            _notify("cleared")
        except Exception:
            traceback.print_exc()
    if TEXT_NAME in bpy.data.texts:
        try:
            _load_saved()  # bring back a log that was saved inside this .blend
        except Exception:
            traceback.print_exc()


# ----------------------------------------------------------------------------
# Properties
# ----------------------------------------------------------------------------
def _item_search(self, context, edit_text):
    s = bpy.context.scene.motion_log
    if 0 <= s.active_index < len(s.objects) and s.objects[s.active_index].obj:
        return _group_keys(s.objects[s.active_index].obj.name)
    return list(_LOG["data"])


class MLOG_TrackedObject(PropertyGroup):
    obj: PointerProperty(name="Object", type=bpy.types.Object)
    reel_mode: EnumProperty(
        name="Reel Mode",
        items=[("BOTH", "Reel + Frames", "Log the whole reel and every frame, each as its own item"),
               ("FRAMES", "Frames Only", "Log each frame as its own item (location, rotation, vertices)"),
               ("WHOLE", "Whole Reel Only", "Log the whole strip as one item (centre and rotation, no vertex columns)"),
               ("MESH", "As Plain Mesh", "Treat the strip like any mesh: one item with every vertex of every frame")],
        default="BOTH", description="What to log for this Film Reel")
    reel_first: IntProperty(name="First Frame", default=1, min=1,
                            description="First reel frame to log (1 = first frame on the strip)")
    reel_last: IntProperty(name="Last Frame", default=0, min=0,
                           description="Last reel frame to log; 0 = through the last frame")


class MLOG_Settings(PropertyGroup):
    objects: CollectionProperty(type=MLOG_TrackedObject)
    active_index: IntProperty(default=0)

    sample_interval: FloatProperty(
        name="Sample Interval (s)",
        description="Time between logged samples. 0.0021 s = shutter open time at 240 fps / 180 degrees. "
                    "Only used when Store Data is set to Per Shutter",
        default=0.0021, min=0.0001, max=1.0, precision=6, step=1)
    store_per: EnumProperty(
        name="Store Data", items=STORE_ITEMS, default="SHUTTER",
        description="How Log Data stores the motion: per shutter (a sample every Sample Interval "
                    "seconds) or per frame (one sample per frame at the project's frame rate)")
    vertex_stride: IntProperty(
        name="Vertex Stride",
        description="Whole-object summary only: use every Nth vertex (1 = all) to find the "
                    "object's centre and rotation. Individually logged vertices are never skipped",
        default=1, min=1, soft_max=100)
    use_scene_range: BoolProperty(
        name="Use Scene Frame Range", default=True)
    frame_start: IntProperty(name="Start", default=1, min=0)
    frame_end: IntProperty(name="End", default=250, min=0)

    display_fps: EnumProperty(
        name="Display Frame Rate", items=FPS_ITEMS, default="24",
        description="The log shows one row per frame at this frame rate")
    display_per: EnumProperty(
        name="Display Per",
        items=[("FRAME", "Per Frame", "One row for every frame at the display frame rate"),
               ("SHUTTER", "Per Shutter", "One row for every shutter-open time (1 / (2 x display fps) s, "
                                          "a 180 degree shutter), i.e. two rows per frame")],
        default="FRAME",
        description="Whether the log shows one row per frame or one row per shutter time. "
                    "Rows sit exactly on the frame (or shutter) times, so Time and Frame are exact")
    rows_per_page: IntProperty(name="Rows Shown", default=20, min=1)
    log_view: EnumProperty(
        name="Log View",
        items=[("VERTICES", "Vertices", "One group of columns per vertex, with its corner icon above"),
               ("OBJECT", "Object", "Whole-object centre (where the paths between opposite vertices converge) and best-fit rotation")],
        default="VERTICES", description="What the log columns show")
    visible_cols: IntProperty(
        name="Columns Shown", default=6, min=1,
        description="How many data columns are visible at once (the Time column stays pinned)")
    scroll_v: FloatProperty(
        name="Vertical Scroll", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
        precision=3, description="Scroll the log up and down")
    scroll_h: FloatProperty(
        name="Horizontal Scroll", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
        precision=3, description="Scroll the log side to side")
    log_item: StringProperty(
        name="Logged Item", default="", search=_item_search, search_options={"SORT"},
        description="Which logged item of the highlighted object the log shows: the whole reel "
                    "or one of its frames. Empty = the first one")
    show_log: BoolProperty(
        name="Show Log Here", default=False,
        description="Show or hide the log inside this panel")


# ----------------------------------------------------------------------------
# Tracked-object operators
# ----------------------------------------------------------------------------
class MLOG_OT_add_selected(Operator):
    bl_idname = "motion_log.add_selected"
    bl_label = "Add Selected Objects"
    bl_description = "Add all selected objects to the tracked list"

    def execute(self, context):
        s = context.scene.motion_log
        have = {t.obj for t in s.objects if t.obj}
        added = 0
        for ob in context.selected_objects:
            if ob not in have:
                s.objects.add().obj = ob
                added += 1
        self.report({"INFO"}, f"Added {added} object(s)")
        return {"FINISHED"}


class MLOG_OT_remove_active(Operator):
    bl_idname = "motion_log.remove_active"
    bl_label = "Remove"
    bl_description = "Remove the highlighted object from the tracked list"

    def execute(self, context):
        s = context.scene.motion_log
        if 0 <= s.active_index < len(s.objects):
            s.objects.remove(s.active_index)
            s.active_index = max(0, s.active_index - 1)
        return {"FINISHED"}


class MLOG_OT_clear_tracked(Operator):
    bl_idname = "motion_log.clear_tracked"
    bl_label = "Clear List"
    bl_description = "Remove all objects from the tracked list"

    def execute(self, context):
        context.scene.motion_log.objects.clear()
        return {"FINISHED"}


class MLOG_UL_objects(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index):
        if item.obj:
            layout.label(text=item.obj.name,
                         icon="SEQUENCE" if _reel_frames(item.obj) else "OBJECT_DATA")
        else:
            layout.label(text="(missing object)", icon="ERROR")


# ----------------------------------------------------------------------------
# Recording (modal so the UI stays alive and ESC cancels)
# ----------------------------------------------------------------------------
RISKY_BYTES = 1.0e9  # warn when the vertex data of one recording is estimated above ~1 GB


def _estimate_log_size(scene, objs):
    """Rough size of the vertex data a recording would hold: (bytes, vertices, samples).
    Uses each object's base mesh, so modifiers that add geometry can make it larger."""
    s = scene.motion_log
    fps = scene.render.fps / scene.render.fps_base
    start = scene.frame_start if s.use_scene_range else s.frame_start
    end = scene.frame_end if s.use_scene_range else s.frame_end
    interval = _record_interval(scene)
    samples = int(math.ceil(max(end - start, 0) / fps / interval - 1e-9)) + 1
    verts = sum(t["nverts"] for t in _build_targets(s)[0])
    return samples * verts * NF * 8, verts, samples


def _fmt_duration(sec):
    """Seconds as a short human string: 45 s, 3 min 20 s, 1 h 05 min."""
    sec = max(0, int(round(sec)))
    if sec < 60:
        return f"{sec} s"
    if sec < 3600:
        return f"{sec // 60} min {sec % 60:02d} s"
    return f"{sec // 3600} h {(sec % 3600) // 60:02d} min"


def _signature(objs):
    """What a recording's speed mostly depends on: which items and how many vertices.
    objs = the targets of _build_targets."""
    return tuple(sorted((t["key"], t["nverts"]) for t in objs))


def _estimate_seconds(scene, objs):
    """Expected recording time, from the speed of the last recording of the same objects.
    None when there is no comparable earlier run (the live estimate then appears in the
    header while logging)."""
    last = _STATE.get("last_run")
    targets = _build_targets(scene.motion_log)[0]
    if not last or not targets or last["sig"] != _signature(targets):
        return None
    return _estimate_log_size(scene, objs)[2] * last["sec_per_sample"]


TICK_BUDGET = 0.08   # seconds of sampling per timer tick: keeps ESC and the UI responsive


class MLOG_OT_record(Operator):
    bl_idname = "motion_log.record"
    bl_label = "Log Data"
    bl_description = ("Step through the frame range and log the location and rotation of every "
                      "vertex (corner) of every tracked object, either every Sample Interval "
                      "(Store Data = Per Shutter) or once per frame (Store Data = Per Frame). "
                      "Press ESC to cancel")

    _timer = None

    confirmed: BoolProperty(default=False, options={"SKIP_SAVE", "HIDDEN"})

    @classmethod
    def poll(cls, context):
        if _STATE["recording"]:
            cls.poll_message_set("A recording is already running (press ESC to cancel it)")
            return False
        return True

    def draw(self, context):
        """Contents of the large-recording confirmation dialog."""
        scene = context.scene
        est, verts, samples = _estimate_log_size(scene, None)
        col = self.layout.column(align=True)
        col.label(text=f"This recording needs about {est / 1e9:.1f} GB of memory.", icon="ERROR")
        col.label(text=f"{verts:,} vertices x {samples:,} samples x 6 values")
        secs = _estimate_seconds(scene, None)
        if secs is not None:
            col.label(text=f"Expected time: about {_fmt_duration(secs)} (from the last run).")
        col.separator()
        col.label(text="Blender may slow down or run out of memory. To reduce it,")
        if scene.motion_log.store_per == "FRAME":
            col.label(text="shorten the frame range or track fewer objects.")
        else:
            col.label(text="raise the Sample Interval, shorten the frame range, or track fewer objects.")

    def execute(self, context):
        # Reached when the user confirms the dialog: start the real (modal) recording.
        bpy.ops.motion_log.record("INVOKE_DEFAULT", confirmed=True)
        return {"FINISHED"}

    def invoke(self, context, event):
        if _STATE["recording"]:   # also guards calls from Python, which bypass poll
            self.report({"ERROR"}, "A recording is already running")
            return {"CANCELLED"}
        scene = context.scene
        s = scene.motion_log
        self.targets, err = _build_targets(s)
        if err:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}
        if not self.targets:
            self.report({"ERROR"}, "Add at least one object to track first")
            return {"CANCELLED"}
        if not self.confirmed and _estimate_log_size(scene, None)[0] > RISKY_BYTES:
            wm = context.window_manager
            try:
                return wm.invoke_props_dialog(self, width=460, confirm_text="Log Anyway")
            except TypeError:  # older Blender: no confirm_text argument
                return wm.invoke_props_dialog(self, width=460)

        self.fps = scene.render.fps / scene.render.fps_base
        self.start = scene.frame_start if s.use_scene_range else s.frame_start
        end = scene.frame_end if s.use_scene_range else s.frame_end
        if end < self.start:
            self.report({"ERROR"}, "End frame is before start frame")
            return {"CANCELLED"}

        # Per shutter: the Sample Interval the user typed. Per frame: exactly one frame.
        self.interval = _record_interval(scene)
        self.store_per = s.store_per
        self.stride = s.vertex_stride
        self.unit_scale = scene.unit_settings.scale_length  # Blender units -> meters
        duration = (end - self.start) / self.fps
        self.duration = duration
        # enough samples to cover the whole range; the last one is placed exactly on the end frame
        self.total = int(math.ceil(duration / self.interval - 1e-9)) + 1
        self.i = 0
        self.orig_frame = scene.frame_current
        self.shared = {}      # one evaluation of a reel's strip per sample, shared by its items
        self.states = {}
        for tg in self.targets:
            st = {}
            if tg["kind"] != "PLAIN":
                st["shared"] = self.shared.setdefault(
                    tg["obj"].name, {"static": {}, "cur": None, "stamp": None, "full": None})
                if tg["kind"] == "WHOLE":
                    st["whole"] = True
                else:
                    st["frame"] = (tg["k"], tg["n"])
            self.states[tg["key"]] = st
        self.scene = scene            # the recording belongs to this scene
        self.area = context.area      # where the progress text is shown
        self.sig = _signature(self.targets)

        _LOG["interval"] = self.interval
        _LOG["store_per"] = self.store_per
        _LOG["fps"] = self.fps
        _LOG["start_frame"] = self.start
        _LOG["data"] = {tg["key"]: [] for tg in self.targets}
        _LOG["verts"] = {tg["key"]: [] for tg in self.targets}
        _LOG["vmeta"] = {}
        _reset_icons()
        _STATE.update(recording=True, stop_requested=False, clear_after=False)
        _notify("started", objects=[tg["key"] for tg in self.targets], interval=self.interval,
                store_per=self.store_per, total=self.total)

        wm = context.window_manager
        try:
            wm.progress_begin(0, self.total)
            self._timer = wm.event_timer_add(0.01, window=context.window)
            wm.modal_handler_add(self)
        except Exception as e:     # never leave the recording flag stuck on
            traceback.print_exc()
            _STATE["recording"] = False
            self.report({"ERROR"}, f"Could not start the recording: {e}")
            return {"CANCELLED"}
        self.t_start = time.perf_counter()
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC" or (event.type == "TIMER" and _STATE["stop_requested"]):
            return self._finish(context, cancelled=True)
        if event.type != "TIMER":
            return {"PASS_THROUGH"}   # the UI stays usable (Clear Data, scrolling, ...) while logging
        if context.scene is not self.scene:
            return self._finish(context, cancelled=True,
                                note="the active scene changed during the recording")
        try:
            dg = None
            t_tick = time.perf_counter()
            while True:   # as many samples as fit in the tick budget (at least one)
                if self.i >= self.total:
                    return self._finish(context)
                t = min(self.i * self.interval, self.duration)
                frame_f = self.start + t * self.fps
                if abs(frame_f - round(frame_f)) < 1e-9:
                    frame_f = float(round(frame_f))
                whole = math.floor(frame_f)
                self.scene.frame_set(whole, subframe=frame_f - whole)
                dg = context.evaluated_depsgraph_get()
                for sh in self.shared.values():
                    sh["cur"] = self.i          # a new sample: the reel is evaluated afresh
                for tg in self.targets:
                    key = tg["key"]
                    st = self.states[key]
                    vals, vvals = sample_object(tg["obj"], dg, st, self.stride, self.unit_scale)
                    _LOG["data"][key].append((t, frame_f, *vals))
                    _LOG["verts"][key].append(vvals)
                    if "wire" in st and key not in _LOG["vmeta"]:
                        _LOG["vmeta"][key] = st["wire"]
                self.i += 1
                if time.perf_counter() - t_tick >= TICK_BUDGET:
                    break
            context.window_manager.progress_update(self.i)
            _notify("progress", done=self.i, total=self.total)
            elapsed = time.perf_counter() - self.t_start
            left = elapsed / self.i * (self.total - self.i)
            if self.area:
                self.area.header_text_set(
                    f"Motion Log: {self.i}/{self.total} samples | {_fmt_duration(elapsed)} elapsed, "
                    f"about {_fmt_duration(left)} left (ESC to cancel)")
        except Exception as e:
            # Any failure (object deleted, mesh error, ...) ends the recording cleanly and
            # keeps the samples taken so far; nothing stays stuck.
            traceback.print_exc()
            return self._finish(context, cancelled=True, note=f"error: {e}")
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        """Blender calls this when it removes the running modal itself (window closed, file
        loaded): release everything so a new recording can be started."""
        self._finish(context, cancelled=True, note="the recording was interrupted")

    def _finish(self, context, cancelled=False, note=None):
        if getattr(self, "_done", False):
            return {"CANCELLED" if cancelled else "FINISHED"}
        self._done = True
        _STATE["recording"] = False          # first, so nothing below can leave it stuck
        for sh in getattr(self, "shared", {}).values():
            sh["full"] = None                # free the shared reel evaluation
        elapsed = time.perf_counter() - getattr(self, "t_start", time.perf_counter())
        if self.i >= 3:                      # remember the speed for the next time estimate
            _STATE["last_run"] = {"sec_per_sample": elapsed / self.i, "sig": self.sig}
        wm = context.window_manager
        for step in (
            lambda: wm.event_timer_remove(self._timer),
            lambda: wm.progress_end(),
            lambda: self.area and self.area.header_text_set(None),
            lambda: self.scene.frame_set(self.orig_frame),
            lambda: [a.tag_redraw() for a in context.screen.areas],
        ):
            try:
                step()
            except Exception:
                traceback.print_exc()
        msg = (f"Motion log {'cancelled' if cancelled else 'finished'}: {self.i} samples "
               f"in {_fmt_duration(elapsed)}" + (f" ({note})" if note else ""))
        try:
            self.report({"WARNING" if cancelled else "INFO"}, msg)
        except Exception:
            print(msg)
        _notify("cancelled" if cancelled else "finished", samples=self.i)
        if _STATE["clear_after"]:
            _clear_data()
            _notify("cleared")
        _STATE.update(stop_requested=False, clear_after=False)
        return {"CANCELLED" if cancelled else "FINISHED"}


class MLOG_OT_clear_log(Operator):
    bl_idname = "motion_log.clear_log"
    bl_label = "Clear Data"
    bl_description = "Erase all logged data (stops a running recording first)"

    def execute(self, context):
        if _STATE["recording"]:
            # Stop the recording; the recorder erases the data as soon as it has stopped.
            _STATE.update(stop_requested=True, clear_after=True)
            self.report({"INFO"}, "Stopping the recording and clearing the data")
            return {"FINISHED"}
        _clear_data()
        _notify("cleared")
        return {"FINISHED"}


# ----------------------------------------------------------------------------
# Log display (shared by the panel and the popup)
# ----------------------------------------------------------------------------
class _GridRows:
    """Rows of the log display, worked out on demand (so a big log is never copied): row k is
    the logged data interpolated to the exact time of that row. Behaves like a read-only list."""

    def __init__(self, seq, times, interval, as_tuple, frames=None):
        self.seq, self.times, self.interval = seq, times, interval
        self.as_tuple, self.frames = as_tuple, frames

    def __len__(self):
        return len(self.times)

    def __bool__(self):
        return len(self.times) > 0

    def _row(self, k):
        t = self.times[k]
        seq = self.seq
        last = len(seq) - 1
        f = min(max(t / self.interval, 0.0), float(last))
        i0 = int(math.floor(f))
        w = f - i0
        a = np.asarray(seq[i0], dtype=np.float64)
        if w > 1e-12 and i0 < last:
            a = a + (np.asarray(seq[i0 + 1], dtype=np.float64) - a) * w
        if self.as_tuple:
            a = a.copy()
            a[0] = t                                  # exact time of the row
            if self.frames is not None:
                fr = self.frames[0] + t * self.frames[1]
                a[1] = float(round(fr)) if abs(fr - round(fr)) < 1e-9 else fr
            return tuple(a.tolist())
        return a

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self._row(i) for i in range(*k.indices(len(self.times)))]
        if k < 0:
            k += len(self.times)
        if not 0 <= k < len(self.times):
            raise IndexError(k)
        return self._row(k)

    def __iter__(self):
        for k in range(len(self.times)):
            yield self._row(k)


def _display_rows(name, s):
    """The rows of the log display: one per frame at the display fps, or (Display Per = Shutter)
    one per shutter-open time, 1 / (2 x fps) s. Row k sits exactly at k x step seconds, so the
    Time column is exact and, when the display fps matches the scene's, so is the Frame column;
    the values are interpolated between the two logged samples around that moment. The last row
    is the final sample if the range ends on a row. Returns (object rows, vertex rows or None,
    samples per row)."""
    raw = _LOG["data"][name]
    vrows = _LOG["verts"].get(name)
    dfps = float(s.display_fps)
    interval = _LOG["interval"]
    step = (0.5 if s.display_per == "SHUTTER" else 1.0) / dfps
    t_end = raw[-1][0] if raw else 0.0
    n = int(t_end / step + 1e-6) + 1 if raw else 0
    times = [k * step for k in range(n)]
    if times and times[-1] >= t_end - 1e-9:
        times[-1] = t_end
    frames = None
    if _LOG["fps"] is not None and _LOG["start_frame"] is not None:
        frames = (float(_LOG["start_frame"]), float(_LOG["fps"]))
    return (_GridRows(raw, times, interval, True, frames),
            _GridRows(vrows, times, interval, False) if vrows else None,
            step / interval)


def _row_mode_text(s):
    return (f"{int(s.display_fps)} fps, "
            + ("one row per shutter time" if s.display_per == "SHUTTER" else "one row per frame"))


def draw_buttons(layout, popup=False):
    scene = bpy.context.scene
    tracked = _build_targets(scene.motion_log)[0]
    est, verts, samples = _estimate_log_size(scene, tracked) if tracked else (0, 0, 0)
    if est > RISKY_BYTES:
        box = layout.box()
        col = box.column(align=True)
        col.alert = True
        col.label(text=f"Large recording: about {est / 1e9:.1f} GB of vertex data", icon="ERROR")
        col.label(text=f"{verts:,} vertices x {samples:,} samples")
        if scene.motion_log.store_per == "FRAME":
            col.label(text="Shorten the range or track fewer objects to reduce it")
        else:
            col.label(text="Raise Sample Interval or shorten the range to reduce it")
    row = layout.row(align=True)
    row.scale_y = 1.4
    row.operator("motion_log.record", text="Log Data", icon="REC")
    row.operator("motion_log.clear_log", text="Clear Data", icon="TRASH")

    # How the data is stored. Sample Interval only applies to Per Shutter, so say so here.
    s = scene.motion_log
    layout.prop(s, "store_per", expand=True)
    sc = _ui_scale(bpy.context)
    avail = _avail_px(bpy.context, popup)
    if s.store_per == "FRAME":
        fps = scene.render.fps / scene.render.fps_base
        _wrapped_label(layout, f"One sample per frame: every {_fmt_seconds(1.0 / fps)} s at the "
                               f"project's {fps:g} fps. Sample Interval is not used.",
                       avail, sc, icon="INFO")
    else:
        _wrapped_label(layout, f"One sample every {_fmt_seconds(_record_interval(scene))} s "
                               "(the Sample Interval).", avail, sc, icon="INFO")
    if tracked:
        secs = _estimate_seconds(scene, tracked)
        if secs is None:
            _wrapped_label(layout, f"{samples:,} samples to log. The time left is shown in the "
                                   "header while logging.", avail, sc, icon="TIME")
        else:
            _wrapped_label(layout, f"{samples:,} samples, about {_fmt_duration(secs)} "
                                   "(from the last run of these objects).", avail, sc, icon="TIME")


# (column title with unit, number format, index into the combined row, vertex index or None)
# Combined row = 8 object values (time, frame, loc xyz, rot xyz) + 6 values per vertex.
# Time is pinned separately. Vertex v, field j lives at 8 + v * 6 + j.
OBJECT_COLUMNS = (
    ("Frame (scene frames)", ".10f", 1, None),
    ("Loc X (m)", ".10f", 2, None), ("Loc Y (m)", ".10f", 3, None), ("Loc Z (m)", ".10f", 4, None),
    ("Rot X (deg)", ".10f", 5, None), ("Rot Y (deg)", ".10f", 6, None), ("Rot Z (deg)", ".10f", 7, None),
)
_VCOL_CACHE = {}


def _vertex_columns(k):
    if k not in _VCOL_CACHE:
        _VCOL_CACHE[k] = (OBJECT_COLUMNS[0],) + tuple(
            (title, fmt, 8 + v * NF + j, v)
            for v in range(k)
            for j, (title, fmt) in enumerate((
                ("Loc X (m)", ".10f"), ("Loc Y (m)", ".10f"), ("Loc Z (m)", ".10f"),
                ("Rot X (deg)", ".10f"), ("Rot Y (deg)", ".10f"), ("Rot Z (deg)", ".10f"))))
    return _VCOL_CACHE[k]


def _columns(s, k):
    return OBJECT_COLUMNS if s.log_view == "OBJECT" else _vertex_columns(k)


def _cell(r, vr, i):
    """Value of combined column i: 8 object values, then the vertex values."""
    if i < 8:
        return r[i]
    j = i - 8
    return float(vr[j]) if vr is not None and j < len(vr) else float("nan")


def _fmt(value, spec):
    return "-" if value != value else format(value, spec)  # NaN (no such vertex) shows "-"


# ----------------------------------------------------------------------------
# Wireframe icons. Each tracked object gets its own set: its own mesh edges drawn as a small
# wireframe, with the vertex a column belongs to marked in yellow. They are drawn here with
# numpy (no image files to install), saved as PNGs in a temp folder and shown as previews.
# ----------------------------------------------------------------------------
ICON_PX = 96
_SS = 3  # supersampling for smooth lines
_ICONS = {"pcoll": None, "dir": None, "bases": {}, "tried": set(), "count": 0}
_WIRE = (165, 165, 165)
_YELLOW = (255, 214, 0)


def _png_bytes(rgba):
    """Encode an (h, w, 4) uint8 array as a PNG."""
    h, w, _c = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _project(co):
    """Local vertex positions -> icon pixel positions and depth (0 near .. 1 far), seen from
    slightly above and to the side, with mild perspective."""
    c = np.asarray(co, dtype=np.float64)
    c = c - (c.min(axis=0) + c.max(axis=0)) / 2.0
    radius = float(np.max(np.linalg.norm(c, axis=1)))
    c = c / radius if radius > 1e-12 else c
    yaw, pitch = math.radians(35.0), math.radians(25.0)
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    v = c @ rz.T @ rx.T          # camera looks along +Y: x right, z up, y = depth
    scale = 5.0 / (5.0 + v[:, 1])
    px = ICON_PX * (0.5 + 0.40 * v[:, 0] * scale)
    py = ICON_PX * (0.5 - 0.40 * v[:, 2] * scale)
    depth = (v[:, 1] + 1.0) / 2.0
    return px, py, depth


def _render_base(co, edges):
    """Wireframe of the whole object as float RGBA (ICON_PX, ICON_PX, 4), plus the projected
    vertex positions."""
    px, py, depth = _project(co)
    big = ICON_PX * _SS
    canvas = np.zeros((big, big))
    if len(edges) > 2500:
        edges = edges[:: int(math.ceil(len(edges) / 2500))]
    offs = (-1, 0, 1) if len(edges) <= 200 else (0, 1)
    for a, b in edges:
        x0, y0, x1, y1 = px[a] * _SS, py[a] * _SS, px[b] * _SS, py[b] * _SS
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 2
        t = np.linspace(0.0, 1.0, steps)
        xs = np.rint(x0 + (x1 - x0) * t).astype(int)
        ys = np.rint(y0 + (y1 - y0) * t).astype(int)
        alpha = 1.0 - 0.55 * (depth[a] + depth[b]) / 2.0   # far edges fainter
        for dx in offs:
            for dy in offs:
                xi = np.clip(xs + dx, 0, big - 1)
                yi = np.clip(ys + dy, 0, big - 1)
                canvas[yi, xi] = np.maximum(canvas[yi, xi], alpha)
    lines = canvas.reshape(ICON_PX, _SS, ICON_PX, _SS).mean(axis=(1, 3))
    img = np.zeros((ICON_PX, ICON_PX, 4))
    img[..., :3] = _WIRE
    img[..., 3] = np.clip(lines * 1.6, 0.0, 1.0)
    return img, px, py


def _stamp(img, cx, cy, r, color):
    """Alpha-composite an anti-aliased disc onto a float RGBA image."""
    x0, x1 = max(int(cx - r - 2), 0), min(int(cx + r + 3), ICON_PX)
    y0, y1 = max(int(cy - r - 2), 0), min(int(cy + r + 3), ICON_PX)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cov = np.clip(r + 0.5 - np.hypot(xx + 0.5 - cx, yy + 0.5 - cy), 0.0, 1.0)
    win = img[y0:y1, x0:x1]
    a_out = cov + win[..., 3] * (1.0 - cov)
    safe = np.maximum(a_out, 1e-9)[..., None]
    win[..., :3] = (np.array(color) * cov[..., None]
                    + win[..., :3] * (win[..., 3] * (1.0 - cov))[..., None]) / safe
    win[..., 3] = a_out


def _icon_base(name):
    """The object's wireframe (drawn once, reused for every vertex icon of that object)."""
    cache = _ICONS["bases"]
    if name not in cache:
        meta = _LOG["vmeta"].get(name)
        if not meta or not len(meta["co"]):
            cache[name] = None
        else:
            co = np.asarray(meta["co"], dtype=np.float64)
            edges = np.asarray(meta["edges"], dtype=np.int64).reshape(-1, 2)
            base, px, py = _render_base(co, edges)
            if len(co) <= 256:
                for i in range(len(co)):
                    _stamp(base, px[i], py[i], 1.6, _WIRE)
            cache[name] = (base, px, py, _ICONS["count"])
            _ICONS["count"] += 1
    return cache[name]


def _make_icon(name, vertex):
    """Draw and register the icon for one vertex of one object."""
    ent = _icon_base(name)
    if ent is None or not 0 <= vertex < len(ent[1]):
        return
    base, px, py, obj_id = ent
    if _ICONS["pcoll"] is None:
        _ICONS["pcoll"] = bpy.utils.previews.new()
        _ICONS["dir"] = tempfile.mkdtemp(prefix="motion_log_icons_")
    img = base.copy()
    _stamp(img, px[vertex], py[vertex], 6.2, (255, 255, 255))  # thin light ring, readable on any theme
    _stamp(img, px[vertex], py[vertex], 5.0, _YELLOW)
    rgba = np.rint(np.dstack((img[..., :3], img[..., 3] * 255))).clip(0, 255).astype(np.uint8)
    path = os.path.join(_ICONS["dir"], f"o{obj_id}_v{vertex}.png")
    with open(path, "wb") as fh:
        fh.write(_png_bytes(rgba))
    _ICONS["pcoll"].load(f"{name}::{vertex}", path, "IMAGE")


def _reset_icons():
    """Forget all generated icons (they are rebuilt on demand)."""
    pc, folder = _ICONS.get("pcoll"), _ICONS.get("dir")
    if pc is not None:
        bpy.utils.previews.remove(pc)
    if folder:
        shutil.rmtree(folder, ignore_errors=True)
    _ICONS.update(pcoll=None, dir=None, bases={}, tried=set(), count=0)


def _free_icons():
    _reset_icons()


def _icon_id(name, vertex):
    """Icon id for this object's vertex (0-based). Icons are drawn only when a column showing
    them is first displayed, so objects with very many vertices cost nothing up front."""
    key = f"{name}::{vertex}"
    if key not in _ICONS["tried"]:
        _ICONS["tried"].add(key)  # added first so a failure is not retried on every redraw
        try:
            _make_icon(name, vertex)
        except Exception:
            traceback.print_exc()
    pc = _ICONS["pcoll"]
    item = pc.get(key) if pc is not None else None
    return item.icon_id if item is not None else 0


def _displayed_name(s):
    """Name of the logged object the panel is showing: the highlighted entry of the tracked
    list if it has logged data, otherwise the first logged object. None if nothing is logged."""
    data = _LOG["data"]
    if not data:
        return None
    name = next(iter(data))
    if 0 <= s.active_index < len(s.objects) and s.objects[s.active_index].obj:
        cand = s.objects[s.active_index].obj.name
        group = _group_keys(cand)
        if s.log_item in group:
            name = s.log_item
        elif cand in data:
            name = cand
        elif group:
            name = group[0]
    return name


def _log_view(context):
    """Everything the log display and the scroll buttons need, or None if nothing is logged."""
    s = context.scene.motion_log
    data = _LOG["data"]
    if not data:
        return None
    name = _displayed_name(s)
    rows, vrows, every = _display_rows(name, s)
    k = len(vrows[0]) // NF if vrows else 0
    columns = _columns(s, k)
    vis_rows = min(s.rows_per_page, len(rows))
    max_row = max(0, len(rows) - vis_rows)
    vis_cols = min(s.visible_cols, len(columns))
    max_col = len(columns) - vis_cols
    return {
        "s": s, "name": name, "raw": data[name], "rows": rows, "vrows": vrows, "every": every,
        "columns": columns,
        "vis_rows": vis_rows, "max_row": max_row,
        "vis_cols": vis_cols, "max_col": max_col,
        "row_off": round(s.scroll_v * max_row), "col_off": round(s.scroll_h * max_col),
    }


class MLOG_OT_step_item(Operator):
    bl_idname = "motion_log.step_item"
    bl_label = "Previous / Next Item"
    bl_description = "Show the previous / next logged item (reel or frame) of the highlighted object"

    step: IntProperty(default=1, options={"HIDDEN"})

    def execute(self, context):
        s = context.scene.motion_log
        cur = _displayed_name(s)
        if cur is None or not (0 <= s.active_index < len(s.objects)) or not s.objects[s.active_index].obj:
            return {"CANCELLED"}
        group = _group_keys(s.objects[s.active_index].obj.name)
        if cur in group:
            s.log_item = group[(group.index(cur) + self.step) % len(group)]
            s.scroll_v = s.scroll_h = 0.0
        return {"FINISHED"}


class MLOG_OT_scroll(Operator):
    bl_idname = "motion_log.scroll"
    bl_label = "Scroll Log"
    bl_description = "Scroll the log"
    bl_options = {"INTERNAL"}

    direction: EnumProperty(items=[(d, d, "") for d in (
        "UP", "DOWN", "PAGE_UP", "PAGE_DOWN", "LEFT", "RIGHT", "PAGE_LEFT", "PAGE_RIGHT")])

    def execute(self, context):
        v = _log_view(context)
        if v is None:
            return {"CANCELLED"}
        s = v["s"]
        d = self.direction
        if d in {"UP", "DOWN", "PAGE_UP", "PAGE_DOWN"}:
            step = v["vis_rows"] if d.startswith("PAGE") else 1
            off = v["row_off"] + (step if d.endswith("DOWN") else -step)
            off = min(max(off, 0), v["max_row"])
            s.scroll_v = off / v["max_row"] if v["max_row"] else 0.0
        else:
            step = v["vis_cols"] if d.startswith("PAGE") else 1
            off = v["col_off"] + (step if d.endswith("RIGHT") else -step)
            off = min(max(off, 0), v["max_col"])
            s.scroll_h = off / v["max_col"] if v["max_col"] else 0.0
        return {"FINISHED"}


def _scroll_row(layout, label, prop_owner, prop, page_a, step_a, step_b, page_b, icons):
    row = layout.row(align=True)
    row.label(text=label)
    for d, ic in zip((page_a, step_a), icons[:2]):
        row.operator("motion_log.scroll", text="", icon=ic).direction = d
    row.prop(prop_owner, prop, text="", slider=True)
    for d, ic in zip((step_b, page_b), icons[2:]):
        row.operator("motion_log.scroll", text="", icon=ic).direction = d


def _ui_scale(context):
    try:
        return float(context.preferences.view.ui_scale)
    except Exception:
        return 1.0


def _avail_px(context, popup):
    """Width in pixels the log has to work with (the sidebar, or the pop-up)."""
    sc = _ui_scale(context)
    if popup:
        return 1100.0 * sc - 40.0 * sc
    try:
        return max(float(context.region.width) - 36.0 * sc, 120.0)
    except Exception:
        return 400.0 * sc


def _stack(text, fit):
    """Split text into lines of at most `fit` characters, so it can be stacked when its column
    is too narrow: words are packed onto lines; a number with no spaces breaks at its decimal
    point (whole part on top, decimals underneath); anything still too long is cut into pieces."""
    fit = max(int(fit), 1)
    if len(text) <= fit:
        return [text]
    lines = []
    if " " in text:
        cur = ""
        for w in text.split():
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= fit:
                cur += " " + w
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    elif "." in text:
        head, tail = text.split(".", 1)
        lines = [head, "." + tail]
    else:
        lines = [text]
    out = []
    for ln in lines:
        while len(ln) > fit:
            out.append(ln[:fit])
            ln = ln[fit:]
        out.append(ln)
    return out


def _wrap_title(text, cap):
    """Split a column title into lines of at most cap characters, breaking between words
    (a single word longer than cap stays whole)."""
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > cap:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    lines.append(cur)
    return lines


MIN_STACK_FIT = 8      # narrower columns than this are never cut into pieces


def _emit_row(grid, texts, fit, titles=False):
    """Put one table row into a grid. A cell that does not fit its column is stacked onto
    extra lines: titles are wrapped between words (never cut, at least 4 characters wide);
    values are split at the decimal point. When the columns are too narrow to stack
    sensibly (fit < MIN_STACK_FIT) values are left whole and simply clip, instead of
    being chopped into single characters. Every cell of the row gets the
    same number of lines. Returns that number."""
    if titles:
        stacks = [_wrap_title(t, max(int(fit), 4)) if t else [""] for t in texts]
    elif fit < MIN_STACK_FIT:
        stacks = [[t] for t in texts]
    else:
        stacks = [_stack(t, fit) for t in texts]
    n = max(len(x) for x in stacks)
    for i in range(n):
        for st in stacks:
            grid.label(text=st[i] if i < len(st) else "")
    return n


def _wrapped_label(layout, text, avail, sc, icon="NONE"):
    """A label that continues on the next line when the text is wider than the space."""
    for k, ln in enumerate(_stack(text, int(avail / (6.8 * sc)))):
        layout.label(text=ln, icon=icon if k == 0 else "NONE")


def draw_log(layout, context, buttons=False, popup=False):
    if buttons:
        draw_buttons(layout, popup=popup)

    v = _log_view(context)
    if v is None:
        layout.label(text="No data yet. Press Log Data.", icon="INFO")
        return
    s, rows = v["s"], v["rows"]

    sc = _ui_scale(context)
    avail = _avail_px(context, popup)
    box = layout.box().column(align=True)
    _wrapped_label(box, f"{v['name']}  |  {len(v['raw'])} raw samples ({_store_label()}) @ {_fmt_seconds(_LOG['interval'])} s  |  "
                        f"showing {_row_mode_text(s)} ({len(rows)} rows, about every "
                        f"{v['every']:.1f} samples, shutter 1/{2 * int(s.display_fps)} s)", avail, sc)

    if 0 <= s.active_index < len(s.objects) and s.objects[s.active_index].obj and \
            len(_group_keys(s.objects[s.active_index].obj.name)) > 1:
        row = layout.row(align=True)
        row.operator("motion_log.step_item", text="", icon="TRIA_LEFT").step = -1
        row.prop(s, "log_item", text="", icon="SEQUENCE")
        row.operator("motion_log.step_item", text="", icon="TRIA_RIGHT").step = 1
    layout.prop(s, "log_view", expand=True)
    row = layout.row(align=True)
    row.prop(s, "display_fps", text="")
    row.prop(s, "display_per", expand=True)
    row = layout.row(align=True)
    row.prop(s, "rows_per_page")
    row.prop(s, "visible_cols")

    # --- scroll controls: |<< < [slider] > >>| ---
    _scroll_row(layout, "Up / Down", s, "scroll_v",
                "PAGE_UP", "UP", "DOWN", "PAGE_DOWN",
                ("TRIA_UP_BAR", "TRIA_UP", "TRIA_DOWN", "TRIA_DOWN_BAR"))
    _scroll_row(layout, "Left / Right", s, "scroll_h",
                "PAGE_LEFT", "LEFT", "RIGHT", "PAGE_RIGHT",
                ("TRIA_LEFT_BAR", "TRIA_LEFT", "TRIA_RIGHT", "TRIA_RIGHT_BAR"))

    r0, c0 = v["row_off"], v["col_off"]
    columns = v["columns"]
    _wrapped_label(layout, f"Rows {r0 + 1}-{r0 + v['vis_rows']} of {len(rows)}   |   "
                           f"Columns {c0 + 1}-{c0 + v['vis_cols']} of {len(columns)} (Time pinned)",
                   avail, sc)

    cols = columns[c0: c0 + v["vis_cols"]]

    # One grid cell per column (Time + the visible columns), all cells the same width, so the
    # icon, the vertex name, the title and the values of a column always sit in the same place.
    ncols = len(cols) + 1
    has_vertex = any(c[3] is not None for c in cols)
    meta = _LOG["vmeta"].get(v["name"])
    if s.log_view == "VERTICES":
        if len(columns) == 1:
            _wrapped_label(layout, "This object has no vertices to log (empty, camera, light...)",
                           avail, sc, icon="INFO")
        else:
            _wrapped_label(layout, f"{len(columns) // NF} vertices logged: face rows +Z to -Z, "
                                   "round the axis from +X toward +Y, each face closed", avail, sc)

    # Characters that fit in one column; text wider than that is stacked onto extra lines.
    fit = int((avail / ncols - 10.0 * sc) / (7.2 * sc))

    head = layout.grid_flow(row_major=True, columns=ncols, even_columns=True,
                            even_rows=False, align=False)
    if has_vertex:
        head.label(text="")
        for _title, _f, _i, vert in cols:
            if vert is None:
                head.label(text="")
            elif _icon_id(v["name"], vert):
                head.template_icon(icon_value=_icon_id(v["name"], vert), scale=2.2)
            else:
                head.label(text="", icon="VERTEXSEL")
        _emit_row(head, [""] + ["" if c[3] is None else f"Vertex {c[3] + 1}" for c in cols], fit, titles=True)
    _emit_row(head, ["Time (s)"] + [c[0] for c in cols], fit, titles=True)

    vrows = v["vrows"]
    grid = layout.grid_flow(row_major=True, columns=ncols, even_columns=True,
                            even_rows=False, align=True)
    for k in range(r0, min(r0 + v["vis_rows"], len(rows))):
        r = rows[k]
        vr = vrows[k] if vrows else None
        lines = _emit_row(grid, [f"{r[0]:.10f}"] + [_fmt(_cell(r, vr, i), f) for _t, f, i, _v in cols],
                          fit)
        if lines > 1:                      # a blank line keeps stacked rows apart
            for _ in range(ncols):
                grid.label(text="")


class MLOG_OT_show_log(Operator):
    bl_idname = "motion_log.show_log"
    bl_label = "Pop Up Log"
    bl_description = "Pop the log up (move the mouse away to close it)"

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=1100)

    def draw(self, context):
        draw_log(self.layout, context, buttons=True, popup=True)

    def execute(self, context):
        return {"FINISHED"}



class MLOG_OT_save_to_blend(Operator):
    bl_idname = "motion_log.save_to_blend"
    bl_label = "Save Log in .blend"
    bl_description = ("Store the full-resolution log inside this .blend file (as a Text datablock "
                      "named '" + TEXT_NAME + "'). Only available once the .blend has been saved "
                      "to disk. Save the .blend afterwards to keep it")

    @classmethod
    def poll(cls, context):
        if not bpy.data.filepath:  # new, never-saved project
            cls.poll_message_set("Save the .blend file first: the log is not stored in an "
                                 "unsaved project")
            return False
        return True

    def execute(self, context):
        if not bpy.data.filepath:  # also guards calls from Python, which bypass poll
            self.report({"ERROR"}, "Save the .blend file first; the log is not stored in an "
                                   "unsaved project")
            return {"CANCELLED"}
        if _STATE["recording"]:
            self.report({"ERROR"}, "Cannot save while logging")
            return {"CANCELLED"}
        if not _LOG["data"]:
            self.report({"ERROR"}, "Nothing logged yet")
            return {"CANCELLED"}
        txt = bpy.data.texts.get(TEXT_NAME) or bpy.data.texts.new(TEXT_NAME)
        txt.use_fake_user = True  # make sure Blender keeps it in the file
        txt.clear()
        txt.write(_pack_log())
        self.report({"INFO"}, f"Log stored in Text '{TEXT_NAME}'. Save the .blend to keep it")
        return {"FINISHED"}


class MLOG_OT_load_from_blend(Operator):
    bl_idname = "motion_log.load_from_blend"
    bl_label = "Load Log from .blend"
    bl_description = "Replace the current log with the one stored in this .blend file"

    def execute(self, context):
        if _STATE["recording"]:
            self.report({"ERROR"}, "Cannot load while logging")
            return {"CANCELLED"}
        try:
            n = _load_saved()
        except Exception as e:
            self.report({"ERROR"}, f"Could not load: {e}")
            return {"CANCELLED"}
        for area in context.screen.areas:
            area.tag_redraw()
        self.report({"INFO"}, f"Loaded log for {n} object(s)")
        return {"FINISHED"}


def _export_table(name, s, max_width=110):
    """The log of one object as aligned plain text, laid out like the log in the panel: the
    object's name and a summary line on top, then rows of group names (Object / Vertex n,
    repeated over each of its columns), column titles and one row per displayed frame.
    A text file has no font size, so a table wider than max_width characters is split into
    blocks of columns stacked one under the other; Time and Frame are repeated in every block
    and a vertex's six columns are never split."""
    rows, vrows, _ = _display_rows(name, s)
    kv = len(vrows[0]) // NF if vrows else 0
    cols = [("", "Time (s)", ".10f", 0), ("", "Frame", OBJECT_COLUMNS[0][1], 1)]
    cols += [("Object", t, f, i) for t, f, i, _v in OBJECT_COLUMNS[1:]]
    cols += [(f"Vertex {v + 1}", t, f, i) for t, f, i, v in _vertex_columns(kv)[1:]]

    body = []
    for k, r in enumerate(rows):
        vr = vrows[k] if vrows else None
        body.append([_fmt(_cell(r, vr, i), f) for _g, _t, f, i in cols])
    groups = [c[0] for c in cols]
    titles = [c[1] for c in cols]
    widths = [max(len(g), len(t), *(len(x) for x in col))
              for g, t, col in zip(groups, titles, zip(*body) if body else [()] * len(cols))]
    gap = "  "

    # Column blocks: Time and Frame (0, 1) pinned, then whole Object / Vertex n groups.
    grouped = []
    for i in range(2, len(cols)):
        if grouped and groups[grouped[-1][0]] == groups[i]:
            grouped[-1].append(i)
        else:
            grouped.append([i])
    pinned = [0, 1]
    pinned_w = sum(widths[i] for i in pinned) + len(gap) * (len(pinned) - 1)
    blocks, cur, cur_w = [], [], pinned_w
    for g in grouped:
        gw = sum(widths[i] for i in g) + len(gap) * len(g)  # incl. the gap before the group
        if cur and cur_w + gw > max_width:
            blocks.append(cur)
            cur, cur_w = [], pinned_w
        cur.extend(g)
        cur_w += gw
    if cur or not blocks:
        blocks.append(cur)

    def line(cells, idx):
        return gap.join(cells[i].ljust(widths[i]) for i in idx).rstrip()

    head = [name,
            f"{len(_LOG['data'][name])} raw samples ({_store_label()}) @ {_fmt_seconds(_LOG['interval'])} s  |  showing "
            f"{_row_mode_text(s)} ({len(rows)} rows)  |  Frame = scene frames"]
    if kv:
        head.append(f"{kv} vertices, ordered by face: rows +Z to -Z, round the axis from +X toward +Y, each face closed")
    if len(blocks) > 1:
        head.append(f"Columns are split into {len(blocks)} blocks; Time and Frame repeat in each")
    out = list(head)
    for blk in blocks:
        idx = pinned + blk
        out.append("")
        out.append(line(groups, idx))
        out.append(line(titles, idx))
        out.extend(line(b, idx) for b in body)
    return "\n".join(out)


class MLOG_OT_export_log(Operator, ExportHelper):
    bl_idname = "motion_log.export_log"
    bl_label = "Export"   # the file browser's confirm button is named after this
    bl_description = ("Choose where to save the log, at the chosen display frame rate, as a "
                      "text document: one table per object, titled with the object's name")

    filename_ext = ".txt"
    filter_glob: StringProperty(default="*.txt", options={"HIDDEN"})
    active_only: BoolProperty(
        name="Only the Highlighted Object",
        description=("Export just the object highlighted in the tracked list (the one the log "
                     "display is showing) instead of every logged object"),
        default=False)
    line_width: IntProperty(
        name="Line Width (characters)",
        description=("A text file has no font size. Tables wider than this many characters are "
                     "split into column blocks stacked under each other so every line fits "
                     "the window. Lower it for a narrower window"),
        default=110, min=40, max=2000)

    def invoke(self, context, event):
        if not _LOG["data"]:
            self.report({"ERROR"}, "Nothing logged yet")
            return {"CANCELLED"}
        if not self.filepath:
            name = (f"MotionLog_{context.scene.motion_log.display_fps}fps_"
                    f"{'shutter' if context.scene.motion_log.display_per == 'SHUTTER' else 'frame'}"
                    f"{self.filename_ext}")
            folder = os.path.dirname(bpy.data.filepath)  # "" for an unsaved project
            self.filepath = os.path.join(folder, name) if folder else name
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        s = context.scene.motion_log
        if not _LOG["data"]:
            self.report({"ERROR"}, "Nothing logged yet")
            return {"CANCELLED"}
        names = [_displayed_name(s)] if self.active_only else list(_LOG["data"])
        sections = [_export_table(name, s, self.line_width) for name in names]
        try:
            with open(bpy.path.abspath(self.filepath), "w", encoding="utf-8") as f:
                f.write("\n\n\n".join(sections) + "\n")
        except OSError as e:
            self.report({"ERROR"}, f"Could not write the file: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Log exported to {self.filepath}")
        return {"FINISHED"}


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
class MLOG_PT_main(Panel):
    bl_label = "Motion Log"
    bl_idname = "MLOG_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Motion Log"

    def draw(self, context):
        layout = self.layout
        s = context.scene.motion_log

        layout.label(text="Tracked Objects")
        row = layout.row()
        row.template_list("MLOG_UL_objects", "", s, "objects", s, "active_index", rows=4)
        col = row.column(align=True)
        col.operator("motion_log.add_selected", text="", icon="ADD")
        col.operator("motion_log.remove_active", text="", icon="REMOVE")
        col.operator("motion_log.clear_tracked", text="", icon="TRASH")
        if 0 <= s.active_index < len(s.objects) and s.objects[s.active_index].obj:
            it = s.objects[s.active_index]
            n = _reel_frames(it.obj)
            if n:
                box = layout.box()
                box.label(text=f"Film Reel: {n} frames", icon="SEQUENCE")
                box.prop(it, "reel_mode", text="")
                if it.reel_mode in ("BOTH", "FRAMES"):
                    r = box.row(align=True)
                    r.prop(it, "reel_first")
                    r.prop(it, "reel_last")
                    first = max(1, it.reel_first)
                    last = n if it.reel_last <= 0 else min(it.reel_last, n)
                    box.label(text=f"{max(0, last - first + 1)} frame item(s); Last Frame 0 = to the end")

        layout.separator()
        col = layout.column(align=True)
        r = col.row(align=True)
        r.active = s.store_per == "SHUTTER"   # Sample Interval only applies to Per Shutter
        if s.store_per == "FRAME":
            r.prop(s, "sample_interval", text="Sample Interval (s) (per shutter only)")
        else:
            r.prop(s, "sample_interval")
        col.prop(s, "vertex_stride")
        col.prop(s, "use_scene_range")
        if not s.use_scene_range:
            r = col.row(align=True)
            r.prop(s, "frame_start")
            r.prop(s, "frame_end")

        draw_buttons(layout)

        layout.separator()
        layout.prop(s, "display_fps")
        layout.prop(s, "display_per", expand=True)
        row = layout.row(align=True)
        row.operator("motion_log.show_log", icon="WINDOW")
        row.prop(s, "show_log", text="", icon="HIDE_OFF" if s.show_log else "HIDE_ON", toggle=True)
        layout.operator("motion_log.export_log", text="Export Log to Text", icon="FILE_TEXT")
        row = layout.row(align=True)
        row.operator("motion_log.save_to_blend", icon="FILE_BLEND")
        row.operator("motion_log.load_from_blend", icon="FILE_REFRESH")

        if s.show_log:
            draw_log(layout, context)


ADDON_ID = __package__ or __name__   # the name Blender files this add-on's preferences under


class MLOG_Preferences(AddonPreferences):
    bl_idname = ADDON_ID

    remove_data_on_uninstall: BoolProperty(
        name="Remove project data when this add-on is removed",
        description=("When ticked, disabling or uninstalling the add-on also deletes everything it "
                     "stored in the open .blend: the saved log, any leftover CSV Texts from older "
                     "versions and the panel settings. Leave it off to keep that data when the add-on is only disabled"),
        default=False)

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "remove_data_on_uninstall")
        col.label(text="Tick this BEFORE uninstalling. While it is ticked, disabling the "
                       "add-on (or Reload Scripts) also removes the data.", icon="INFO")


def _remove_data_requested():
    try:
        return bool(bpy.context.preferences.addons[ADDON_ID].preferences.remove_data_on_uninstall)
    except Exception:
        return False


classes = (
    MLOG_Preferences, MLOG_TrackedObject, MLOG_Settings, MLOG_OT_add_selected, MLOG_OT_remove_active,
    MLOG_OT_clear_tracked, MLOG_UL_objects, MLOG_OT_record, MLOG_OT_clear_log,
    MLOG_OT_step_item, MLOG_OT_scroll, MLOG_OT_show_log, MLOG_OT_save_to_blend, MLOG_OT_load_from_blend,
    MLOG_OT_export_log, MLOG_PT_main,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.motion_log = PointerProperty(type=MLOG_Settings)
    _install_api()
    if _reinstall_api not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_reinstall_api)
    bpy.app.timers.register(_restore_saved_log, first_interval=0.5)


def unregister():
    if bpy.app.timers.is_registered(_restore_saved_log):
        bpy.app.timers.unregister(_restore_saved_log)
    if _reinstall_api in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_reinstall_api)
    if bpy.app.driver_namespace.get(API_KEY) is API:
        del bpy.app.driver_namespace[API_KEY]
    _SUBSCRIBERS.clear()
    if _remove_data_requested():
        _purge_project_data()
    _free_icons()
    del bpy.types.Scene.motion_log
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
