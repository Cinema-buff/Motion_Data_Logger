bl_info = {
    "name": "Motion Data Logger",
    "author": "Claude",
    "version": (1, 12, 1),
    "blender": (3, 6, 0),
    "location": "3D Viewport > Sidebar (N) > Motion Log",
    "description": "Logs location and rotation of every vertex (corner) of any number of "
                   "objects every 0.0021 s, with a wireframe icon of the object above each vertex "
                   "column, a log display driven by the chosen frame rate and a public API "
                   "other add-ons can read",
    "category": "Animation",
}

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

Units: time = seconds, frame = scene frames, loc = meters, rot = XYZ Euler degrees.
Vertices are numbered in a fixed order, taken from the world-space positions at the first
sample: the first vertex is the top one (highest Z; ties: highest X, then closest to Y = 0).
Each face is walked from its top corner to the closest corner towards -Z, then along the row
towards +Y, then back up towards +Z, and closes on its first corner (triangles and quads
alike). The faces follow each other along a row, then the next row down, until every vertex
is numbered; a vertex shared by several faces is numbered where it is first reached.
Vertices that belong to no face come last.
Every object logs ALL of its own vertices (no limit) and gets its own wireframe icon with that vertex marked in yellow.
All values handed out are copies, so other add-ons cannot corrupt the log.
"""

import base64
import io
import json
import math
import os
import shutil
import struct
import tempfile
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
#   l*           location  = point where the paths from each vertex to the vertex directly
#                across from it meet (the object's centre), in meters
#   r*           rotation  = best-fit (Kabsch) rotation of the vertices, XYZ Euler, degrees
# ----------------------------------------------------------------------------
# _LOG["verts"] = { object_name: [ vrow, vrow, ... ] }  (same length/order as "data")
# vrow = float64 array of K vertices x (lx, ly, lz, rx, ry, rz) flattened to 6*K values, where
#        K depends on the object (its own vertex count, no limit; 0 for objects
#        without geometry). NaN where a vertex disappears during the recording.
#        Vertex n = n-th vertex in face order: top vertex first, each face walked top corner ->
#        down (-Z) -> along the row (+Y) -> up (+Z) -> closed, faces along a row and then row by
#        row downwards (see _vertex_order), using world
#        positions at the first sample.
# _LOG["vmeta"] = { object_name: {"total": mesh vertex count, "co": (K,3) rest positions,
#                                 "edges": (M,2) edges between logged vertices} }
#        (what the wireframe icons are drawn from)
_LOG = {"interval": 0.0021, "data": {}, "verts": {}, "vmeta": {}, "fps": None,
        "start_frame": None}

VERT_FIELDS = ("loc_x", "loc_y", "loc_z", "rot_x", "rot_y", "rot_z")
NF = len(VERT_FIELDS)


def _clear_data():
    _LOG["data"] = {}
    _LOG["verts"] = {}
    _LOG["vmeta"] = {}
    _reset_icons()


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
def _eval_vertices(obj_eval, state):
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
    if "k" not in state:
        state["k"] = n
        state["total"] = n
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


def _legacy_vertex_order(world):
    """Order by position alone: from -Y to +Y; vertices on the same Y (a row) from +Z to -Z;
    ties from -X to +X. Positions are rounded so tiny float noise does not split a row."""
    ext = float(np.ptp(world, axis=0).max())
    tol = max(ext * 1e-6, 1e-12)
    x, y, z = (np.round(world[:, i] / tol) for i in range(3))
    return np.lexsort((x, -z, y))  # last key is the primary one


def _narrow(items, key, tol, smallest=True):
    """Keep the items whose key is (within tol of) the smallest / largest one."""
    vals = [key(it) for it in items]
    best = min(vals) if smallest else max(vals)
    return [it for it, v in zip(items, vals) if abs(v - best) <= tol]


def _vertex_order(world, faces=None):
    """Order in which vertices are logged, from the world positions at the first sample.

    1. First vertex: the highest Z (ties: highest X, then closest to Y = 0, then lowest Y).
    2. Each face is walked A -> B -> C (-> D ...) and closes back on A:
         A = the face's top corner (highest Z; with a flat top edge, the end of that edge so
             that the walk first heads down),
         B = the next corner towards -Z, C = the next one along the row (+Y side), D = back up
             (+Z), then the first corner again.  A triangle is A, B, C and closes on A.
       Which way round the face is walked is fixed by the very first face: from the first
       vertex the closest vertex towards -Z (ties: lowest Y, highest X) is B and the walk
       goes on to the neighbour with the higher Y (ties: higher X).
    3. Faces are taken in rows, top row (+Z) first; a row = faces whose top corner has (about)
       the same Z.  Inside a row the faces follow each other around the object (seen from
       +Z, from the first face on, in the direction of the walk); flat faces such as caps
       come after the other faces of their row.
    4. A vertex shared by several faces is logged where it is first reached.  Vertices that
       belong to no face come last, ordered by position (-Y to +Y, +Z to -Z, -X to +X).
    Without faces the order is by position only.
    Returns the original vertex index for each logged position."""
    legacy = _legacy_vertex_order(world)
    if faces is None:
        return legacy
    loop_vert, start = faces
    n, nf = len(world), len(start)
    if nf == 0 or len(loop_vert) == 0:
        return legacy
    ext = float(np.ptp(world, axis=0).max())
    tol = max(ext * 1e-5, 1e-12)          # "same value" tolerance
    row_tol = max(ext * 1e-4, 1e-12)      # faces closer than this in Z are in the same row
    rtol = max(ext * 1e-3, 1e-12)         # closer than this to the centre axis = on the axis
    atol = 1e-5                           # "same angle" tolerance (radians)
    two_pi = 2.0 * math.pi

    wx, wy, wz = (world[:, i].tolist() for i in range(3))
    lvs = loop_vert.tolist()
    counts = np.diff(np.append(start, len(loop_vert))).tolist()
    starts = start.tolist()

    # ---- the first vertex -----------------------------------------------------------
    used = np.zeros(n, dtype=bool)
    used[loop_vert] = True
    cand = np.flatnonzero(used)
    cz = world[cand, 2]
    cand = cand[cz >= cz.max() - tol]
    cx = world[cand, 0]
    cand = cand[cx >= cx.max() - tol]
    cy = world[cand, 1]
    cand = cand[np.abs(cy) <= np.abs(cy).min() + tol]
    a0 = int(cand[int(np.argmin(world[cand, 1]))])

    # ---- walk direction, from the first face ----------------------------------------
    h = 1
    b0 = None
    opts = []
    for p in np.flatnonzero(loop_vert == a0).tolist():
        f = int(np.searchsorted(start, p, side="right")) - 1
        st, m = starts[f], counts[f]
        k = p - st
        for d in (1, -1):
            b, c = lvs[st + (k + d) % m], lvs[st + (k + 2 * d) % m]
            if wz[b] < wz[a0] - tol:                      # B must lie towards -Z
                opts.append((b, c, d))
    if opts:
        dist = lambda o: math.dist((wx[a0], wy[a0], wz[a0]), (wx[o[0]], wy[o[0]], wz[o[0]]))
        opts = _narrow(opts, dist, tol)                    # closest B
        opts = _narrow(opts, lambda o: wy[o[0]], tol)      # lowest Y
        opts = _narrow(opts, lambda o: wx[o[0]], tol, smallest=False)   # highest X
        opts = _narrow(opts, lambda o: wy[o[1]], tol, smallest=False)   # C towards +Y
        opts = _narrow(opts, lambda o: wx[o[1]], tol, smallest=False)
        b0, _, h = opts[0]

    # ---- row axis and reference angle -------------------------------------------------
    ux, uy = world[used, 0], world[used, 1]
    ax, ay = (ux.max() + ux.min()) / 2.0, (uy.max() + uy.min()) / 2.0

    def polar(x, y):                      # angle seen from +Z, 0 at -Y, growing towards +X
        return math.atan2(x - ax, -(y - ay)) % two_pi, math.hypot(x - ax, y - ay)

    ang, rad = polar(wx[a0], wy[a0])
    if rad <= rtol and b0 is not None:
        ang, rad = polar(wx[b0], wy[b0])
    phi0 = ang if rad > rtol else 0.0

    def offset(a):                        # how far round from the reference angle, 0 .. 2*pi
        o = (a - phi0) % two_pi
        return o - two_pi if o > two_pi - atol else o

    # ---- per face: start corner, row height, position round the object ---------------
    first = np.zeros(nf, dtype=np.int64)   # position (inside the face) of corner A
    rowz = np.zeros(nf)
    flat = np.zeros(nf, dtype=np.int8)
    foff = np.zeros(nf)
    coff = np.zeros(nf)
    for f in range(nf):
        st, m = starts[f], counts[f]
        lv = lvs[st: st + m]
        zs = [wz[v] for v in lv]
        zt = max(zs)
        top = [i for i in range(m) if zs[i] >= zt - tol]
        pick = top
        if len(top) > 1:
            ends = [i for i in top if zs[(i - h) % m] >= zt - tol and zs[(i + h) % m] < zt - tol]
            if ends:
                pick = ends
        if len(pick) > 1:                  # flat face / odd shape: highest X, closest Y = 0
            pick = _narrow(pick, lambda i: wx[lv[i]], tol, smallest=False)
            pick = _narrow(pick, lambda i: abs(wy[lv[i]]), tol)
            pick = [min(pick, key=lambda i: wy[lv[i]])]
        s = pick[0]
        first[f], rowz[f], flat[f] = s, zt, 1 if len(top) == m else 0
        a = lv[s]
        ang, rad = polar(wx[a], wy[a])
        mx = sum(wx[v] for v in lv) / m
        my = sum(wy[v] for v in lv) / m
        cang, _ = polar(mx, my)
        coff[f] = offset(cang)
        foff[f] = offset(ang) if rad > rtol else coff[f]

    # ---- rows (top first), then round each row ------------------------------------------
    by_z = np.argsort(-rowz, kind="stable")
    zs_sorted = rowz[by_z]
    row_sorted = np.concatenate(([0], np.cumsum((zs_sorted[:-1] - zs_sorted[1:]) > row_tol)))
    row = np.empty(nf, dtype=np.int64)
    row[by_z] = row_sorted
    face_order = np.lexsort((np.arange(nf), np.round(coff / atol), np.round(foff / atol),
                             flat, row))

    seen = np.zeros(n, dtype=bool)
    out = []
    for f in face_order.tolist():
        st, m = starts[f], counts[f]
        s = int(first[f])
        for i in range(m):
            v = lvs[st + (s + h * i) % m]
            if not seen[v]:
                seen[v] = True
                out.append(v)
    rest = legacy[~seen[legacy]]
    return np.concatenate((np.array(out, dtype=np.int64), rest.astype(np.int64)))


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


def _converge(a, b):
    """Point where the straight paths a[i] -> b[i] come closest to all meeting (least squares).
    Paths that all cross in one point (opposite vertices of a sphere, a cube, ...) give exactly
    that point; for a slightly irregular object it is the best compromise. If the paths do not
    pin the point down in some direction (flat or one-dimensional objects) that direction
    takes the average of the path midpoints."""
    d = b - a
    ln = np.linalg.norm(d, axis=1)
    ok = ln > 1e-12
    mid = ((a + b) * 0.5)[ok].mean(axis=0) if ok.any() else a.mean(axis=0)
    if not ok.any():
        return mid
    a, d = a[ok], d[ok] / ln[ok][:, None]
    # sum of (I - d d^T) over all paths, written out without a Python loop
    m = len(a) * np.eye(3) - d.T @ d
    rhs = (a - d * np.einsum("ij,ij->i", a, d)[:, None]).sum(axis=0)
    eps = 1e-9 * len(a)
    return np.linalg.solve(m + eps * np.eye(3), rhs + eps * mid)


def _opposite_pairs(pos, max_sources=4000):
    """Pairs (i, j) of vertices that lie directly across from each other. A first estimate of
    the centre comes from pairing every vertex with the vertex farthest from it; then each
    vertex is paired with the vertex nearest to its mirror image through the centre, and the
    centre is moved to where those paths meet (repeated a few times). Worked out once from the
    first sample; the pairs then stay fixed while the object moves.
    Returns an (M, 2) int array, empty if there is nothing to pair."""
    n = len(pos)
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    src = np.arange(n) if n <= max_sources else np.unique(
        np.linspace(0, n - 1, max_sources).astype(np.int64))
    chunk = max(1, int(2_000_000 // n))
    sq = (pos * pos).sum(axis=1)

    def partners(point_of, farthest):
        tgt = np.empty(len(src), dtype=np.int64)
        for s in range(0, len(src), chunk):
            idx = src[s: s + chunk]
            q = point_of(idx)                                             # (k, 3)
            d2 = (q * q).sum(axis=1)[:, None] - 2.0 * q @ pos.T + sq[None, :]
            tgt[s: s + chunk] = np.argmax(d2, axis=1) if farthest else np.argmin(d2, axis=1)
        keep = tgt != src
        if not keep.any():
            return None
        return np.unique(np.sort(np.stack((src[keep], tgt[keep]), axis=1), axis=1), axis=0)

    pairs = partners(lambda idx: pos[idx], True)
    if pairs is None:
        return np.empty((0, 2), dtype=np.int64)
    c = _converge(pos[pairs[:, 0]], pos[pairs[:, 1]])
    scale = max(1.0, float(np.ptp(pos, axis=0).max()))
    for _ in range(4):
        cc = c
        new = partners(lambda idx: 2.0 * cc - pos[idx], False)
        if new is None:
            break
        pairs = new
        c = _converge(pos[pairs[:, 0]], pos[pairs[:, 1]])
        if np.allclose(c, cc, atol=1e-9 * scale):
            break
    return pairs


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
        # Object location = the point where the paths from each vertex to the vertex directly
        # across from it meet (pairs fixed at the first sample).
        cp = state.get("cpairs")
        if cp is None or cp[0] != len(local):
            state["cpairs"] = cp = (len(local), _opposite_pairs(local))
        pr = cp[1]
        loc = _converge(world[pr[:, 0]], world[pr[:, 1]]) if len(pr) else world.mean(axis=0)
        ref = state.get("ref")
        if ref is None or len(ref) != len(local):
            state["ref"] = ref = local.copy()
        rot = _kabsch(ref, world) if len(local) >= 3 else np.array(
            obj_eval.matrix_world.to_3x3().normalized())

        # --- per-vertex location and rotation (every logged vertex, in the fixed order) ---
        kk = state["k"]
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
# Public API for other add-ons (see the usage notes at the top of this file)
# ----------------------------------------------------------------------------
API_KEY = "motion_data_logger"
API_VERSION = (1, 2, 0)
FIELDS = ("time", "frame", "loc_x", "loc_y", "loc_z", "rot_x", "rot_y", "rot_z")
UNITS = {"time": "s", "frame": "scene frames", "loc_x": "m", "loc_y": "m", "loc_z": "m",
         "rot_x": "deg", "rot_y": "deg", "rot_z": "deg"}

_SUBSCRIBERS = []
_STATE = {"recording": False, "stop_requested": False, "clear_after": False}


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
    _LOG["fps"] = meta["fps"]
    _LOG["start_frame"] = meta["start_frame"]
    if notify:
        _notify("loaded", objects=list(data))
    return len(data)



CSV_PREFIX = "MotionLog_"   # CSV Texts made by older versions: MotionLog_<object>_<fps>fps.csv


def _purge_project_data():
    """Remove everything this add-on has put into the open .blend, plus its in-memory log:
    the saved-log Text, exported CSV Texts, and the per-scene settings (tracked objects,
    display options). Only called from unregister(), and only when the "Remove project
    data" preference is ticked. Each step is guarded so a failure cannot raise."""
    _STATE.update(stop_requested=True, clear_after=False)  # a running recording stops itself
    try:
        _clear_data()  # also frees the icon previews and their temp folder
        _LOG.update(interval=0.0021, fps=None, start_frame=None)
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


@persistent
def _reinstall_api(_dummy=None):
    _install_api()  # keeps the API published after a file is opened
    _STATE.update(recording=False, stop_requested=False, clear_after=False)
    if not _LOG["data"] and not _STATE["recording"] and TEXT_NAME in bpy.data.texts:
        try:
            _load_saved()  # bring back a log that was saved inside this .blend
        except Exception:
            traceback.print_exc()


# ----------------------------------------------------------------------------
# Properties
# ----------------------------------------------------------------------------
class MLOG_TrackedObject(PropertyGroup):
    obj: PointerProperty(name="Object", type=bpy.types.Object)


class MLOG_Settings(PropertyGroup):
    objects: CollectionProperty(type=MLOG_TrackedObject)
    active_index: IntProperty(default=0)

    sample_interval: FloatProperty(
        name="Sample Interval (s)",
        description="Time between logged samples. 0.0021 s = shutter open time at 240 fps / 180 degrees",
        default=0.0021, min=0.0001, max=1.0, precision=6, step=1)
    vertex_stride: IntProperty(
        name="Vertex Stride",
        description="Whole-object summary only: use every Nth vertex (1 = all). The eight "
                    "individually logged vertices are never skipped",
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
               ("OBJECT", "Object", "Whole-object centre point (where opposite-vertex paths meet) and best-fit rotation")],
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

    def execute(self, context):
        context.scene.motion_log.objects.clear()
        return {"FINISHED"}


class MLOG_UL_objects(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index):
        if item.obj:
            layout.label(text=item.obj.name, icon="OBJECT_DATA")
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
    interval = float(str(np.float32(s.sample_interval)))
    samples = int(math.ceil(max(end - start, 0) / fps / interval - 1e-9)) + 1
    verts = sum(len(getattr(ob.data, "vertices", ()) or ()) for ob in objs if ob)
    return samples * verts * NF * 8, verts, samples


class MLOG_OT_record(Operator):
    bl_idname = "motion_log.record"
    bl_label = "Log Data"
    bl_description = ("Step through the frame range in sample-interval increments and log the "
                      "location and rotation of every vertex (corner) of every tracked object. "
                      "Press ESC to cancel")

    _timer = None

    confirmed: BoolProperty(default=False, options={"SKIP_SAVE", "HIDDEN"})

    def draw(self, context):
        """Contents of the large-recording confirmation dialog."""
        scene = context.scene
        est, verts, samples = _estimate_log_size(
            scene, [t.obj for t in scene.motion_log.objects if t.obj])
        col = self.layout.column(align=True)
        col.label(text=f"This recording needs about {est / 1e9:.1f} GB of memory.", icon="ERROR")
        col.label(text=f"{verts:,} vertices x {samples:,} samples x 6 values")
        col.separator()
        col.label(text="Blender may slow down or run out of memory. To reduce it,")
        col.label(text="raise the Sample Interval, shorten the frame range, or track fewer objects.")

    def execute(self, context):
        # Reached when the user confirms the dialog: start the real (modal) recording.
        bpy.ops.motion_log.record("INVOKE_DEFAULT", confirmed=True)
        return {"FINISHED"}

    def invoke(self, context, event):
        scene = context.scene
        s = scene.motion_log
        self.objs = [t.obj for t in s.objects if t.obj]
        if not self.objs:
            self.report({"ERROR"}, "Add at least one object to track first")
            return {"CANCELLED"}
        if not self.confirmed and _estimate_log_size(scene, self.objs)[0] > RISKY_BYTES:
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

        # Blender float properties are 32-bit (0.0021 becomes 0.002099999925...). The shortest
        # decimal that round-trips to that float32 is the value the user typed, kept as a double.
        self.interval = float(str(np.float32(s.sample_interval)))
        self.stride = s.vertex_stride
        self.unit_scale = scene.unit_settings.scale_length  # Blender units -> meters
        duration = (end - self.start) / self.fps
        self.duration = duration
        # enough samples to cover the whole range; the last one is placed exactly on the end frame
        self.total = int(math.ceil(duration / self.interval - 1e-9)) + 1
        self.i = 0
        self.orig_frame = scene.frame_current
        self.states = {ob.name: {} for ob in self.objs}

        _LOG["interval"] = self.interval
        _LOG["fps"] = self.fps
        _LOG["start_frame"] = self.start
        _LOG["data"] = {ob.name: [] for ob in self.objs}
        _LOG["verts"] = {ob.name: [] for ob in self.objs}
        _LOG["vmeta"] = {}
        _reset_icons()
        _STATE.update(recording=True, stop_requested=False, clear_after=False)
        _notify("started", objects=[ob.name for ob in self.objs], interval=self.interval,
                total=self.total)

        wm = context.window_manager
        wm.progress_begin(0, self.total)
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC" or (event.type == "TIMER" and _STATE["stop_requested"]):
            return self._finish(context, cancelled=True)
        if event.type == "TIMER":
            scene = context.scene
            for _ in range(20):  # samples per tick
                if self.i >= self.total:
                    return self._finish(context)
                t = min(self.i * self.interval, self.duration)
                frame_f = self.start + t * self.fps
                if abs(frame_f - round(frame_f)) < 1e-9:
                    frame_f = float(round(frame_f))
                whole = math.floor(frame_f)
                scene.frame_set(whole, subframe=frame_f - whole)
                dg = context.evaluated_depsgraph_get()
                for ob in self.objs:
                    st = self.states[ob.name]
                    vals, vvals = sample_object(ob, dg, st, self.stride, self.unit_scale)
                    _LOG["data"][ob.name].append((t, frame_f, *vals))
                    _LOG["verts"][ob.name].append(vvals)
                    if "wire" in st and ob.name not in _LOG["vmeta"]:
                        _LOG["vmeta"][ob.name] = st["wire"]
                self.i += 1
            context.window_manager.progress_update(self.i)
            _notify("progress", done=self.i, total=self.total)
            if context.area:
                context.area.header_text_set(
                    f"Motion Log: {self.i}/{self.total} samples (ESC to cancel)")
        return {"RUNNING_MODAL"}

    def _finish(self, context, cancelled=False):
        _STATE["recording"] = False
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        wm.progress_end()
        if context.area:
            context.area.header_text_set(None)
        context.scene.frame_set(self.orig_frame)
        for area in context.screen.areas:
            area.tag_redraw()
        self.report({"WARNING" if cancelled else "INFO"},
                    f"Motion log {'cancelled' if cancelled else 'finished'}: {self.i} samples")
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


def draw_buttons(layout):
    scene = bpy.context.scene
    tracked = [t.obj for t in scene.motion_log.objects if t.obj]
    est, verts, samples = _estimate_log_size(scene, tracked) if tracked else (0, 0, 0)
    if est > RISKY_BYTES:
        box = layout.box()
        col = box.column(align=True)
        col.alert = True
        col.label(text=f"Large recording: about {est / 1e9:.1f} GB of vertex data", icon="ERROR")
        col.label(text=f"{verts:,} vertices x {samples:,} samples")
        col.label(text="Raise Sample Interval or shorten the range to reduce it")
    row = layout.row(align=True)
    row.scale_y = 1.4
    row.operator("motion_log.record", text="Log Data", icon="REC")
    row.operator("motion_log.clear_log", text="Clear Data", icon="TRASH")


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
        if cand in data:
            name = cand
    return name


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


def draw_log(layout, context, buttons=False):
    if buttons:
        draw_buttons(layout)

    v = _log_view(context)
    if v is None:
        layout.label(text="No data yet. Press Log Data.", icon="INFO")
        return
    s, rows = v["s"], v["rows"]

    box = layout.box()
    box.label(text=f"{v['name']}  |  {len(v['raw'])} raw samples @ {_LOG['interval']} s  |  "
                   f"showing {s.display_fps} fps, "
                   + ("one row per shutter time" if s.display_per == "SHUTTER" else "one row per frame")
                   + f" ({len(rows)} rows, about every {v['every']:.1f} samples, "
                   f"shutter 1/{2 * int(s.display_fps)} s)")

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
    layout.label(text=f"Rows {r0 + 1}-{r0 + v['vis_rows']} of {len(rows)}   |   "
                      f"Columns {c0 + 1}-{c0 + v['vis_cols']} of {len(columns)} (Time pinned)")

    cols = columns[c0: c0 + v["vis_cols"]]

    # One grid cell per column (Time + the visible columns), all cells the same width, so the
    # icon, the vertex name, the title and the values of a column always sit in the same place.
    ncols = len(cols) + 1
    has_vertex = any(c[3] is not None for c in cols)
    meta = _LOG["vmeta"].get(v["name"])
    if s.log_view == "VERTICES":
        if len(columns) == 1:
            layout.label(text="This object has no vertices to log (empty, camera, light...)",
                         icon="INFO")
        else:
            layout.label(text=f"{len(columns) // NF} vertices logged: top vertex first, face by face (down, along, up), row by row downwards")

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
        head.label(text="")
        for _title, _f, _i, vert in cols:
            head.label(text="" if vert is None else f"Vertex {vert + 1}")
    # Column titles: when a title is wider than its column, its words are stacked underneath
    # each other (one grid row per line) instead of being cut off.
    region = getattr(context, "region", None)
    width = getattr(region, "width", 0) or 1000
    scale = context.preferences.view.ui_scale
    cap = max(4, int(((width - 24 * scale) / ncols - 14 * scale) / (6.8 * scale)))
    heads = [_wrap_title("Time (s)", cap)] + [_wrap_title(t, cap) for t, _f, _i, _v in cols]
    for j in range(max(len(h) for h in heads)):
        for h in heads:
            head.label(text=h[j] if j < len(h) else "")

    vrows = v["vrows"]
    grid = layout.grid_flow(row_major=True, columns=ncols, even_columns=True,
                            even_rows=False, align=True)
    for k in range(r0, min(r0 + v["vis_rows"], len(rows))):
        r = rows[k]
        vr = vrows[k] if vrows else None
        grid.label(text=f"{r[0]:.10f}")
        for _t, f, i, _v in cols:
            grid.label(text=_fmt(_cell(r, vr, i), f))


class MLOG_OT_show_log(Operator):
    bl_idname = "motion_log.show_log"
    bl_label = "Pop Up Log"
    bl_description = "Pop the log up (move the mouse away to close it)"

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=1100)

    def draw(self, context):
        draw_log(self.layout, context, buttons=True)

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
            f"{len(_LOG['data'][name])} raw samples @ {_LOG['interval']} s  |  showing "
            f"{s.display_fps} fps, "
            + ("one row per shutter time" if s.display_per == "SHUTTER" else "one row per frame")
            + f" ({len(rows)} rows)  |  Frame = scene frames"]
    if kv:
        head.append(f"{kv} vertices, ordered face by face (down, along, up), row by row downwards from the top vertex")
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

        layout.separator()
        col = layout.column(align=True)
        col.prop(s, "sample_interval")
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
                     "stored in the open .blend: the saved log, exported CSV Texts and the panel "
                     "settings. Leave it off to keep that data when the add-on is only disabled"),
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
    MLOG_OT_scroll, MLOG_OT_show_log, MLOG_OT_save_to_blend, MLOG_OT_load_from_blend,
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
