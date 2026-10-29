"""Turn a cleaned-up splat (.ply exported from SuperSplat) into a surface mesh.

Uses the splat centers as a point cloud and runs Poisson surface reconstruction.
Works best on a 2DGS splat whose background has been deleted in SuperSplat.

Usage:
    python make_mesh.py cleaned_2dgs.ply --out specimen_mesh
    -> specimen_mesh.ply (with vertex normals) and specimen_mesh.stl (for 3D printing)
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("splat", type=Path, help="cleaned splat .ply (uncompressed) from SuperSplat")
    ap.add_argument("--out", type=Path, required=True, help="output path without extension")
    ap.add_argument("--depth", type=int, default=9, help="Poisson detail level: 8 = smoother, 10 = more detail")
    ap.add_argument("--min-opacity", type=float, default=0.0,
                    help="drop splats more transparent than this (0-1) before meshing")
    ap.add_argument("--trim", type=float, default=0.02,
                    help="remove this fraction of the least-supported mesh vertices (cleans up 'bubbles')")
    args = ap.parse_args()

    v = PlyData.read(str(args.splat))["vertex"].data
    if "x" not in v.dtype.names:
        raise SystemExit("This .ply has no x/y/z columns. In SuperSplat export as plain 'PLY', not 'Compressed PLY'.")
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    if args.min_opacity > 0 and "opacity" in v.dtype.names:
        alpha = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
        xyz = xyz[alpha > args.min_opacity]
    print(f"Meshing {len(xyz):,} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # Normal-estimation radius scales with specimen size so it doesn't depend on SuperSplat's units.
    diag = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04 * diag, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, args.trim))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".ply", ".stl"):
        o3d.io.write_triangle_mesh(str(args.out.with_suffix(ext)), mesh)
    print(f"Wrote {args.out}.ply and {args.out}.stl "
          f"({len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles)")


if __name__ == "__main__":
    main()
