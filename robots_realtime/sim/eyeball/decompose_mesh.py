"""
Mesh preprocessing for MuJoCo: convex decomposition + optional texture extraction.

Handles both plain meshes (OBJ/STL) and textured GLB/glTF files. For GLB inputs,
extracts per-material OBJ submeshes with PNG textures, then decomposes a merged
mesh for collision geometry. Results are cached in collision_config.json.

Python usage (in scene construction):
    from .decompose_mesh import MeshAsset

    mesh = MeshAsset.load("path/to/assets/yellow_tape")
    mesh.add_visual_assets_to_spec(spec, "yellow_tape")
    mesh.add_visual_geoms(body, "yellow_tape")
    mesh.add_collision_geoms(body, "yellow_tape", friction=[1, 0.05, 0.01])
"""

import argparse
import json
from pathlib import Path


class MeshAsset:
    """Cached mesh asset with convex collision decomposition and optional textures.

    For GLB inputs, also stores per-material visual submeshes (OBJ + PNG texture).
    For plain OBJ/STL inputs, visual_submeshes is None and rendering uses flat rgba.
    """

    def __init__(
        self,
        directory: Path,
        visual_mesh: str,
        collision_meshes: list[str],
        visual_submeshes: list[dict] | None = None,
        mesh_scale: float | None = None,
    ):
        self.directory = Path(directory)
        self.visual_mesh = visual_mesh
        self.collision_meshes = collision_meshes
        self.visual_submeshes = visual_submeshes
        self.mesh_scale = mesh_scale

    @property
    def visual_mesh_path(self) -> Path:
        return self.directory / self.visual_mesh

    @property
    def collision_mesh_paths(self) -> list[Path]:
        return [self.directory / f for f in self.collision_meshes]

    @property
    def has_textures(self) -> bool:
        return self.visual_submeshes is not None and len(self.visual_submeshes) > 0

    @property
    def _scale_vec(self) -> list[float] | None:
        """[sx, sy, sz] scale vector for MuJoCo mesh assets, or None."""
        if self.mesh_scale is None:
            return None
        if isinstance(self.mesh_scale, list):
            return self.mesh_scale
        return [self.mesh_scale] * 3

    # ── Persistence ──

    def save(self):
        config = {
            "visual_mesh": self.visual_mesh,
            "collision_meshes": self.collision_meshes,
        }
        if self.visual_submeshes is not None:
            config["visual_submeshes"] = self.visual_submeshes
        if self.mesh_scale is not None:
            config["mesh_scale"] = self.mesh_scale
        with open(self.directory / "collision_config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "MeshAsset":
        """Load cached decomposition from a directory."""
        directory = Path(directory)
        config_path = directory / "collision_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No collision_config.json in {directory}. Run decomposition first."
            )
        with open(config_path) as f:
            config = json.load(f)
        return cls(directory=directory, **config)

    # ── Decomposition from plain mesh ──

    @classmethod
    def from_mesh(
        cls,
        mesh_path: str | Path,
        threshold: float = 0.02,
        max_convex_hull: int = -1,
        force: bool = False,
    ) -> "MeshAsset":
        """Decompose a mesh into convex parts (or load from cache).

        Args:
            mesh_path: Path to input mesh (OBJ/STL)
            threshold: CoACD concavity threshold (lower = more parts, tighter fit)
            max_convex_hull: Max convex parts (-1 = auto)
            force: Force re-decomposition even if cached
        """
        mesh_path = Path(mesh_path)
        directory = mesh_path.parent
        config_path = directory / "collision_config.json"
        name = mesh_path.stem

        # Check cache
        if config_path.exists() and not force:
            obj = cls.load(directory)
            if all((directory / f).exists() for f in obj.collision_meshes):
                print(f"  Cached: {len(obj.collision_meshes)} convex parts")
                return obj

        # Run decomposition
        import coacd
        import trimesh

        print(f"  Decomposing {mesh_path} (threshold={threshold})...")
        mesh = trimesh.load(str(mesh_path), force="mesh")
        coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        parts = coacd.run_coacd(
            coacd_mesh,
            threshold=threshold,
            max_convex_hull=max_convex_hull,
            preprocess_mode="auto",
        )
        print(f"  → {len(parts)} convex parts")

        # Save collision meshes
        collision_files = []
        for i, (vertices, faces) in enumerate(parts):
            col_filename = f"{name}_collision_{i}.obj"
            part_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            part_mesh.export(str(directory / col_filename), file_type="obj")
            collision_files.append(col_filename)
            print(f"    {col_filename}: {len(vertices)} verts, {len(faces)} faces")

        obj = cls(
            directory=directory,
            visual_mesh=mesh_path.name,
            collision_meshes=collision_files,
        )
        obj.save()
        return obj

    # ── Decomposition from GLB (textured) ──

    @classmethod
    def from_glb(
        cls,
        glb_path: str | Path,
        threshold: float = 0.02,
        max_convex_hull: int = -1,
        force: bool = False,
        scale: float | None = None,
    ) -> "MeshAsset":
        """Convert GLB to textured OBJ submeshes + convex collision parts.

        Pipeline:
        1. Load GLB scene with trimesh (preserving textures)
        2. For each geometry: export OBJ with texcoords, extract PNG texture
        3. Merge all geometries for collision decomposition via CoACD
        4. Save collision_config.json with visual_submeshes metadata

        Args:
            glb_path: Path to input GLB/glTF file
            threshold: CoACD concavity threshold
            max_convex_hull: Max convex parts (-1 = auto)
            force: Force re-processing even if cached
            scale: Uniform scale factor stored in config and applied at MuJoCo load time
        """
        import numpy as np
        import trimesh
        from PIL import Image

        glb_path = Path(glb_path)
        directory = glb_path.parent
        config_path = directory / "collision_config.json"
        name = glb_path.stem

        # Check cache — must have visual_submeshes and all files present
        if config_path.exists() and not force:
            obj = cls.load(directory)
            if obj.visual_submeshes is not None and all(
                (directory / f).exists() for f in obj.collision_meshes
            ) and all(
                (directory / sm["mesh_file"]).exists()
                and (directory / sm["texture_file"]).exists()
                for sm in obj.visual_submeshes
            ):
                print(
                    f"  Cached: {len(obj.visual_submeshes)} textured submeshes, "
                    f"{len(obj.collision_meshes)} collision parts"
                )
                return obj

        print(f"  Processing GLB: {glb_path}")
        scene = trimesh.load(str(glb_path))

        # Handle single-mesh GLB (trimesh returns Trimesh instead of Scene)
        if isinstance(scene, trimesh.Trimesh):
            geometries = {"mesh_0": scene}
        else:
            geometries = scene.geometry

        # GLB uses Y-up, MuJoCo uses Z-up → rotate +90° around X
        y_up_to_z_up = np.array([
            [1,  0,  0, 0],
            [0,  0, -1, 0],
            [0,  1,  0, 0],
            [0,  0,  0, 1],
        ], dtype=np.float64)
        for mesh in geometries.values():
            if isinstance(mesh, trimesh.Trimesh):
                mesh.apply_transform(y_up_to_z_up)

        visual_submeshes = []
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        for i, (geom_name, mesh) in enumerate(geometries.items()):
            if not isinstance(mesh, trimesh.Trimesh):
                continue

            mesh_file = f"{name}_visual_{i}.obj"
            texture_file = f"{name}_visual_{i}.png"
            material_name = f"{name}_mat_{i}"

            # Extract texture from PBR material
            has_texture = False
            if hasattr(mesh.visual, "material") and hasattr(
                mesh.visual.material, "baseColorTexture"
            ):
                tex_image = mesh.visual.material.baseColorTexture
                if tex_image is not None:
                    if not isinstance(tex_image, Image.Image):
                        tex_image = Image.fromarray(np.asarray(tex_image))
                    tex_image.save(str(directory / texture_file))
                    has_texture = True
                    print(f"    Texture {i}: {tex_image.size[0]}x{tex_image.size[1]}")

            if not has_texture:
                # Try TextureVisuals with an image
                if hasattr(mesh.visual, "image") and mesh.visual.image is not None:
                    mesh.visual.image.save(str(directory / texture_file))
                    has_texture = True
                    print(
                        f"    Texture {i}: {mesh.visual.image.size[0]}x{mesh.visual.image.size[1]}"
                    )

            if not has_texture:
                print(f"    Warning: no texture found for geometry '{geom_name}', skipping submesh")
                # Still accumulate for collision even without texture
                all_vertices.append(mesh.vertices)
                all_faces.append(mesh.faces + vertex_offset)
                vertex_offset += len(mesh.vertices)
                continue

            # Export OBJ with texcoords (trimesh includes UV mapping)
            obj_data = mesh.export(file_type="obj")
            if isinstance(obj_data, bytes):
                obj_data = obj_data.decode("utf-8")

            # Rewrite the OBJ to reference our material and remove the mtllib line.
            # We'll create a minimal MTL that points to the PNG texture.
            mtl_file = f"{name}_visual_{i}.mtl"
            lines = obj_data.split("\n")
            new_lines = []
            for line in lines:
                if line.startswith("mtllib "):
                    new_lines.append(f"mtllib {mtl_file}")
                elif line.startswith("usemtl "):
                    new_lines.append(f"usemtl {material_name}")
                else:
                    new_lines.append(line)
            with open(directory / mesh_file, "w") as f:
                f.write("\n".join(new_lines))

            # Write MTL file referencing the PNG texture
            mtl_content = (
                f"newmtl {material_name}\n"
                f"Ka 1.0 1.0 1.0\n"
                f"Kd 1.0 1.0 1.0\n"
                f"Ks 0.0 0.0 0.0\n"
                f"d 1.0\n"
                f"map_Kd {texture_file}\n"
            )
            with open(directory / mtl_file, "w") as f:
                f.write(mtl_content)

            visual_submeshes.append(
                {
                    "mesh_file": mesh_file,
                    "texture_file": texture_file,
                    "material_name": material_name,
                }
            )

            # Accumulate for merged collision mesh
            all_vertices.append(mesh.vertices)
            all_faces.append(mesh.faces + vertex_offset)
            vertex_offset += len(mesh.vertices)

            print(
                f"    Submesh {i} ({geom_name}): {len(mesh.vertices)} verts, "
                f"{len(mesh.faces)} faces"
            )

        if not visual_submeshes:
            raise ValueError(
                f"No textured geometries found in {glb_path}. "
                "Use from_mesh() for untextured meshes."
            )

        # Merge all geometries for collision decomposition
        merged_vertices = np.concatenate(all_vertices, axis=0)
        merged_faces = np.concatenate(all_faces, axis=0)
        merged_mesh = trimesh.Trimesh(vertices=merged_vertices, faces=merged_faces)

        # Also save merged mesh as the visual_mesh fallback (flat-color OBJ)
        visual_mesh_file = f"{name}.obj"
        merged_mesh.export(str(directory / visual_mesh_file), file_type="obj")

        # Run CoACD on merged mesh
        import coacd

        print(f"  Decomposing merged mesh (threshold={threshold})...")
        coacd_mesh = coacd.Mesh(merged_mesh.vertices, merged_mesh.faces)
        parts = coacd.run_coacd(
            coacd_mesh,
            threshold=threshold,
            max_convex_hull=max_convex_hull,
            preprocess_mode="auto",
        )
        print(f"  → {len(parts)} convex parts")

        collision_files = []
        for i, (vertices, faces) in enumerate(parts):
            col_filename = f"{name}_collision_{i}.obj"
            part_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            part_mesh.export(str(directory / col_filename), file_type="obj")
            collision_files.append(col_filename)
            print(f"    {col_filename}: {len(vertices)} verts, {len(faces)} faces")

        obj = cls(
            directory=directory,
            visual_mesh=visual_mesh_file,
            collision_meshes=collision_files,
            visual_submeshes=visual_submeshes,
            mesh_scale=scale,
        )
        obj.save()
        return obj

    # ── MjSpec integration ──

    def add_meshes_to_spec(self, spec, name: str):
        """Add all collision mesh assets to a MjSpec.

        Args:
            spec: mujoco.MjSpec
            name: Base name for mesh identifiers
        """
        for col_file in self.collision_meshes:
            col_name = Path(col_file).stem
            m = spec.add_mesh(name=col_name, file=str(self.directory / col_file))
            if self._scale_vec:
                m.scale = self._scale_vec

    def add_collision_geoms(self, body, name: str, **geom_kwargs):
        """Add collision geoms to an existing MjSpec body.

        Args:
            body: MjSpec body to add geoms to
            name: Base name for geom identifiers
            **geom_kwargs: Extra args passed to each geom (friction, solref, etc.)
        """
        import mujoco

        for i, col_file in enumerate(self.collision_meshes):
            col_mesh_name = Path(col_file).stem
            body.add_geom(
                name=f"{name}_collision_{i}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=col_mesh_name,
                group=3, contype=1, conaffinity=1, mass=0,
                **geom_kwargs,
            )

    def add_visual_assets_to_spec(self, spec, name: str):
        """Register texture, material, and mesh assets for each visual submesh.

        Must be called before add_visual_geoms(). Only valid when has_textures is True.

        Args:
            spec: mujoco.MjSpec
            name: Base name prefix for asset identifiers
        """
        import mujoco

        assert self.visual_submeshes, "No visual submeshes — use flat-color rendering path"

        for sm in self.visual_submeshes:
            mat_name = sm["material_name"]

            # Register texture asset (2D image)
            tex = spec.add_texture()
            tex.name = f"{mat_name}_tex"
            tex.type = mujoco.mjtTexture.mjTEXTURE_2D
            tex.file = str(self.directory / sm["texture_file"])

            # Register material referencing the texture
            mat = spec.add_material()
            mat.name = mat_name
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{mat_name}_tex"

            # Register visual mesh
            mesh_name = Path(sm["mesh_file"]).stem
            vm = spec.add_mesh(name=mesh_name, file=str(self.directory / sm["mesh_file"]))
            if self._scale_vec:
                vm.scale = self._scale_vec

    def add_visual_geoms(self, body, name: str, **geom_kwargs):
        """Add textured visual geoms (one per submesh) to a body.

        Must call add_visual_assets_to_spec() first. Each geom uses material
        binding instead of rgba for textured rendering.

        Args:
            body: MjSpec body to add geoms to
            name: Base name prefix for geom identifiers
            **geom_kwargs: Extra args (mass, etc.)
        """
        import mujoco

        assert self.visual_submeshes, "No visual submeshes — use flat-color rendering path"

        for i, sm in enumerate(self.visual_submeshes):
            mesh_name = Path(sm["mesh_file"]).stem
            body.add_geom(
                name=f"{name}_visual_{i}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                material=sm["material_name"],
                contype=0,
                conaffinity=0,
                group=0,
                **geom_kwargs,
            )


# Keep backward-compatible alias
CollisionMesh = MeshAsset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess mesh for MuJoCo: convex decomposition + optional texture extraction"
    )
    parser.add_argument("mesh", type=str, help="Path to input mesh (OBJ/STL/GLB/glTF)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Concavity threshold (lower = more parts)")
    parser.add_argument("--max-parts", type=int, default=-1,
                        help="Max convex parts (-1 = auto)")
    parser.add_argument("--scale", type=float, nargs="+", default=None,
                        help="Scale factor: 1 float for uniform, or 3 floats (x y z)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-processing (ignore cache)")
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    suffix = mesh_path.suffix.lower()

    # Parse scale: single float → uniform, 3 floats → per-axis [x, y, z]
    scale = args.scale
    if scale is not None:
        if len(scale) == 1:
            scale = scale[0]
        elif len(scale) == 3:
            pass  # keep as list
        else:
            parser.error("--scale must be 1 or 3 values")

    if suffix in (".glb", ".gltf"):
        obj = MeshAsset.from_glb(
            glb_path=mesh_path,
            threshold=args.threshold,
            max_convex_hull=args.max_parts,
            force=args.force,
            scale=scale,
        )
        print(f"\nSaved → {obj.directory / 'collision_config.json'}")
        print(f"Visual submeshes: {len(obj.visual_submeshes)}")
        print(f"Collision parts: {len(obj.collision_meshes)}")
    else:
        obj = MeshAsset.from_mesh(
            mesh_path=mesh_path,
            threshold=args.threshold,
            max_convex_hull=args.max_parts,
            force=args.force,
        )
        print(f"\nSaved → {obj.directory / 'collision_config.json'}")
        print(f"Visual mesh: {obj.visual_mesh}")
        print(f"Collision parts: {len(obj.collision_meshes)}")


if __name__ == "__main__":
    main()
