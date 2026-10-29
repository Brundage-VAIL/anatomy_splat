# Anatomy Splat

Turn a phone video of an anatomical specimen into an interactive 3D **Gaussian splat**, and optionally a 3D-printable **mesh**. Everything runs in Google Colab, so you don't need a GPU or any software on your own computer.

**[▶ Open the notebook in Google Colab](https://colab.research.google.com/github/Brundage-VAIL/anatomy_splat/blob/main/anatomy_splat.ipynb)**

```
video ──► sharp frames ──► COLMAP (camera positions) ──► gsplat training ──► splat .ply ──► SuperSplat (clean / view / share)
                                                                 └──► 2DGS training ──► cleaned splat ──► mesh .ply / .stl
```

## What you need
- A Google account with about 2 GB free in Google Drive
- A phone that can record video (any recent iPhone or Android works)
- About 1.5 hours, mostly waiting while the model trains

---

## 1. Film your specimen
Video quality matters more than any setting in the notebook. A good video takes 1–2 minutes.

**Set up**
- **Even, diffuse light.** Use an overcast window or several lamps. Avoid a single harsh light and avoid flash.
- **Matte, plain surface** under the specimen, such as a dark cloth or paper towel. Avoid shiny steel trays.
- **Blot wet tissue dry.** Glistening highlights move as the camera moves, and that confuses the reconstruction.
- Add a few **non-repeating objects** around the specimen (a ruler, a labelled card). They give COLMAP extra features to track.

**Record**
- **Lock focus and exposure.** On iPhone, press and hold on the specimen until *AE/AF LOCK* appears. On Android, tap and hold, or use Pro mode.
- Use **1080p or 4K at 30 fps**. Turn off Cinematic, Portrait and Action modes. Don't zoom; move closer instead.
- **Walk slowly in a full circle** around the specimen while keeping it in the middle of the frame. Then do **a second circle higher up**, looking down, and a third low circle if you can.
- **Move slowly and smoothly.** Motion blur is the most common cause of failure.
- **The specimen must not move.** Don't turn it on a turntable or by hand while filming. The camera moves; the specimen stays still.
- If one side is hidden (for example the underside), you can flip the specimen and film a **second video**. Upload both. Expect some mismatch where the two halves meet.

## 2. Upload to Google Drive
Create this folder structure in Google Drive and put your video(s) in `videos/`:

```
My Drive/
└── anatomy_splat/
    └── equine_heart/          ← your project name (no spaces)
        └── videos/
            └── IMG_1234.MOV
```

`.mp4` and `.mov` both work. A folder of photos (`.jpg` / `.png`) also works instead of a video.

## 3. Run the notebook
1. Open the **[notebook in Colab](https://colab.research.google.com/github/Brundage-VAIL/anatomy_splat/blob/main/anatomy_splat.ipynb)**.
2. Go to **Runtime → Change runtime type → T4 GPU → Save**.
3. In cell ①, set `PROJECT_NAME` to your folder name, e.g. `equine_heart`.
4. Run the cells one at a time, top to bottom. Each cell explains what it does and roughly how long it takes.
5. When asked, allow Colab to access your Google Drive.

Tip: for your first attempt, set `TRAIN_STEPS = 7000` for a quick result (about 10 min of training) to check that everything works. Then run again with 30000.

Results appear in `My Drive/anatomy_splat/<project>/outputs/`:

| File | What it is |
|---|---|
| `<project>_splat.ply` | **Your Gaussian splat.** Open it in SuperSplat. |
| `training_3dgs/videos/traj_*.mp4` | fly-around video rendered from the splat |
| `training_3dgs/stats/` | quality scores (PSNR / SSIM / LPIPS) |
| `colmap_data.zip` | camera positions, kept so you can retrain without rerunning COLMAP |
| `<project>_2dgs.ply`, `<project>_mesh.ply/.stl` | from the optional mesh section |

## 4. Clean up, view and share in SuperSplat
1. Open **[superspl.at/editor](https://superspl.at/editor)** and drag your `.ply` in.
2. Select the background, table and floating "fog" with the **Sphere/Box** selection tools and press **Delete**. Tip: select the specimen, then **Invert** the selection and delete.
3. Use **File → Export → PLY** to save, or **File → Publish** to get a shareable web link.

## 5. (Optional) Make a mesh
The last section of the notebook trains a 2D Gaussian splat, has you clean it in SuperSplat, then converts it to a mesh (`.ply` and `.stl`). You can open the mesh in MeshLab, Blender or 3D Slicer, or use it for 3D printing. The notebook walks you through each step.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| **"No GPU found"** | Runtime → Change runtime type → **T4 GPU**. If no GPU is available, Colab's free tier is busy; try again later. |
| **"No videos found"** | Check that the folder is exactly `My Drive/anatomy_splat/<PROJECT_NAME>/videos/` and that the name in cell ① matches, including capital letters. |
| **COLMAP places under 70% of frames, or fails** | Usually the video is blurry, moves too fast, or the specimen is shiny or featureless. Try `NUM_FRAMES = 250`. If that doesn't help, re-film more slowly with locked exposure and a textured background. Delete `outputs/colmap_data.zip` before rerunning. |
| **"Scene split into N pieces"** | COLMAP could not connect parts of the video, often where the camera moved quickly. Only the biggest piece is used. Re-film with more overlap between views. |
| **CUDA out of memory** during training | Set `MAX_IMAGE_SIZE = 1200` and rerun from step ④. Delete `outputs/colmap_data.zip` first, because the image size changed. |
| **Colab disconnected** mid-training | Run cells ①–③ again. Step ⑤ reloads the saved COLMAP result. Then rerun training. Keep the tab open; free Colab disconnects idle tabs after about 90 minutes. |
| **Splat looks foggy or has floaters** | Normal at the edges. Delete them in SuperSplat. If the specimen itself is blurry, the video needs more angles. |
| **Splat is sharp from filmed angles but smeared in between** | Not enough viewpoints. Add another circle at a different height. |
| **Mesh step: "no x/y/z columns"** | In SuperSplat, export as plain **PLY**, not *Compressed PLY*. |
| **Mesh has bubbles or webbing** | Clean the 2DGS splat more aggressively in SuperSplat, or try `POISSON_DEPTH = 8`. |

## What's in this repository
```
anatomy_splat.ipynb        the Colab notebook students run
scripts/
  colab_setup.sh           installs COLMAP, gsplat 1.5.3 and pinned dependencies
  extract_frames.py        picks the sharpest frame in each time window of the video
  run_colmap.py            COLMAP structure-from-motion + undistortion
  export_ply.py            training checkpoint → standard splat .ply
  make_mesh.py             cleaned splat → Poisson surface mesh (.ply/.stl)
  patch_pycolmap.py        compatibility fix for NumPy 2 / Windows
docs/instructor_notes.md   how the pipeline is put together and how to update it
```

## Credits
Built on [gsplat](https://github.com/nerfstudio-project/gsplat) (Apache-2.0), [COLMAP](https://colmap.github.io/) (BSD), [SuperSplat](https://github.com/playcanvas/supersplat) (MIT) and [Open3D](https://www.open3d.org/) (MIT). Methods: Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH 2023); Huang et al., *2D Gaussian Splatting for Geometrically Accurate Radiance Fields* (SIGGRAPH 2024).
