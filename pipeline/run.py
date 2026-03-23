
from huggingface_hub import login
import imageio
from streamlit import video
from text2image import load_model, generate_image
from image23d import (
    remove_background_v2,
    load_model_img23d,
    run_model,
)
import os
from dis_image_background_removal import build_model, get_hyperparameters
# Load Dict
from typing import Dict


class MARVEL_FX3D:
    def __init__(self, text2image_model=None, image2mesh_model=None):
        pass

    def run(self):
        print("Running MARVEL-FX3D pipeline...")
        
    @staticmethod
    def from_pretrained(model_name:str, image23d_model:Dict, **kwargs):
        """
        ```
        model = MARVEL_FX3D.from_pretrained(
            model_name="SadilKhan/MARVEL-FX3D-LORA-SD3",
            image23d_model={
                "type" : "Trellis2",
                "ckpt" : "microsoft/TRELLIS.2-4B"
        ```
        """
        
        verbose = kwargs.get("verbose", False)
        if verbose:
            print(f"Loading MARVEL-FX3D model '{model_name}'...")
            
        return MARVEL_FX3D(model_name, image23d_model, **kwargs)
    
    def from_trellis2(self, image, generate_video=False, output_folder="./output"):
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        ))
        
        mesh = self.image23d_model.run(image)[0]
        mesh.simplify(16777216) # nvdiffrast 
        
        # Save output
        os.makedirs(output_folder, exist_ok=True, mode=0o777)
        glb = o_voxel.postprocess.to_glb(
            vertices            =   mesh.vertices,
            faces               =   mesh.faces,
            attr_volume         =   mesh.attrs,
            coords              =   mesh.coords,
            attr_layout         =   mesh.layout,
            voxel_size          =   mesh.voxel_size,
            aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target   =   1000000,
            texture_size        =   4096,
            remesh              =   True,
            remesh_band         =   1,
            remesh_project      =   0,
            verbose             =   True
        )
        
        glb.export(os.path.join(output_folder, "sample.glb"), extension_webp=True)
        
        if generate_video:
            video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
            imageio.mimsave(os.path.join(output_folder, "sample.mp4"), video, fps=15)
        