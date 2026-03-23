from dis_image_background_removal import build_model, inference, get_hyperparameters
from typing import List
from imagegen import load_model, generate_image
import numpy as np


class MARVEL_FX3DPipeline:
    def __init__(self, text2image_model, image23d_model_type: str, image23d_model):
        self.text2image_model = text2image_model
        self.image23d_model = image23d_model
        self._hypar = get_hyperparameters("DIS/saved_models")
        self.dis_model = build_model()
        self.image23d_model_type = image23d_model_type

    def _generate_image(
        self,
        prompts,
        negative_prompt="blurry, low quality, bad lighting",
        width=512,
        height=512,
        num_inference_steps=30,
        guidance_scale=7.5,
    ):
        return generate_image(
            pipeline=self.text2image_model,
            prompt=prompts,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )

    def _remove_background(self, image):
        image_list = [image] if not isinstance(image, list) else image
        masked_image_list = []
        pil_masked_image_list = []
        for image in image_list:
            masked_image = inference(self.dis_model, self._hypar, image)
            masked_image_list.append(masked_image[0])
            pil_masked_image_list.append(masked_image[1])

        return masked_image_list, pil_masked_image_list

    def _generate_3d(self, image):
        masked_image_list, pil_masked_image_list = self._remove_background(image)
        if self.image23d_model_type == "sam3d":
            return self._generate_3d_sam3d(masked_image_list, pil_masked_image_list)
        elif self.image23d_model_type == "trellis2":
            return self._generate_3d_trellis2(masked_image_list, pil_masked_image_list)
        else:
            raise ValueError(f"Invalid image23d_model_type: {self.image23d_model_type}")

    def _generate_3d_sam3d(self, masked_image_list, pil_masked_image_list):
        output_list = []
        for masked_image, pil_masked_image in zip(
            masked_image_list, pil_masked_image_list
        ):
            output = self.image23d_model(
                np.asarray(masked_image)[:, :, :3],
                np.asarray(pil_masked_image),
                seed=42,
            )
            output_list.append(output)
        return output_list

    def _generate_3d_trellis2(self, masked_image_list, pil_masked_image_list):
        raise NotImplementedError("Trellis2 3D generation is not implemented yet.")

    def run(
        self,
        prompts,
        negative_prompt="blurry, low quality, bad lighting",
        width=512,
        height=512,
        num_inference_steps=30,
        guidance_scale=7.5,
    ):
        image = self._generate_image(
            prompts, negative_prompt, width, height, num_inference_steps, guidance_scale
        )
        return self._generate_3d(image)
