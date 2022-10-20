import json
import os
import sys
import time
from random import randint
from typing import Dict
import matplotlib.pyplot as plt
import requests
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from io import BytesIO

sys.path.append("./")
from common import FastInferenceInterface


class FastStableDiffusion(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        super().__init__(model_name, args)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16", 
            torch_dtype=torch.float16,
        )
        self.pipe = self.pipe.to("cuda")
        self.generator = torch.Generator(device="cuda")

    def infer(self, job_id, args) -> Dict:
        coord_url = os.environ.get("COORDINATOR_URL", "localhost:8092/my_coord")
        worker_name = os.environ.get("WORKER_NAME", "planetv2")
        upload_token = os.environ.get("UPLOAD_TOKEN", "token")
        res = requests.patch(
            f"http://{coord_url}/api/v1/g/jobs/atomic_job/{job_id}",
            json={
                "status": "running",
            }
        )
        
        self.generator.manual_seed(args['seed'])
        strength = 0.8
        guidance_scale = 7.5

        if 'strength' in args:
            strength = args['strength']
        
        if 'guidance_scale' in args:
            guidance_scale = args['guidance_scale']

        start = time.time()
        with torch.autocast("cuda"):
            image = self.pipe(
                prompt=args['prompt'],
                strength=strength,
                guidance_scale=guidance_scale,
                generator=self.generator,
            ).images[0]
        end = time.time()
        # save images to file
        fileid = randint(0, 100000)
        image.save(f"/app/results/image_{fileid}.png")
        # upload images to s3
        with open(f"/app/results/image_{fileid}.png", "rb") as fp:
            files = {"file": fp}
            res = requests.post("https://api.together.xyz/file", files=files, headers={
                "together-token": upload_token
            }).json()
            filename=res["filename"]
        os.remove(f"/app/results/image_{fileid}.png")
        # delete the file
        print("sending requests to global")
        # write results back
        requests.patch(
            f"http://{coord_url}/api/v1/g/jobs/atomic_job/{job_id}",
            json={
                "status": "finished",
                "output": {
                    "output": filename,
                    "cal_time": end - start,
                },
                "processed_by": worker_name,
            },
        )
        return {"output": filename}


if __name__ == "__main__":
    fip = FastStableDiffusion(model_name="together.StableDiffusion")
    fip.start()
