import os
import sys
import time
from random import randint
from typing import Dict
import requests
from diffusers import StableDiffusionPipeline
import torch

sys.path.append("./")
from common import FastInferenceInterface


class FastStableDiffusion(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        super().__init__(model_name, args)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "/model/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096/",
            revision="fp16", 
            torch_dtype=torch.float16,
        )
        self.pipe = self.pipe.to("cuda")

    def infer(self, job_id, args) -> Dict:
        coord_url = os.environ.get("COORDINATOR_URL", "localhost:8092/my_coord")
        worker_name = os.environ.get("WORKER_NAME", "planetv2")
        res = requests.patch(
            f"http://{coord_url}/api/v1/g/jobs/atomic_job/{job_id}",
            json={
                "status": "running",
            }
        )
        start = time.time()
        image = self.pipe(args['input']).images[0]
        end = time.time()
        # save images to file
        fileid = randint(0, 100000)
        image.save(f"/app/results/image_{fileid}.png")
        # upload images to s3
        with open(f"/app/results/image_{fileid}.png", "rb") as fp:
            files = {"file": fp}
            res = requests.post("https://planetd.shift.ml/file", files=files).json()
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