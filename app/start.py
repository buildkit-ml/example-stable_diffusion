import sys
sys.path.append("./")
from stable_diffusion import FastStableDiffusion

if __name__ == "__main__":
    fip = FastStableDiffusion(model_name="together.StableDiffusion")
    fip.start()
