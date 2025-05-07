from . import smea
from .smea import sample_smea, sample_smea_beta, sample_smea_dyn_beta


if smea.BACKEND == "ComfyUI":
    if not smea.INITIALIZED:
        from comfy.k_diffusion import sampling as k_diffusion_sampling
        from comfy.samplers import SAMPLER_NAMES

        setattr(k_diffusion_sampling, "spwaner_sample_smea", sample_smea)
        setattr(k_diffusion_sampling, "spwaner_sample_smea_beta", sample_smea_beta)
        setattr(k_diffusion_sampling, "spwaner_sample_smea_dyn_beta", sample_smea_dyn_beta)

        SAMPLER_NAMES.append("spwaner_sample_smea")
        SAMPLER_NAMES.append("spwaner_sample_smea_beta")
        SAMPLER_NAMES.append("spwaner_sample_smea_dyn_beta")

        smea.INITIALIZED = True

NODE_CLASS_MAPPINGS = {}
