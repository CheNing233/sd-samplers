from . import smea
from .smea import sample_spawner_smea, sample_spawner_smea_beta, sample_spawner_smea_dyn_beta, sample_spawner_smea_dyn_beta1


if smea.BACKEND == "ComfyUI":
    if not smea.INITIALIZED:
        from comfy.k_diffusion import sampling as k_diffusion_sampling
        from comfy.samplers import SAMPLER_NAMES

        setattr(k_diffusion_sampling, "sample_spawner_smea", sample_spawner_smea)
        setattr(k_diffusion_sampling, "sample_spawner_smea_beta", sample_spawner_smea_beta)
        setattr(k_diffusion_sampling, "sample_spawner_smea_dyn_beta", sample_spawner_smea_dyn_beta)
        setattr(k_diffusion_sampling, "sample_spawner_smea_dyn_beta1", sample_spawner_smea_dyn_beta1)

        SAMPLER_NAMES.append("spawner_smea")
        SAMPLER_NAMES.append("spawner_smea_beta")
        SAMPLER_NAMES.append("spawner_smea_dyn_beta")
        SAMPLER_NAMES.append("spawner_smea_dyn_beta1")

        smea.INITIALIZED = True

NODE_CLASS_MAPPINGS = {}
