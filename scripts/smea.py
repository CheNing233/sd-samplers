try:
    import smea
    from smea import sample_smea, sample_smea_beta, sample_smea_dyn_beta


    if smea.BACKEND == "WebUI":
        from modules import scripts, sd_samplers_common, sd_samplers
        from modules.sd_samplers_kdiffusion import sampler_extra_params, KDiffusionSampler

        class SMEA(scripts.Script):
            def title(self):
                "Spwaner SMEA Samplers"

            def show(self, is_img2img):
                return False

            def __init__(self):
                if not smea.INITIALIZED:
                    samplers_smea = [
                        ("Spwaner Euler SMEA", sample_smea, ["k_spwaner_euler_smea"], {}),
                        ("Spwaner Euler SMEA Dy Beta", sample_smea_beta, ["k_spwaner_euler_smea_dy"], {}),
                        ("Spwaner Euler SMEA Dyn Beta", sample_smea_dyn_beta, ["k_spwaner_euler_smea_dyn_beta"], {}),
                    ]
                    samplers_data_smea = [
                        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
                        for label, funcname, aliases, options in samplers_smea
                        if callable(funcname)
                    ]
                    sampler_extra_params["sample_spwaner_smea"] = ["s_churn", "s_tmin", "s_tmax", "s_noise"]
                    sampler_extra_params["sample_spwaner_smea_beta"] = ["s_churn", "s_tmin", "s_tmax", "s_noise"]
                    sampler_extra_params["sample_spwaner_smea_dyn_beta"] = ["s_churn", "s_tmin", "s_tmax", "s_noise"]
                    
                    sd_samplers.all_samplers.extend(samplers_data_smea)
                    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
                    sd_samplers.set_samplers()
                    smea.INITIALIZED = True

except ImportError as _:
    pass
