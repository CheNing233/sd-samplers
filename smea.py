from importlib import import_module
import torch
from tqdm.auto import trange

sampling = None
BACKEND = None
INITIALIZED = False

if not BACKEND:
    try:
        _ = import_module("modules.sd_samplers_kdiffusion")
        sampling = import_module("k_diffusion.sampling")
        BACKEND = "WebUI"
    except ImportError as _:
        pass

if not BACKEND:
    try:
        sampling = import_module("comfy.k_diffusion.sampling")
        BACKEND = "ComfyUI"
    except ImportError as _:
        pass

def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


# ---------------------下面这段复制到k_diffusion\sampling.py的最后面(我用的forge，别的我不知道)--------------------------------------------
@torch.no_grad()
def sample_spawner_smea(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, alpha=0.5):
    print("Use Spawner SMEA")
    
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        current_denoised = model(x, sigmas[i] * s_in, **extra_args)

        if old_denoised is not None and sigmas[i+1] > 0:
            effective_denoised = (1 - alpha) * current_denoised + alpha * old_denoised
        else:
            effective_denoised = current_denoised

        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        if callback is not None:
            callback_payload = {'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': current_denoised}
            if old_denoised is not None:
                callback_payload['effective_denoised'] = effective_denoised
            callback(callback_payload)

        d = to_d(x, sigmas[i], effective_denoised)

        dt = sigma_down - sigmas[i]
        x = x + d * dt

        if sigmas[i + 1] > 0 and sigma_up > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        old_denoised = current_denoised

    return x

@torch.no_grad()
def sample_spawner_smea_beta(model, x, sigmas,
                             extra_args=None, callback=None, disable=None,
                             eta=0.85,
                             s_noise=1.0,
                             noise_sampler=None,
                             beta=0.55):
    print("Use Spawner SMEA beta")
    
    if not isinstance(sigmas, torch.Tensor):
        sigmas = x.new_tensor(sigmas)
    if not (0.0 <= beta <= 1.0):
        raise ValueError("Parameter 'beta' must be between 0.0 and 1.0.")
    if not (0.0 <= eta <= 1.0):
        raise ValueError("Parameter 'eta' must be between 0.0 and 1.0.")

    extra_args = extra_args or {}
    noise_sampler = noise_sampler or default_noise_sampler(x)
    
    s_in = x.new_ones([x.shape[0]]) 
    
    old_denoised_for_beta = None

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_current = sigmas[i]
        sigma_next_actual = sigmas[i+1]

        sigma_for_model = sigma_current * s_in 
        denoised_at_current_sigma = model(x, sigma_for_model, **extra_args)

        if old_denoised_for_beta is not None and beta > 0.0 and sigma_next_actual < sigma_current:
            effective_x0_hat = (1 - beta) * denoised_at_current_sigma + beta * old_denoised_for_beta
        else:
            effective_x0_hat = denoised_at_current_sigma
        
        old_denoised_for_beta = denoised_at_current_sigma

        d = to_d(x, sigma_current, effective_x0_hat)
        
        sigma_down_float, sigma_up_float = get_ancestral_step(sigma_current.item(), sigma_next_actual.item(), eta=eta)
        
        sigma_down = x.new_tensor(sigma_down_float)
        sigma_up = x.new_tensor(sigma_up_float)
        
        if callback is not None:
            callback_dict = {'x': x, 'i': i, 'sigma': sigma_current, 'sigma_hat': sigma_current, 'denoised': denoised_at_current_sigma}
            if old_denoised_for_beta is not None and beta > 0.0 and 'effective_x0_hat' in locals() and effective_x0_hat is not denoised_at_current_sigma:
                 callback_dict['effective_denoised'] = effective_x0_hat
            callback(callback_dict)

        x = x + d * (sigma_down - sigma_current)

        if sigma_up > 0:
            added_noise = noise_sampler(sigma_current, sigma_next_actual)
            x = x + added_noise * s_noise * sigma_up
            
    return x

@torch.no_grad()
def sample_spawner_smea_dyn_beta(model, x, sigmas,
                                 extra_args=None, callback=None, disable=None,
                                 eta_start=0.95,
                                 eta_end=0.70,
                                 eta_exponent=1.0,
                                 s_noise=1.0,
                                 noise_sampler=None,
                                 beta=0.55,
                                 sigma_max_for_dyn_eta=None
                                 ):
    print("Use Spawner SMEA Dyn beta")
    
    if not isinstance(sigmas, torch.Tensor):
        sigmas = x.new_tensor(sigmas)
    if not (0.0 <= beta <= 1.0):
        raise ValueError("Parameter 'beta' must be between 0.0 and 1.0.")
    if not (0.0 <= eta_start <= 1.0) or not (0.0 <= eta_end <= 1.0):
        raise ValueError("eta_start and eta_end must be between 0.0 and 1.0.")

    extra_args = extra_args or {}
    noise_sampler = noise_sampler or default_noise_sampler(x)
    
    s_in = x.new_ones([x.shape[0]])
    
    old_denoised_for_beta = None
    
    actual_sigma_max = sigma_max_for_dyn_eta if sigma_max_for_dyn_eta is not None else sigmas[0].item()
    if actual_sigma_max <= 0:
        actual_sigma_max = 1.0 

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_current = sigmas[i]
        sigma_next_actual = sigmas[i+1]

        current_sigma_ratio = (sigma_current.item() / actual_sigma_max) ** eta_exponent
        current_sigma_ratio = max(0.0, min(1.0, current_sigma_ratio))
        current_eta = eta_end + (eta_start - eta_end) * current_sigma_ratio
        current_eta = max(0.0, min(1.0, current_eta))

        sigma_for_model = sigma_current * s_in
        denoised_at_current_sigma = model(x, sigma_for_model, **extra_args)

        if old_denoised_for_beta is not None and beta > 0.0 and sigma_next_actual < sigma_current:
            effective_x0_hat = (1 - beta) * denoised_at_current_sigma + beta * old_denoised_for_beta
        else:
            effective_x0_hat = denoised_at_current_sigma
        
        old_denoised_for_beta = denoised_at_current_sigma

        d = to_d(x, sigma_current, effective_x0_hat)
        
        sigma_down_float, sigma_up_float = get_ancestral_step(sigma_current.item(), sigma_next_actual.item(), eta=current_eta)
        sigma_down = x.new_tensor(sigma_down_float)
        sigma_up = x.new_tensor(sigma_up_float)
        
        if callback is not None:
            callback_dict = {
                'x': x, 'i': i, 'sigma': sigma_current, 'sigma_hat': sigma_current, 
                'denoised': denoised_at_current_sigma, 'current_eta': current_eta
            }
            if old_denoised_for_beta is not None and beta > 0.0 and 'effective_x0_hat' in locals() and effective_x0_hat is not denoised_at_current_sigma:
                 callback_dict['effective_denoised'] = effective_x0_hat
            callback(callback_dict)

        x = x + d * (sigma_down - sigma_current)

        if sigma_up > 0:
            added_noise = noise_sampler(sigma_current, sigma_next_actual)
            x = x + added_noise * s_noise * sigma_up
            
    return x

@torch.no_grad()
def sample_spawner_smea_dyn_beta1(model, x, sigmas,
                                 extra_args=None, callback=None, disable=None,
                                 eta_start=0.98,
                                 eta_end=0.65,
                                 eta_exponent=1.5,
                                 s_noise=1.0,
                                 noise_sampler=None,
                                 beta=0.55,
                                 sigma_max_for_dyn_eta=None
                                 ):
    print("Use Spawner SMEA Dyn beta")
    
    if not isinstance(sigmas, torch.Tensor):
        sigmas = x.new_tensor(sigmas)
    if not (0.0 <= beta <= 1.0):
        raise ValueError("Parameter 'beta' must be between 0.0 and 1.0.")
    if not (0.0 <= eta_start <= 1.0) or not (0.0 <= eta_end <= 1.0):
        raise ValueError("eta_start and eta_end must be between 0.0 and 1.0.")

    extra_args = extra_args or {}
    noise_sampler = noise_sampler or default_noise_sampler(x)
    
    s_in = x.new_ones([x.shape[0]])
    
    old_denoised_for_beta = None
    
    actual_sigma_max = sigma_max_for_dyn_eta if sigma_max_for_dyn_eta is not None else sigmas[0].item()
    if actual_sigma_max <= 0:
        actual_sigma_max = 1.0 

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_current = sigmas[i]
        sigma_next_actual = sigmas[i+1]

        current_sigma_ratio = (sigma_current.item() / actual_sigma_max) ** eta_exponent
        current_sigma_ratio = max(0.0, min(1.0, current_sigma_ratio))
        current_eta = eta_end + (eta_start - eta_end) * current_sigma_ratio
        current_eta = max(0.0, min(1.0, current_eta))

        sigma_for_model = sigma_current * s_in
        denoised_at_current_sigma = model(x, sigma_for_model, **extra_args)

        if old_denoised_for_beta is not None and beta > 0.0 and sigma_next_actual < sigma_current:
            effective_x0_hat = (1 - beta) * denoised_at_current_sigma + beta * old_denoised_for_beta
        else:
            effective_x0_hat = denoised_at_current_sigma
        
        old_denoised_for_beta = denoised_at_current_sigma

        d = to_d(x, sigma_current, effective_x0_hat)
        
        sigma_down_float, sigma_up_float = get_ancestral_step(sigma_current.item(), sigma_next_actual.item(), eta=current_eta)
        sigma_down = x.new_tensor(sigma_down_float)
        sigma_up = x.new_tensor(sigma_up_float)
        
        if callback is not None:
            callback_dict = {
                'x': x, 'i': i, 'sigma': sigma_current, 'sigma_hat': sigma_current, 
                'denoised': denoised_at_current_sigma, 'current_eta': current_eta
            }
            if old_denoised_for_beta is not None and beta > 0.0 and 'effective_x0_hat' in locals() and effective_x0_hat is not denoised_at_current_sigma:
                 callback_dict['effective_denoised'] = effective_x0_hat
            callback(callback_dict)

        x = x + d * (sigma_down - sigma_current)

        if sigma_up > 0:
            added_noise = noise_sampler(sigma_current, sigma_next_actual)
            x = x + added_noise * s_noise * sigma_up
            
    return x

# ---------------------下面这段在modules\sd_samplers_kdiffusion.py对应位置改(我用的forge，别的我不知道)--------------------------------------------
# samplers_k_diffusion = [
#     ('DPM++ 2M', 'sample_dpmpp_2m', ['k_dpmpp_2m'], {'scheduler': 'karras'}),
#     ('DPM++ SDE', 'sample_dpmpp_sde', ['k_dpmpp_sde'], {'scheduler': 'karras', "second_order": True, "brownian_noise": True}),
#     ('DPM++ 2M SDE', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde'], {'scheduler': 'exponential', "brownian_noise": True}),
#     ('DPM++ 2M SDE Heun', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_heun'], {'scheduler': 'exponential', "brownian_noise": True, "solver_type": "heun"}),
#     ('DPM++ 2S a', 'sample_dpmpp_2s_ancestral', ['k_dpmpp_2s_a'], {'scheduler': 'karras', "uses_ensd": True, "second_order": True}),
#     ('DPM++ 3M SDE', 'sample_dpmpp_3m_sde', ['k_dpmpp_3m_sde'], {'scheduler': 'exponential', 'discard_next_to_last_sigma': True, "brownian_noise": True}),
#     ('Euler a', 'sample_euler_ancestral', ['k_euler_a', 'k_euler_ancestral'], {"uses_ensd": True}),
#     ('Euler', 'sample_euler', ['k_euler'], {}),
#     ('LMS', 'sample_lms', ['k_lms'], {}),
#     ('Heun', 'sample_heun', ['k_heun'], {"second_order": True}),
#     ('DPM2', 'sample_dpm_2', ['k_dpm_2'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "second_order": True}),
#     ('DPM2 a', 'sample_dpm_2_ancestral', ['k_dpm_2_a'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
#     ('DPM fast', 'sample_dpm_fast', ['k_dpm_fast'], {"uses_ensd": True}),
#     ('DPM adaptive', 'sample_dpm_adaptive', ['k_dpm_ad'], {"uses_ensd": True}),
#     ('Restart', sd_samplers_extra.restart_sampler, ['restart'], {'scheduler': 'karras', "second_order": True}),
#     ('HeunPP2', 'sample_heunpp2', ['heunpp2'], {}),
#     ('IPNDM', 'sample_ipndm', ['ipndm'], {}),
#     ('IPNDM_V', 'sample_ipndm_v', ['ipndm_v'], {}),
#     ('DEIS', 'sample_deis', ['deis'], {}),
#     ('SMEA', 'sample_smea', ['k_smea'], {"uses_ensd": True}),
#     ('SMEA (beta)','sample_smea_beta',['k_smea_nai', 'smea_b'],{"uses_ensd": True}),
#     ('SMEA DYN (dyn_eta)','sample_smea_dyn_beta',['k_smea_dyn_nai', 'smea_dyn_e'],{"uses_ensd": True}),
# ]
# 相比原来加了    ('SMEA', 'sample_smea', ['k_smea'], {"uses_ensd": True}),
    #('SMEA (beta)','sample_smea_beta',['k_smea_nai', 'smea_b'],{"uses_ensd": True}),
    #('SMEA DYN (dyn_eta)','sample_smea_dyn_beta',['k_smea_dyn_nai', 'smea_dyn_e'],{"uses_ensd": True}),

# sampler_extra_params = {
#     'sample_euler': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
#     'sample_heun': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
#     'sample_dpm_2': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
#     'sample_dpm_fast': ['s_noise'],
#     'sample_dpm_2_ancestral': ['s_noise'],
#     'sample_dpmpp_2s_ancestral': ['s_noise'],
#     'sample_dpmpp_sde': ['s_noise'],
#     'sample_dpmpp_2m_sde': ['s_noise'],
#     'sample_dpmpp_3m_sde': ['s_noise'],
#     'sample_smea': ['s_noise', 'eta'],
#     'sample_smea_beta': ['eta', 's_noise', 'beta'],
#     'sample_smea_dyn_beta': ['eta_start', 'eta_end', 'eta_exponent', 's_noise', 'beta', 'sigma_max_for_dyn_eta'],
# }
# 相比原来加了    'sample_smea': ['s_noise', 'eta'],
    #'sample_smea_beta': ['eta', 's_noise', 'beta'],
    #'sample_smea_dyn_beta': ['eta_start', 'eta_end', 'eta_exponent', 's_noise', 'beta', 'sigma_max_for_dyn_eta'],
