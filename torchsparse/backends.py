import torch


def init():
    global benchmark, allow_tf32, allow_fp16, device_capability, hash_rsv_ratio
    benchmark = False
    hash_rsv_ratio = 2
    if not torch.cuda.is_available():
        device_capability = 0
        allow_tf32 = False
        allow_fp16 = False
        return
    device_capability = torch.cuda.get_device_capability()
    device_capability = device_capability[0] * 100 + device_capability[1] * 10
    allow_tf32 = device_capability >= 800
    allow_fp16 = device_capability >= 750
