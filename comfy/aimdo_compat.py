try:
    import comfy_aimdo.control as control
    import comfy_aimdo.model_vbar as model_vbar
    import comfy_aimdo.host_buffer as host_buffer
    import comfy_aimdo.vram_buffer as vram_buffer
    import comfy_aimdo.torch as torch
    AIMDO_AVAILABLE = True
except ImportError:
    control = None
    model_vbar = None
    host_buffer = None
    vram_buffer = None
    torch = None
    AIMDO_AVAILABLE = False
