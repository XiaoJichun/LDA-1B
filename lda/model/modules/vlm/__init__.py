


def _infer_vlm_type(vlm_name: str) -> str:
    """Infer VLM type from path name or config.json for local directories."""
    import os, json
    if "Qwen2.5-VL" in vlm_name or "nora" in vlm_name.lower():
        return "qwen2.5-vl"
    if "Qwen3-VL" in vlm_name:
        return "qwen3-vl"
    if "florence" in vlm_name.lower():
        return "florence"
    # fallback: read config.json for local directory
    if os.path.isdir(vlm_name):
        cfg_path = os.path.join(vlm_name, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                model_type = json.load(f).get("model_type", "")
            if "qwen3_vl" in model_type:
                return "qwen3-vl"
            if "qwen2_5_vl" in model_type or "qwen2.5_vl" in model_type:
                return "qwen2.5-vl"
    return "unknown"


def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.base_vlm
    vlm_type = _infer_vlm_type(vlm_name)

    if vlm_type == "qwen2.5-vl":
        from .QWen2_5 import _QWen_VL_Interface
        return _QWen_VL_Interface(config)
    elif vlm_type == "qwen3-vl":
        from .QWen3 import _QWen3_VL_Interface
        return _QWen3_VL_Interface(config)
    elif vlm_type == "florence":
        from .Florence2 import _Florence_Interface
        return _Florence_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")



