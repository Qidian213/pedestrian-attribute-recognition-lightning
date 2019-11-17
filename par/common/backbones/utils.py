

def load_state_dict(model, state_dict):
    model_sd = model.state_dict()
    pretrained_sd = {}
    diff = []
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].size() == v.size():
            pretrained_sd[k] = v
        else:
            diff += [k]

    model_sd.update(pretrained_sd)
    model.load_state_dict(model_sd)
    print(
        "=> Loaded ImageNet pretrained weights. "
        f"\n=> Ignored parameters from state_dict: {diff}"
    )
