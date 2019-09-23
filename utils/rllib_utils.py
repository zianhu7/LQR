

def merge_config(config, args):
    """Takes a config and overrides the key-value pair if there is a matching key in args"""
    for key in args.__dict__:
        if key in config:
            config[key] = args.__dict__[key]
    return config

def merge_dicts(base_dict, override_dict):
    """Takes an override dict and replaces all the matching keys in base_dict with the values in override_dict"""
    for key in override_dict:
        if key in base_dict:
            base_dict[key] = override_dict[key]
    return base_dict