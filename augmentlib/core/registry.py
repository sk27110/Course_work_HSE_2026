AUGMENTATION_REGISTRY = {}


def register_augmentation(name):
    def decorator(cls):
        AUGMENTATION_REGISTRY[name] = cls
        return cls
    return decorator


def get_augmentation(name, **kwargs):
    if name not in AUGMENTATION_REGISTRY:
        raise ValueError(f"Unknown augmentation: {name}")
    return AUGMENTATION_REGISTRY[name](**kwargs)