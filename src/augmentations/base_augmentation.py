class BaseAugmentation:
    def __init__(self, config):
        self.config = config

    def generate(self, images, labels, save_dir):
        raise NotImplementedError
