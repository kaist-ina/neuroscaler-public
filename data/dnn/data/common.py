class Configuration():
    def __init__(is_train, patch_size, load_on_memory, samples_per_epoch=None):
        self.is_train = is_train
        self.patch_size = patch_size
        self.load_on_memory = load_on_memory
        self.samples_per_epoch = samples_per_epoch
