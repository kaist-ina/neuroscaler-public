from .edsr import EDSR
from .nemo import NEMO

def build(args):
    if args.model_name == 'edsr':
        model = EDSR(args)
    elif args.model_name == 'nemo':
        model = NEMO(args)
    else:
        raise NotImplementedError

    return model
