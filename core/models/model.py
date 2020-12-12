import importlib
import torch


def get_model(model_config):
    def _model_class(class_name):
        m = importlib.import_module('core.models')
        clazz = getattr(m, class_name)
        return clazz

    # assert 'model' in config, 'Could not find model configuration'

    model_class = _model_class(model_config['name'])
    return model_class(**model_config)


if __name__ == '__main__':
    a = torch.zeros((1, 1, 64, 64, 64))
    model = get_model({'name': 'StackHourglass', 'nstack': 2, 'inp_dim': 32, 'oup_dim': 17, 'conv_layer_order': 'gcr'})
    b = model(a)
    print(b.shape)
