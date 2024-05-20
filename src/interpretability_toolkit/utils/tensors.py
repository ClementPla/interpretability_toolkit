def normalize(tensor):
    """
    Normalize a tensor to have a mean of 0 and a standard deviation of 1.
    """

    tensor = tensor.float()
    min_tensor = tensor.amin((-2, -1), keepdim=True)
    max_tensor = tensor.amax((-2, -1), keepdim=True)
    tensor = (tensor - min_tensor) / (max_tensor - min_tensor + 1e-7)
    return tensor

def get_resnet_layer(resnet):
    """
    Get the layers of a ResNet model.
    """
    i = 1
    layer = None
    while True:
        if hasattr(resnet, f'layer{i}'):
            layer = getattr(resnet, f'layer{i}')
            i += 1
        else:
            break
    return layer

def get_last_layer(model):
    match model.__class__.__name__:
        case 'ConvNeXt':
            layer = model.stages[-1]
        case 'SwinTransformerV2':
            layer = None
        case 'ByobNet':
            layer = None
        case 'ResNet':
            layer = get_resnet_layer(model)
        case 'EfficientNet':
            layer = model.blocks[-1]
        case 'VGG':
            layer = model.features
        case 'VisionTransformer':
            layer = model.norm
        case 'MobileNetV3':
            layer = model.conv_head
    
    return layer


    