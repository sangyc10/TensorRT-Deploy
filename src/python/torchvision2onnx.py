import torch
import torchvision
import onnxsim
import onnx
import argparse

def get_model(type, dir):
    if type == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        file  = dir + "resnet50.onnx"
    elif type == "resnet101":
        model = torchvision.models.resnet101(pretrained=True)
        file  = dir + "resnet101.onnx"
    elif type == "resnet152":
        model = torchvision.models.resnet152(pretrained=True)
        file  = dir + "resnet152.onnx"
    elif type == "vgg11":
        model = torchvision.models.vgg11(pretrained=True)
        file  = dir + "vgg11.onnx"
    elif type == "vgg19":
        model = torchvision.models.vgg19(pretrained=True)
        file  = dir + "vgg19.onnx"
    elif type == "mobilenet_v3_small":
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        file  = dir + "mobilenet_v3_small.onnx"
    elif type == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(pretrained=True)
        file  = dir + "efficientnet_b0.onnx"
    elif type == "efficientnet_v2_s":
        model = torchvision.models.efficientnet_v2_s(pretrained=True)
        file  = dir + "efficientnet_v2_s.onnx"
    return model, file

def export_norm_onnx(model, file, input):
    model.cuda()
    torch.onnx.export(
        model         = model, 
        args          = (input,),
        f             = file,
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 15)
    print("Finished normal onnx export")

    model_onnx = onnx.load(file)

    onnx.checker.check_model(model_onnx)

    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, file)


def main(args):
    type        = args.type
    dir         = args.dir
    input       = torch.rand(1, 3, 224, 224, device='cuda')
    model, file = get_model(type, dir)

    export_norm_onnx(model, file, input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, default="vgg11")
    parser.add_argument("-d", "--dir", type=str, default="./models/onnx/")
    
    opt = parser.parse_args()
    main(opt)
