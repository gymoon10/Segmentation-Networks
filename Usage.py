

from models.unet import UNet
from models.HRNet import HighResolutionNet
from models.efficient_vit import VIT_Model
from models.segnext2 import SegNext
from models.segnextv2 import SegNeXt_L, SegNeXt_B, SegNeXt_S
from models.saunet import SA_UNet
from models.transattunet import UNet_Attention_Transformer_Multiscale
from models.dpt import DPTSegmentationModel
from models.subpixelembedding import SPIN, SPiNModel
from models.cmunet import CMUNet
from models.mfnet import MFNet
from models.mcinet import Mcinet
from models.setr import SETR
from models.paratranscnn import ParaTransCNN
from models.lawin import Lawin
from models.sfnet import SFNet
from models.ddrnet import DDRNet
from models.MANet import UNet2D
from models.BiSeNetV2 import BiSeNetV2
from models.UCTransNet import UCTransNet
from models.GAUNet import GAUNet



def get_model(name, model_opts):

    if name == "EfficientVit":
        model = VIT_Model()
        return model

    elif name == "SegNext":
        model = SegNext()
        return model

    elif name == "SegNextV2":
        model = SegNeXt_B(num_classes=3)
        return model

    elif name == "SAUNet":
        model = SA_UNet()
        return model

    elif name == "TransAttUNet":
        model = UNet_Attention_Transformer_Multiscale(3, 3)
        return model

    elif name == "DPT":
        model = DPTSegmentationModel(num_classes=3)
        return model

    elif name == "Subpixel":
        model = SPIN()
        return model

    elif name == "CMUNet":
        model = CMUNet(img_ch=3, output_ch=3)
        return model

    elif name == "UNet":
        model = UNet()
        return model

    elif name == "HRNet":
        model = HighResolutionNet(num_classes=3)
        return model

    elif name == "MFNet":
        model = MFNet(classes=3)
        return model

    elif name == "MCINet":
        model = Mcinet()
        return model

    elif name == "SETR":
        model = SETR()
        return model

    elif name == "ParaTransCNN":
        model = ParaTransCNN(num_classes=3)
        return model

    elif name == "Lawin":
        model = Lawin('MiT-B1', 3)
        return model

    elif name == "SFNet":
        model = SFNet('ResNetD-18', 3)
        return model

    elif name == "DDRNet":
        model = DDRNet(num_classes=3)
        return model

    elif name == "BiSeNetV2":
        model = BiSeNetV2(3)
        return model

    elif name == "UCTransNet":
        model = UCTransNet(n_channels=3, n_classes=3,img_size=512)
        return model

    elif name == "GAUNet":
        model = GAUNet()
        return model

    elif name == "MANet":
        model = UNet2D(3, 3)
        return model

    else:
        raise RuntimeError("model \"{}\" not available".format(name))
