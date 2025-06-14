import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T

class FSRCNN(nn.Module):
    def __init__(self, scale_factor=5):
        super(FSRCNN, self).__init__()
        d = 56
        s = 12
        m = 4

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=5, padding=2),
            nn.PReLU()
        )
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU()
        )
        self.mapping = nn.Sequential(*[
            nn.Conv2d(s, s, kernel_size=3, padding=1),
            nn.PReLU()
        ] * m)
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU()
        )
        self.deconv = nn.ConvTranspose2d(d, 1, kernel_size=9, stride=scale_factor, padding=3, output_padding=scale_factor-1)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.deconv(x)
        return x

def enhance_image(img_path):
    model = FSRCNN(scale_factor=5)
    model.load_state_dict(torch.load("models/fsrcnn.pth", map_location="cpu"))
    model.eval()

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = T.ToTensor()(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img).squeeze().numpy()

    output = (output * 255).clip(0, 255).astype("uint8")
    return output
