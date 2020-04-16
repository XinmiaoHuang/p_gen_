from base.base_parser import BaseParser


class CustomParser(BaseParser):
    def __init__(self):
        super(CustomParser, self).__init__()
        self.parser.add_argument("--base_dir", type=str, default='D:/Dataset/deepfashion')
        self.parser.add_argument("--index_dir", type=str, default='D:/Dataset/deepfashion/index.p')
        self.parser.add_argument("--bilinear", type=bool, default=True)
        self.parser.add_argument("--semantic_channels", type=int, default=3)
        self.parser.add_argument("--lambda_kl", type=float, default=1e-6)
        self.parser.add_argument("--lambda_perceptual", type=float, default=10)
        self.parser.add_argument("--img_height", type=int, default=256)
        self.parser.add_argument("--img_width", type=int, default=256)
        self.parser.add_argument("--n_channels", type=int, default=6)
        self.parser.add_argument("--pose_channels", type=int, default=19)
        self.parser.add_argument("--dnet_inputc", type=int, default=3)
        self.parser.add_argument("--n_classes", type=int, default=3)
        self.parser.add_argument("--learning_rate", type=float, default=0.0002)

    def parse(self):
        return self.parser.parse_args()