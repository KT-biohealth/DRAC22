# Copyright (c) OpenMMLab. All rights reserved.

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DRAC2022SegDataset(CustomDataset):
    """DRIVE dataset.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """

    CLASSES = ('background', 'IMA', 'NA', 'N') #IMA = Intraretinal Microvascular Abnormalities, NA = Nonperfusion Areas, N = Neovascularization

    PALETTE = [[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(DRAC2022SegDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
