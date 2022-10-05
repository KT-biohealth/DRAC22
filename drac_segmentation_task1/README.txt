reference : https://github.com/open-mmlab/mmsegmentation

1. Pre-processing (Gray -> Color)
 cd preprocessing
 python image_coloring.py
 
2. Training
 SegFormer : python train.py configs/segformer/segformer_mit-b5_80k_drac2022seg.py
 Convnext : python train.py configs/convnext/upernet_convnext_xlarge_80k_drac2022seg.py
 SwinTransformer : python train.py configs/swin/upernet_swin_large_80k_drac2022seg.py
 
3. Test
 ex) python test.py configs/segformer/segformer_mit-b5_80k_drac2022seg.py work_dirs/segformer_mit-b5_80k_drac2022seg/iter_80000.pth --eval mDice --show-dir result/segformer_mit-b5_80k_drac2022seg
 
4. Post-processing (models ensemble)
 cd postprocessing
 Class 1 : python ensemble_cls1.py (edit results path)
 Class 2 : python ensemble_cls2.py (edit results path)
 Class 3 : python ensemble_cls3.py (edit results path)
 