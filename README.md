# MTCL
This is the implementation of the method proposed in "Towards generalizable and robust image tampering localization with multi-task learning and contrastive learning".
## Test with proposed model 
1. Create conda environment and install the requirements.
```
conda create -n MTCL python==3.7.11
conda activate MTCL
pip install -r requirements.txt
```
2. Download the model from [Baiduyun](https://pan.baidu.com/s/1PCEFJ9mzOcwIghLuLcSh2w?pwd=MTCL) or [Google Drive](https://drive.google.com/drive/folders/1LklFg3iQq7Cf0I5YGmeyw29xpPp0PNCc?usp=sharing), and put it in the folder `./ckpts/`.

3. Modify the path of dataset(test_image_path, test_mask_path, and test_dataset) in main_test.py to your own dataset path.

4. Run the following command to test the model:

```bash
python main_test.py
```

5. The results will be saved in the folder `./results/`.

## Train with proposed model
1. Download the training dataset mentioned in our paper.

2. Modify the path of dataset( `args.dataset_root_path `) in main_train.py to your own dataset path.

3. Run the following command to train the model:

```bash
python main_train.py
```

4. The results will be saved in the folder `args.save_model_path`.

## Citation
If you find this code useful in your research, please consider citing our paper:
```
@article{li2025towards,
  title={Towards generalizable and robust image tampering localization with multi-task learning and contrastive learning},
  author={Li, Haodong and Zhuang, Peiyu and Su, Yang and Huang, Jiwu},
  journal={Expert Systems with Applications},
  volume={270},
  pages={126492},
  year={2025},
  publisher={Elsevier}
}
```