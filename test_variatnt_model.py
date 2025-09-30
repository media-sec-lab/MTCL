import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from branch_decouple import decouple_model,SegFormer,SegFormer_Restore,SegFormer_Restore_Complex,SegFormer_Restore_Simple,SegFormer_Restore_returnDecoderFeatures




if __name__ == '__main__':
    model = SegFormer_Restore().cuda() # 这个是一个编码器，两个解码器（一个定位，一个复原的模型）

    print(model)