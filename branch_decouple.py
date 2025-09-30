import os
import torch
from torch import nn
from seg_former_backup import mit_b5,SegFormerHead
import os
from functools import partial

pretrained_path = "ckpts/mit_b5.pth"
resize = torch.nn.functional.interpolate



class SegFormer(nn.Module):
	def __init__(self):
		super(SegFormer,self).__init__()
		self.seg_extrator = mit_b5() #embed_dims=[64, 128, 320, 512]
		self.init_weight(pretrained_path)
		self.decoder = SegFormerHead(feature_strides=[4, 8, 16, 32], in_channels=[64, 128, 320, 512],
                                         decoder_params=dict(embed_dim=768), num_classes=1,return_midlle = True)
	def forward(self,x):
		seg_former_features = self.seg_extrator(x)
		# decoder_output = self.decoder(seg_former_features)

		decoder_output,decoder_features = self.decoder(seg_former_features)

		x_loc = resize(
			input=decoder_output,
			size=x.shape[2:],
			mode='bilinear',
			align_corners=False)
		x_loc = torch.sigmoid(x_loc)
		# return x_loc
		return x_loc,seg_former_features,decoder_features
		# return x_loc,seg_former_features

	def init_weight(self,pretrained):
		pretrained_dict = torch.load(pretrained)
		model_dict = {}
		state_dict = self.seg_extrator.state_dict()
		# print("Model:")
		# for k,v in state_dict.items():
		# 	print(k)
		# print("Model pretrained:")
		for k, v in pretrained_dict.items():
			# print(k)
			# k = 'encoder.' + k
			if k in state_dict:
				model_dict[k] = v
		# print(model_dict)
		state_dict.update(model_dict)
		self.seg_extrator.load_state_dict(state_dict)


class SegFormer_Restore_returnDecoderFeatures(nn.Module): #分成两个阶段进行训练
	def __init__(self):
		super(SegFormer_Restore_returnDecoderFeatures,self).__init__()
		self.seg_extrator = mit_b5() #embed_dims=[64, 128, 320, 512]
		self.init_weight(pretrained_path)
		self.decoder = SegFormerHead(feature_strides=[4, 8, 16, 32], in_channels=[64, 128, 320, 512],
                                         decoder_params=dict(embed_dim=768), num_classes=1,return_midlle = True)
		self.decoder_restore = SegFormerHead(feature_strides=[4, 8, 16, 32], in_channels=[64, 128, 320, 512],
		                             decoder_params=dict(embed_dim=768), num_classes=3)
	def forward(self,x):
		seg_former_features = self.seg_extrator(x)

		decoder_output,decoder_features = self.decoder(seg_former_features)

		x_loc = resize(
			input=decoder_output,
			size=x.shape[2:],
			mode='bilinear',
			align_corners=False)
		x_loc = torch.sigmoid(x_loc)
		restore_output = self.decoder_restore(seg_former_features)
		restore_output = resize(input = restore_output,size = x.shape[2:],mode = 'bilinear',align_corners = False)
		return x_loc,restore_output,seg_former_features,decoder_features

	def init_weight(self,pretrained):
		pretrained_dict = torch.load(pretrained)
		model_dict = {}
		state_dict = self.seg_extrator.state_dict()
		# print("Model:")
		# for k,v in state_dict.items():
		# 	print(k)
		# print("Model pretrained:")
		for k, v in pretrained_dict.items():
			# print(k)
			# k = 'encoder.' + k
			if k in state_dict:
				model_dict[k] = v
		# print(model_dict)
		state_dict.update(model_dict)
		self.seg_extrator.load_state_dict(state_dict)



