import torch
from model_arch import SRModel
pre_dict = torch.load("./PANx4_DF2K.pth")



model = SRModel(3, 40, 24, 3, 16)
model_dict = model.state_dict()
for name, param in model.named_parameters():
    print(name)
    print(param.shape)

model_dict['conv_embed.weight'] = pre_dict['conv_first.weight']
model_dict['conv_embed.bias'] = pre_dict['conv_first.bias']
for i in range(16):
    model_dict['sc_blocks.{}.conv_upper.weight'.format(i)] = pre_dict['SCPA_trunk.{}.conv1_b.weight'.format(i)]
    model_dict['sc_blocks.{}.conv_lower.weight'.format(i)] = pre_dict['SCPA_trunk.{}.conv1_a.weight'.format(i)]
    model_dict['sc_blocks.{}.att_conv.weight'.format(i)] = pre_dict['SCPA_trunk.{}.PAConv.k2.weight'.format(i)]
    model_dict['sc_blocks.{}.att_conv.bias'.format(i)] = pre_dict['SCPA_trunk.{}.PAConv.k2.bias'.format(i)]
    model_dict['sc_blocks.{}.conv_upper1.weight'.format(i)] = pre_dict['SCPA_trunk.{}.PAConv.k3.weight'.format(i)]
    model_dict['sc_blocks.{}.conv_upper2.weight'.format(i)] = pre_dict['SCPA_trunk.{}.PAConv.k4.weight'.format(i)]
    model_dict['sc_blocks.{}.conv_lower1.weight'.format(i)] = pre_dict['SCPA_trunk.{}.k1.0.weight'.format(i)]
    model_dict['sc_blocks.{}.conv_out.weight'.format(i)] = pre_dict['SCPA_trunk.{}.conv3.weight'.format(i)]
model_dict['conv_after_sc.weight'] = pre_dict['trunk_conv.weight']
model_dict['conv_after_sc.bias'] = pre_dict['trunk_conv.bias']
model_dict['sr_block.up_conv1.weight'] = pre_dict['upconv1.weight']
model_dict['sr_block.up_conv1.bias'] = pre_dict['upconv1.bias']
model_dict['sr_block.pa1.conv_att.weight'] = pre_dict['att1.conv.weight']
model_dict['sr_block.pa1.conv_att.bias'] = pre_dict['att1.conv.bias']
model_dict['sr_block.up_conv2.weight'] = pre_dict['HRconv1.weight']
model_dict['sr_block.up_conv2.bias'] = pre_dict['HRconv1.bias']
model_dict['sr_block.up_conv3.weight'] = pre_dict['upconv2.weight']
model_dict['sr_block.up_conv3.bias'] = pre_dict['upconv2.bias']
model_dict['sr_block.pa2.conv_att.weight'] = pre_dict['att2.conv.weight']
model_dict['sr_block.pa2.conv_att.bias'] = pre_dict['att2.conv.bias']
model_dict['sr_block.up_conv4.weight'] = pre_dict['HRconv2.weight']
model_dict['sr_block.up_conv4.bias'] = pre_dict['HRconv2.bias']
model_dict['conv_out.weight'] = pre_dict['conv_last.weight']
model_dict['conv_out.bias'] = pre_dict['conv_last.bias']

model.load_state_dict(model_dict)
torch.save(model.state_dict(), 'pre.pth')