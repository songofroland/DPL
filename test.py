import torch
from model import RGATABSA
from utils.arg import arg_generate

parser = arg_generate()
args = parser.parse_args()

best_path = "pre_model/rest/pytorch_model.bin"
trainer = torch.load(best_path)
model = RGATABSA(args)
model.load_state_dict(trainer)
