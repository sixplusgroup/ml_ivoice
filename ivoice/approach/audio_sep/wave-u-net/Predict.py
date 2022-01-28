from sacred import Experiment
from Config import config_ingredient
import Evaluate
import os

ex = Experiment('Waveunet Prediction', ingredients=[config_ingredient])
# model_path1 = os.path.join("checkpoints", "full_44KHz", "full_44KHz-236118")  # Load stereo vocal model by default
input_path1 = os.path.join("audio_examples", "3.mp3") # Which audio file to separate
# output_path1 = "C:\\Users\\MSI-PC\\Desktop\\研究生\\记录\\结果"  # Where to save results. Default: Same location as input.

#这个标签意味着可以加别的一些配置，比如说，之前config中只有a，b,c的取值，但是现在我们需要d的取值，我们重新新建一个config然后加入我们想加入的
@ex.config
def cfg():
    #这里加入存储的训练好的模型的参数，以及他输入输出的目标文件夹，如果我们需要进行修改，那么需要修改就是这里，修改他的相关路径和

    model_path = os.path.join("checkpoints", "full_44KHz", "full_44KHz-236118") # Load stereo vocal model by default

    input_path = os.path.join("audio_examples", "3.mp3") # Which audio file to separate
    output_path = "C:\\Users\\MSI-PC\\Desktop\\研究生\\记录\\结果" # Where to save results. Default: Same location as input.

@ex.automain
def main(cfg, model_path, input_path, output_path):
    print("准备预测")
    print(cfg)
    model_config = cfg["model_config"]
    print(model_config)
    Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path)
