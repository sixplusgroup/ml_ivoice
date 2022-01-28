from sacred import Experiment
from Config import config_ingredient
import Evaluate
import os

#首先新建实验,并进行对应的模块化配置
ex = Experiment("WaveUNet", ingredients=[config_ingredient])


#在原有标签的基础上添加新的标签
'''
@ex.config 来实现的, 它在实验运行之前 (即定义阶段) 执行. 装饰器装饰的函数中所有受到赋值的局部变量会被搜集并整合为实验配置.
这里理想情况下是进行一个键盘的输入与输出作为选择，这里我们可以我试一下这边是可以获取到命令行输入的，现在的问题是想要将这样
'''
@ex.config
def cfg():
    # 这个地方组合的是模型参数文件相对路径，有对应的checkpoint文件夹，full——44的子文件夹，以及其他的路径
    test_model_path = os.path.join("checkpoints","full_44KHz", "full_44KHz-236118")
    # 这个地方存放的是需要分离的音乐的路径，这里可以修改成别的路径方法
    test_input_path = os.path.join("audio_examples","example1.mp3")
    #根据自己的路径进行一个修改
    our_input_path = input("Enter your input_path:")
    # 这里存放一个输出结果的参数
    test_output_path = "F:\\研究生\\iVoice实验结果"

#相关启动的操作
@ex.automain
def main(cfg, test_model_path, our_input_path, test_output_path):
    print("准备预测")
    print("当前的配置是：")
    print(cfg)
    model_config = cfg['model_config']
    Evaluate.produce_source_estimates(model_config, test_model_path, our_input_path, test_output_path)