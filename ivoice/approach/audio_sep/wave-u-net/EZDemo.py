# #encoding:UTF-8
# from flask import Flask,request,render_template,redirect,url_for
# from werkzeug.utils import secure_filename
# from flask_cors import CORS
# import Evaluate
# import os
#
# from Predict import main,cfg,input_path1
# from Config import cfg as model_cfg_sum
# app = Flask(__name__)
# CORS(app, supports_credentials=True)
# model_config: {'musdb_path': 'E:\\音乐分离系统相关\\数据集',
#                  'estimates_path': '/mnt/windaten/Source_Estimates',
#                  'data_path': 'data',
#                  'model_base_dir': 'checkpoints',
#                  'log_dir': 'logs',
#                  'batch_size': 32,
#                  'init_sup_sep_lr': 1e-05,
#                  'epoch_it': 2000,
#                  'cache_size': 4000,
#                  'num_workers': 4,
#                  'num_snippets_per_track': 100,
#                  'num_layers': 12,
#                  'filter_size': 15,
#                  'merge_filter_size': 5,
#                  'input_filter_size': 15,
#                  'output_filter_size': 1,
#                  'num_initial_filters': 24,
#                  'num_frames': 16384,
#                  'expected_sr': 44100,
#                  'mono_downmix': False,
#                  'output_type': 'difference',
#                  'output_activation': 'tanh',
#                  'context': True,
#                  'network': 'unet',
#                  'upsampling': 'learned',
#                  'task': 'voice',
#                  'augmentation': True,
#                  'raw_audio_loss': True,
#                  'worse_epochs': 20,
#                  'source_names': ['accompaniment', 'vocals'], 'num_sources': 2, 'num_channels': 2}
#
# @app.route("/event/up", methods=["POST"])
# def hello_world():
#     print('到啦')
#     f = request.files.get("file")
#     print(f)
#     print(f.name)
#     basepath = os.path.dirname(__file__)  # 当前文件所在路径
#     upload_path = os.path.abspath(os.path.join(basepath,"testdata",f.name))
#     print(upload_path)
#     #文件保存
#     f.save(upload_path)
#     print("保存成功")
#     #模型载入,修改传入的参数,但是这个里耦合性很高，照理来说应该使用内部方法访问参数
#     model_path = os.path.join("checkpoints", "full_44KHz", "full_44KHz-236118")  # Load stereo vocal model by default
#     output_path = "C:\\Users\\MSI-PC\\Desktop\\研究生\\记录\\结果"  # Where to save results. Default: Same location as input.
#     Evaluate.produce_source_estimates(model_config, model_path, upload_path, output_path)
#
#     print("分离完成")
#     return "上传完毕"
#
#
# @app.route('/upload',methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         f=request.files['file']
#         basepath = os.path.dirname(__file__) #当前文件所在路径
#         upload_path = os.path.join(basepath, 'record\\',secure_filename(f.filename))
#         f.save(upload_path)
#         return redirect(url_for('upload'))
#     return "错误"
#
# if __name__ == '__main__':
#     app.run()

