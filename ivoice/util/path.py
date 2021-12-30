import os

# 获取项目根目录
def get_project_dir():
  dir_path = os.path.dirname(__file__)
  root_path = os.path.dirname(dir_path)
  return os.path.abspath(root_path)