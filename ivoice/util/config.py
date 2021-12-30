import copy

def override_config(configs, override_config_items):
  new_configs = copy.deepcopy(configs)
  for config_item in override_config_items:
    pairs = config_item.split()
    if len(pairs) != 2:
      print(
        '覆盖配置项格式有误, 中间以空格隔开'
      )
    keys = pairs[0].split('.')
    temp_configs = new_configs
    for i, key in enumerate(keys):
      if key not in temp_configs:
        print('配置项不存在')
      if i == len(keys) - 1:
        param_type = type(temp_configs[key])
        temp_configs[key] = param_type(pairs[1])
        print(f'override {pairs[0]} with {pairs[1]}')
      else:
        temp_configs = temp_configs[key]
    
  return new_configs