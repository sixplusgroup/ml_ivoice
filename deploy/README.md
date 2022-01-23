# 部署相关
更新proto后重新生成server和client命令：

`python -m grpc_tools.protoc -I./deploy/protos --python_out=./deploy --grpc_python_out=./deploy ./deploy/protos/ivoice.proto
`(在项目根目录下执行)