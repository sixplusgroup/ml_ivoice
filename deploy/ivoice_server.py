from ftplib import FTP
from typing import Dict
from ivoice.approach.audio_transcribe.audio_transcript import transcribe
from concurrent import futures
import yaml
import os
import ivoice_pb2_grpc
import ivoice_pb2
import grpc

def read_yaml_conf(config_path = './conf.yaml'):
  with open(config_path, 'r', encoding='UTF-8') as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)
  print('config ', config);
  return config

def set_ftp_client(config: Dict):
  ftp_config = config['FTP']
  host = ftp_config['host']
  port = ftp_config['port']
  user = ftp_config['user']
  password = ftp_config['password']
  ftp = FTP(host=host, user=user, passwd=password)
  return ftp;

def download_file(ftp: FTP, remote_path, local_path):
  buffer_size = 1024
  with open(local_path, 'wb') as file:
    ftp.retrbinary('RETR ' + remote_path, file.write, buffer_size)
    file.close()


def delete_file():
  pass;


class AudioTranscribeServicer(ivoice_pb2_grpc.IVoiceToolkitServicer):
  def transcribeAudioFile(self, request, context):
    config = read_yaml_conf()
    ftp = set_ftp_client(config)
    remote_path = request.remoteFilePath
    local_path = os.path.join('../temp', remote_path.split('/')[-1])
    download_file(ftp, remote_path, local_path)
    result = transcribe(local_path)
    for pair in result:
      segment_result = ivoice_pb2.SegmentResult(
        segment = ivoice_pb2.Segment(
          start = pair['segment'].start,
          end = pair['segment'].end,
        ),
        label = pair['label'],
        word = pair['word']
      )
      yield segment_result

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  ivoice_pb2_grpc.add_IVoiceToolkitServicer_to_server(AudioTranscribeServicer(), server)
  server.add_insecure_port('[::]:9000')
  server.start()
  server.wait_for_termination()
  print('server initialized successfully')

if __name__ == '__main__':
  print('server start')
  serve()


