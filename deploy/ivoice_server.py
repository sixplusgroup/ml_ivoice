from ftplib import FTP
from typing import Dict
from ivoice.approach.audio_transcribe.audio_transcript import transcribe
from ivoice.approach.keywords_extraction.keywords_with_textrank import extract_keywords
from concurrent import futures
import yaml
import os
import ivoice_pb2_grpc
import ivoice_pb2
import grpc
import json

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
  return ftp

def download_file(ftp: FTP, remote_path, local_path):
  buffer_size = 1024
  with open(local_path, 'wb') as file:
    print('start downloading file')
    print(ftp.dir())
    # ftp.retrbinary('RETR ' + remote_path, file.write, buffer_size)
    # ftp.retrbinary('RETR ' + 'upload/siri-rc-upload-1642410971426-2.wav', file.write, buffer_size)
    ftp.retrbinary('RETR ' + 'ivoicesiri.wav', file.write, buffer_size)
    file.close()
    ftp.close()


def delete_file():
  pass;


class AudioTranscribeServicer(ivoice_pb2_grpc.IVoiceToolkitServicer):
  def transcribeAudioFile(self, request, context):
    config = read_yaml_conf()
    ftp = set_ftp_client(config)
    remote_path = request.remoteFilePath
    local_path = os.path.join('../temp', remote_path)
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

    print('transcribe end')

  def extractKeywords(self, request, context):
    content = request.content
    count = request.count
    top_keywords = extract_keywords(content, count)
    return ivoice_pb2.ResultKeywords(
      keywords=json.dumps(top_keywords)
    )


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


