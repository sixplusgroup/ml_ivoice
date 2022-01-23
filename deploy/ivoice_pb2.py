# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ivoice.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='ivoice.proto',
  package='ivoice',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0civoice.proto\x12\x06ivoice\"/\n\rResultContent\x12\x0f\n\x07\x63ontent\x18\x01 \x01(\t\x12\r\n\x05\x63ount\x18\x02 \x01(\x05\"\"\n\x0eResultKeywords\x12\x10\n\x08keywords\x18\x01 \x01(\t\"%\n\x07Segment\x12\r\n\x05start\x18\x01 \x01(\x02\x12\x0b\n\x03\x65nd\x18\x02 \x01(\x02\"+\n\x11TranscribeRequest\x12\x16\n\x0eremoteFilePath\x18\x01 \x01(\t\"S\n\x12TranscribeResponse\x12 \n\x07segment\x18\x01 \x01(\x0b\x32\x0f.ivoice.Segment\x12\r\n\x05label\x18\x02 \x01(\x05\x12\x0c\n\x04word\x18\x03 \x01(\t2\xa5\x01\n\rIVoiceToolkit\x12P\n\x13transcribeAudioFile\x12\x19.ivoice.TranscribeRequest\x1a\x1a.ivoice.TranscribeResponse\"\x00\x30\x01\x12\x42\n\x0f\x65xtractKeywords\x12\x15.ivoice.ResultContent\x1a\x16.ivoice.ResultKeywords\"\x00\x62\x06proto3'
)




_RESULTCONTENT = _descriptor.Descriptor(
  name='ResultContent',
  full_name='ivoice.ResultContent',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='content', full_name='ivoice.ResultContent.content', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='count', full_name='ivoice.ResultContent.count', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=24,
  serialized_end=71,
)


_RESULTKEYWORDS = _descriptor.Descriptor(
  name='ResultKeywords',
  full_name='ivoice.ResultKeywords',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='keywords', full_name='ivoice.ResultKeywords.keywords', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=73,
  serialized_end=107,
)


_SEGMENT = _descriptor.Descriptor(
  name='Segment',
  full_name='ivoice.Segment',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='start', full_name='ivoice.Segment.start', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='end', full_name='ivoice.Segment.end', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=109,
  serialized_end=146,
)


_TRANSCRIBEREQUEST = _descriptor.Descriptor(
  name='TranscribeRequest',
  full_name='ivoice.TranscribeRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='remoteFilePath', full_name='ivoice.TranscribeRequest.remoteFilePath', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=148,
  serialized_end=191,
)


_TRANSCRIBERESPONSE = _descriptor.Descriptor(
  name='TranscribeResponse',
  full_name='ivoice.TranscribeResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='segment', full_name='ivoice.TranscribeResponse.segment', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='label', full_name='ivoice.TranscribeResponse.label', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='word', full_name='ivoice.TranscribeResponse.word', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=193,
  serialized_end=276,
)

_TRANSCRIBERESPONSE.fields_by_name['segment'].message_type = _SEGMENT
DESCRIPTOR.message_types_by_name['ResultContent'] = _RESULTCONTENT
DESCRIPTOR.message_types_by_name['ResultKeywords'] = _RESULTKEYWORDS
DESCRIPTOR.message_types_by_name['Segment'] = _SEGMENT
DESCRIPTOR.message_types_by_name['TranscribeRequest'] = _TRANSCRIBEREQUEST
DESCRIPTOR.message_types_by_name['TranscribeResponse'] = _TRANSCRIBERESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ResultContent = _reflection.GeneratedProtocolMessageType('ResultContent', (_message.Message,), {
  'DESCRIPTOR' : _RESULTCONTENT,
  '__module__' : 'ivoice_pb2'
  # @@protoc_insertion_point(class_scope:ivoice.ResultContent)
  })
_sym_db.RegisterMessage(ResultContent)

ResultKeywords = _reflection.GeneratedProtocolMessageType('ResultKeywords', (_message.Message,), {
  'DESCRIPTOR' : _RESULTKEYWORDS,
  '__module__' : 'ivoice_pb2'
  # @@protoc_insertion_point(class_scope:ivoice.ResultKeywords)
  })
_sym_db.RegisterMessage(ResultKeywords)

Segment = _reflection.GeneratedProtocolMessageType('Segment', (_message.Message,), {
  'DESCRIPTOR' : _SEGMENT,
  '__module__' : 'ivoice_pb2'
  # @@protoc_insertion_point(class_scope:ivoice.Segment)
  })
_sym_db.RegisterMessage(Segment)

TranscribeRequest = _reflection.GeneratedProtocolMessageType('TranscribeRequest', (_message.Message,), {
  'DESCRIPTOR' : _TRANSCRIBEREQUEST,
  '__module__' : 'ivoice_pb2'
  # @@protoc_insertion_point(class_scope:ivoice.TranscribeRequest)
  })
_sym_db.RegisterMessage(TranscribeRequest)

TranscribeResponse = _reflection.GeneratedProtocolMessageType('TranscribeResponse', (_message.Message,), {
  'DESCRIPTOR' : _TRANSCRIBERESPONSE,
  '__module__' : 'ivoice_pb2'
  # @@protoc_insertion_point(class_scope:ivoice.TranscribeResponse)
  })
_sym_db.RegisterMessage(TranscribeResponse)



_IVOICETOOLKIT = _descriptor.ServiceDescriptor(
  name='IVoiceToolkit',
  full_name='ivoice.IVoiceToolkit',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=279,
  serialized_end=444,
  methods=[
  _descriptor.MethodDescriptor(
    name='transcribeAudioFile',
    full_name='ivoice.IVoiceToolkit.transcribeAudioFile',
    index=0,
    containing_service=None,
    input_type=_TRANSCRIBEREQUEST,
    output_type=_TRANSCRIBERESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='extractKeywords',
    full_name='ivoice.IVoiceToolkit.extractKeywords',
    index=1,
    containing_service=None,
    input_type=_RESULTCONTENT,
    output_type=_RESULTKEYWORDS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_IVOICETOOLKIT)

DESCRIPTOR.services_by_name['IVoiceToolkit'] = _IVOICETOOLKIT

# @@protoc_insertion_point(module_scope)
