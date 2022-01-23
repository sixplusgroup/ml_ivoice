import torch
import re
import logging
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
import numpy as np
from plane.pattern import EMAIL, TELEPHONE
from ivoice.util.constant import (
  NUMBER,
  CURRENCY,
  URL,
  EMAIL_TOKEN,
  URL_TOKEN,
  CURRENCY_TOKEN,
  TELEPHONE_TOKEN,
  NUMBER_TOKEN
)
from ivoice.util.process import chinese_split, is_ascii

logger = logging.getLogger(__name__)
num_regex = re.compile(f"{NUMBER.pattern}")
tel_regex = re.compile(f"{TELEPHONE.pattern}")
currency_regex = re.compile(f"{CURRENCY.pattern}")
email_regex = re.compile(f"{EMAIL.pattern}")
url_regex = re.compile(f"{URL.pattern}")


class PunctuationRestoration:
  def __init__(
      self,
      tag2punctuator,
      classifier_name='Qishuai/distilbert_punctuator_zh',
      tokenizer_name='Qishuai/distilbert_punctuator_zh',
      verbose=False
  ):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.classifier = DistilBertForTokenClassification.from_pretrained(classifier_name).to(self.device)
    self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
    self.tag2punctuator = tag2punctuator
    self.id2tag = self.classifier.config.id2label
    self.max_sequence_length = (
      self.classifier.config.max_position_embeddings // 2
    )
    self._reset_values()
    self.verbose = verbose

  def pre_process(self, inputs):
    def _input_process(tokens):

      special_token_index = {}
      for index, token in enumerate(tokens):
        if email_regex.match(token):
          tokens[index] = EMAIL_TOKEN
          special_token_index[index] = token
          continue
        if url_regex.match(token):
          tokens[index] = URL_TOKEN
          special_token_index[index] = token
          continue
        if currency_regex.match(token):
          tokens[index] = CURRENCY_TOKEN
          special_token_index[index] = token
          continue
        if tel_regex.match(token):
          tokens[index] = TELEPHONE_TOKEN
          special_token_index[index] = token
          continue
        if num_regex.match(token):
          tokens[index] = NUMBER_TOKEN
          special_token_index[index] = token
          continue
      return tokens, special_token_index

    index = 0
    last_is_split = False
    self.split_inputs_indexes = []
    for input in inputs:
      input_tokens = chinese_split(input).split()
      while len(input_tokens) > self.max_sequence_length:
        processed_input_tokens, special_token_index = list(
          _input_process(input_tokens[: self.max_sequence_length])
        )
        self.special_token_indexes.append(special_token_index)
        self.all_tokens.append(processed_input_tokens)
        self.split_inputs_indexes.append(index)
        input_tokens = input_tokens[self.max_sequence_length:]
        index += 1
        last_is_split = True
      else:
        if last_is_split:
          self.split_inputs_indexes.append(index)
          last_is_split = False
        index += 1
        processed_input_tokens, special_token_index = list(
          _input_process(input_tokens)
        )
        self.special_token_indexes.append(special_token_index)
        self.all_tokens.append(processed_input_tokens)
    logger.info(f"self split indexes: {self.split_inputs_indexes}")
    return self

  def tokenize(self):
    tokenized_inputs = self.tokenizer(
      self.all_tokens,
      is_split_into_words=True,
      padding=True,
      return_offsets_mapping=True,
      return_tensors="pt",
    )
    self.marks = self._mark_ignored_tokens(tokenized_inputs["offset_mapping"])
    self.tokenized_input_ids = tokenized_inputs["input_ids"].to(self.device)
    self.attention_mask = tokenized_inputs["attention_mask"].to(self.device)
    return self

  def classify(self):
    try:
      logits = self.classifier(
        self.tokenized_input_ids, self.attention_mask
      ).logits
      if self.device.type == "cuda":
        self.argmax_preds = logits.argmax(dim=2).detach().cpu().numpy()
      else:
        self.argmax_preds = logits.argmax(dim=2).detach().numpy()
    except RuntimeError as e:
      logger.error(f"error doing punctuation: {str(e)}")
    return self

  def post_process(self):
    reduce_ignored_marks = self.marks >= 0

    self.outputs_labels = []
    temp_ouputs = ""
    temp_outputs_labels = []
    for input_index, (
        pred,
        reduce_ignored,
        tokens,
        special_token_index,
    ) in enumerate(
      zip(
        self.argmax_preds,
        reduce_ignored_marks,
        self.all_tokens,
        self.special_token_indexes,
      )
    ):
      next_upper = True
      true_pred = pred[reduce_ignored]

      result_text = ""
      output_labels = []
      for id, (index, token) in zip(true_pred, enumerate(tokens)):
        tag = self.id2tag[id]
        output_labels.append(tag)
        if index in special_token_index:
          token = special_token_index[index]
        if next_upper:
          token = token.capitalize()
        punctuator, next_upper = self.tag2punctuator[tag]
        if is_ascii(token):
          result_text += token + punctuator + " "
        else:
          result_text += token + punctuator
      if input_index in self.split_inputs_indexes:
        temp_ouputs += result_text.strip()
        temp_outputs_labels.extend(output_labels)
      else:
        if temp_ouputs and temp_outputs_labels:
          self.outputs.append(temp_ouputs.strip())
          self.outputs_labels.append(temp_outputs_labels)
          temp_ouputs = ""
          temp_outputs_labels = []

        self.outputs.append(result_text.strip())
        self.outputs_labels.append(output_labels)

    if temp_ouputs and temp_outputs_labels:
      self.outputs.append(temp_ouputs.strip())
      self.outputs_labels.append(temp_outputs_labels)

    return self

  def punctuation(self, inputs):
    self._reset_values()
    self.pre_process(inputs).tokenize().classify().post_process()

    return self.outputs, self.outputs_labels

  def _reset_values(self):
    self.special_token_indexes = []
    self.all_tokens = []
    self.outputs = []

  def _mark_ignored_tokens(self, offset_mapping):
    samples = []
    for sample_offset in offset_mapping:
      # create an empty array of -100
      sample_marks = np.ones(len(sample_offset), dtype=int) * -100
      arr_offset = np.array(sample_offset)

      # set labels whose first offset position is 0 and the second is not 0, only special tokens second is also 0
      sample_marks[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = 0
      samples.append(sample_marks.tolist())

    return np.array(samples)
