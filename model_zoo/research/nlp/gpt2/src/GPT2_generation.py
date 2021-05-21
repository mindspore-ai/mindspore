# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
generation class for downstream task (Summarization, Reading Comprehension, Translation)
"""
import numpy as np

from .utils.task_utils import extract_logits
from .utils.generation_utils import Sample
from .utils.tensor_manipulations import extract_string_from_tensor

INF = 1. * 1e9


class GenerateForSummarization():
    """
    generate for summarization task
    """

    def __init__(self,
                 decoder,
                 config=None,
                 tokenizer=None,
                 select_sentence=3,
                 eval_type="finetuned",
                 temperature=1.0,
                 generate_length=100,
                 topk=2,
                 topp=1.0):

        self.decoder = decoder
        self.config = config
        self.tokenizer = tokenizer
        self.select_sentence = select_sentence
        self.eval_type = eval_type

        self.generator = Sample(decoder,
                                tokenizer=tokenizer,
                                config=config,
                                topk_num=topk,
                                topp_prob=topp,
                                min_tokens_to_keep=1,
                                temperature=temperature)
        self.generate_length = generate_length

    def generate_for_summarization(self, input_ids):
        """generation function for summarization task"""

        # prepare input_str
        article_str, summary_str = extract_string_from_tensor(input_ids=input_ids,
                                                              mode="pair",
                                                              config=self.config,
                                                              tokenizer=self.tokenizer)
        generated_summary_list = [""] * self.config.batch_size

        # clip overflow
        for batch_idx in range(self.config.batch_size):
            last_dot_pos = max(article_str[batch_idx].rfind(' .'), article_str[batch_idx].rfind('. ')) + 2
            article_str[batch_idx] = article_str[batch_idx][:last_dot_pos]

        # pad a <TL,DR;> token(<EOS>) after the string of Article.
        tldr_str = "TL;DR:"
        if self.eval_type == "finetuned":
            for batch_idx in range(self.config.batch_size):
                article_str[batch_idx] += (" " + tldr_str)

        # add prefix
        for batch_idx in range(self.config.batch_size):
            article_str[batch_idx] = article_str[batch_idx]
        generate_str_list, _ = self.generator.generate(input_str=article_str, generate_length=self.generate_length)
        for batch_idx in range(self.config.batch_size):
            generate_str = generate_str_list[batch_idx]
            generated_summary = ""

            if self.select_sentence > 0:
                # check if there are number of select_sentence of sentences in generated text,
                # if not enough, it will return full generated string
                len_generate_str = len(generate_str)
                search_index = -1
                for _ in range(self.select_sentence):
                    search_index = generate_str.find('.', search_index + 1)
                    if search_index == -1 or search_index >= len_generate_str:
                        search_index = len_generate_str
                        break

                # increase search_index to add period token('.') if search_index does not overflow.
                search_index = search_index + 1 if search_index < len_generate_str else len_generate_str
                generated_summary = generate_str[:search_index]
                if generated_summary.find(self.tokenizer.eos_token) != -1:
                    cut_pos = generated_summary.find(self.tokenizer.eos_token, 0)
                    generated_summary = generated_summary[:cut_pos]
            else:
                generated_summary = generate_str

            # if all of str hs been clipped, restore it to beginning state.
            if generated_summary == '':
                generated_summary = generate_str
            # empty str check
            if generated_summary == '':
                generated_summary = '<empty>'
            generated_summary_list[batch_idx] = generated_summary

        return generated_summary_list, summary_str  # Hypo and Ref


class GenerateForLambada():
    """
    generate class for lambada task, which is to predict the final word of sentence.
    """
    def __init__(self,
                 decoder,
                 config=None,
                 tokenizer=None,
                 generate_length_dynamic=True,
                 generate_length=1,
                 max_iterations=200,
                 stop_word_file=""):
        """
        Args:
            decoder: decoder (Model): GPT2 model to do generation.
            config (object): configuration of given GPT2 model.
            tokenizer (object): if choose to use input_str parameter in self.generate(), a tokenizer is compulsory.
            generate_length_dynamic (bool): True for the generate length is dynamic, False for fixed. Default: True.
            max_iterations (int): choose the top k token according to selected probability, there k = `max_iterations`.
            generate_length (int): the final word max generated token length.
            stop_word_file (str): stop word file is used to be a stop-word filter.
        """
        self.decoder = decoder
        self.config = config
        self.batch_size = config.batch_size
        self.tokenizer = tokenizer
        self.generate_length_dynamic = generate_length_dynamic
        self.generate_length = generate_length
        self.max_iterations = max_iterations
        self.stop_word_set = self.build_stop_word(stop_word_file)

        self.generator = Sample(decoder=decoder,
                                config=config,
                                batch_size=1,
                                tokenizer=tokenizer,
                                topk_num=1,
                                topp_prob=1,
                                return_ids=True
                                )
        self.stop_eos = ['.', ',', '!', '?', '"', " '", " and", " says", " said"]

    def build_stop_word(self, stop_word_file):
        stop_words_set = set()
        with open(stop_word_file, 'r', encoding="utf8") as file:
            for line in file.readlines():
                line = line.strip('\n')
                stop_words_set.add(line)
        return stop_words_set

    def is_stop_word(self, word):
        flag = False
        if word in self.stop_word_set:
            flag = True
            return flag
        return flag

    def generate_for_lambada(self, input_ids, logits, input_length):
        """
        generation function for lambada task

        Args:
            input_ids (Tensor): input sentences with shape [batch_size, seq_len].
            logits (Tensor): the language model distribution.
            input_length (Tensor): store the context length which not including final word , and whole sentence length

        return:
            batch_predict_words (list): the list of predict_words

        """
        batch_predict_words = ["" for _ in range(self.batch_size)]
        input_len_np = input_length.asnumpy()
        input_ids_list = input_ids.asnumpy().tolist()

        extracted_logits = extract_logits(logits=logits, position=input_len_np)  # [batch_size, vocab_size]
        extracted_logits = extracted_logits.asnumpy()
        sorted_ids = np.argsort(-extracted_logits, axis=-1)[::, :self.max_iterations]  # [batch_size, max_iterations]

        for batch_idx in range(self.batch_size):
            final_word_spos = input_len_np[batch_idx, 0]
            context_ids = input_ids_list[batch_idx][1:final_word_spos]  # 1 for dropping <bos> token
            last_word_token_num = input_len_np[batch_idx, 1] - input_len_np[batch_idx, 0]

            if self.generate_length_dynamic:
                generate_length = last_word_token_num
            else:
                generate_length = self.generate_length

            for num in range(self.max_iterations):
                id_ = sorted_ids[batch_idx][num]
                source_ids = context_ids + [id_]
                source_string = self.tokenizer.decode(source_ids)
                generated_ids_list = self.generator.generate(input_str=source_string,
                                                             generate_length=generate_length,
                                                             do_sample=False)
                predict_tokens_ids = [id_] + generated_ids_list[0]
                predict_word = self.tokenizer.decode(predict_tokens_ids)

                eos_pos = min(predict_word.find(word) if predict_word.find(word) >= 0
                              else INF for word in self.stop_eos)
                if eos_pos == INF:
                    continue
                else:
                    predict_word = predict_word[:eos_pos]
                predict_word = predict_word.strip()
                if predict_word.find(" ") == -1:
                    if self.is_stop_word(word=predict_word.lower()):
                        continue
                    batch_predict_words[batch_idx] = predict_word
                    print("predict word: {}".format(predict_word))
                    break
        return batch_predict_words


class GenerateForTranslation():
    """
    generate class for translation task
    """
    def __init__(self,
                 decoder,
                 config=None,
                 tokenizer=None,
                 generate_length=1,
                 use_hint=True,
                 select_first_sentence=True,
                 topk_num=None,
                 topp_prob=None,
                 temperature=None
                 ):

        self.decoder = decoder
        self.config = config
        self.batch_size = config.batch_size
        self.tokenizer = tokenizer
        self.generate_length = generate_length
        self.use_hint = use_hint
        self.select_first_sentence = select_first_sentence

        self.generator = Sample(decoder=decoder,
                                config=config,
                                tokenizer=tokenizer,
                                topk_num=topk_num,
                                topp_prob=topp_prob,
                                temperature=temperature,
                                min_tokens_to_keep=1,
                                early_stop=True)

    def generate_for_translation(self, input_ids):
        """generation function for translation task"""
        source_str_list, ref_str_list = extract_string_from_tensor(input_ids=input_ids,
                                                                   mode="pair",
                                                                   config=self.config,
                                                                   tokenizer=self.tokenizer)
        final_predict_translation_list = [""] * self.batch_size

        if self.use_hint:
            for index in range(self.batch_size):
                source_str_list[index] += " ="  # now source_str is "english sentence ="

        translation_str_list, _ = self.generator.generate(input_str=source_str_list,
                                                          generate_length=self.generate_length,
                                                          do_sample=False)

        for index in range(self.batch_size):
            generate_str = translation_str_list[index].replace('<|endoftext|>', '')
            predict_translation = ""

            # According to the GPT2 paper, the select_first_sentence will be set "True"
            if self.select_first_sentence:
                # check if there are number of select_sentence of sentences in generated text,
                # if not enough, it will return full generated string
                search_index = generate_str.find('.', 0, len(generate_str))
                if search_index == -1:
                    search_index = len(generate_str)
                else:
                    search_index = search_index + 1
                predict_translation = generate_str[:search_index]
            else:
                predict_translation = generate_str

            if predict_translation == '':
                predict_translation = '<empty>'

            final_predict_translation_list[index] = predict_translation

        return final_predict_translation_list, ref_str_list


class GenerateForReadComprehension():
    """
    generate class for Reading Comprehension task.

    Args:
        decoder: decoder (Model): GPT2 model to do generation.
        config (object): configuration of given GPT2 model.
        tokenizer (object): if choose to use input_str parameter in self.generate(), a tokenizer is compulsory.
        generate_length (int):

    """

    def __init__(self,
                 decoder,
                 config=None,
                 tokenizer=None,
                 generate_length=1,
                 topk_num=None,
                 topp_prob=None,
                 temperature=None
                 ):

        self.decoder = decoder
        self.config = config
        self.batch_size = config.batch_size
        self.tokenizer = tokenizer
        self.generate_length = generate_length

        self.generator = Sample(decoder=decoder,
                                config=config,
                                tokenizer=tokenizer,
                                topk_num=topk_num,
                                topp_prob=topp_prob,
                                temperature=temperature,
                                min_tokens_to_keep=1,
                                )

    def generate_for_read_comprehension(self, input_ids):
        """generation function for reading comprehension task"""
        passage_str_list, answer_str_list = extract_string_from_tensor(input_ids=input_ids,
                                                                       mode="pair",
                                                                       config=self.config,
                                                                       tokenizer=self.tokenizer)
        passage = passage_str_list[:]

        generate_str_list, _ = self.generator.generate(input_str=passage_str_list,
                                                       generate_length=self.generate_length,
                                                       do_sample=False)

        pred_answer = []
        for batch_id in range(self.batch_size):
            new_str = generate_str_list[batch_id].replace('<|endoftext|>', '')
            index_a = new_str.find('.')
            index_b = new_str.find('Q:')
            if index_a != -1 or index_b != -1:
                index = max(index_a, index_b)
                pred_answer += [new_str[1:index]]  # 1 represents skip the space in the beginning of the sentence
            else:
                pred_answer += [new_str]

        return passage, pred_answer, answer_str_list
