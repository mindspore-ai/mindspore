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

"""Evaluation reading comprehension result with additional answer."""

import json
import re
import string
import argparse
from collections import Counter


def get_normalize_answer_token(string_):
    """normalize the answer token, Lower text and remove punctuation, article and extra whitespace"""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(char for char in text if char not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(string_)))).split()


def calculate_f1(pred_answer, gold_answer):
    """
    calculate final F1 score with addition answer
    """
    f1_score = 0
    pred_answer = get_normalize_answer_token(pred_answer)
    gold_answer = get_normalize_answer_token(gold_answer)
    common = Counter(pred_answer) & Counter(gold_answer)
    num_same = sum(common.values())
    # the number of same tokens between pred_answer and gold_answer
    precision = 1.0 * num_same / len(pred_answer) if "".join(pred_answer).strip() != "" else 0
    recall = 1.0 * num_same / len(gold_answer) if "".join(gold_answer).strip() != "" else 0
    if "".join(pred_answer).strip() == "" and "".join(gold_answer).strip() == "":
        f1_score = 1
    else:
        f1_score = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0.0
    return f1_score


def main():
    parser = argparse.ArgumentParser(description="All Task dataset preprocessing")
    parser.add_argument("--input_file", type=str, default="",
                        help="The log file path of evaluation in Reading Comprehension. ")
    parser.add_argument("--addition_file", type=str, default="", help="Coqa-dev-v1.0.json path")
    args_opt = parser.parse_args()
    input_file = args_opt.input_file
    addition_file = args_opt.addition_file

    find_word = 'Pred_answer:'
    find_word_length = len(find_word)
    pred_answer_list = []

    with open(input_file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            index = line.find(find_word)
            if index != -1:
                pred_answer = line[index + find_word_length:].strip()
                pred_answer_list.append(pred_answer)

    dataset = json.load(open(addition_file))
    pred_answer_num = 0
    total_f1score = 0
    average_f1score = 0
    data_num = len(pred_answer_list)

    for story in dataset['data']:
        questions = story['questions']
        multiple_answers = [story['answers']]
        multiple_answers += story['additional_answers'].values()
        for question in questions:
            pred_a = pred_answer_list[pred_answer_num]
            turn_id = question['turn_id']
            max_score = 0
            max_group = 0
            flag = 0
            for i, answer in enumerate(multiple_answers):
                gold_a = answer[turn_id - 1]['input_text']
                score = calculate_f1(pred_a, gold_a)
                if score > max_score:
                    max_score = score
                    max_group = i
                # calculate the max score in multiple answers and record it's number.
            gold_a = multiple_answers[max_group][turn_id - 1]['input_text']
            pred_answer_num += 1
            total_f1score += max_score
            average_f1score = total_f1score / pred_answer_num

            print('====================   data {}   ===================='.format(pred_answer_num))
            print('| Gold_answer:{}'.format(gold_a))
            print('| Pred_answer:{}'.format(pred_a))
            print('| F1_Score:{:.8f}'.format(average_f1score))
            print('=====================================================\n')

            if pred_answer_num >= data_num:
                flag = 1
                break
                # Stop flag
        if flag:
            print('Finished evaluation with addition answer! \n')
            print("********************** Testing Finished **********************")
            print('| Test file name: {}'.format(input_file))
            print('| Final F1 score: {:.8f}'.format(average_f1score))
            print('| Total data num: {}'.format(pred_answer_num))
            print("**************************************************************")
            break


if __name__ == "__main__":
    main()
