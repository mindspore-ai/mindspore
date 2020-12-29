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
"""data preprocess for downstream task"""
import re
import json
import random


def lambada_detokenizer(string):
    string = re.sub(r"``", "-DQ-", string)
    string = re.sub(r"`", "-SQ-", string)
    string = re.sub(r"''", "-DQ-", string)
    string = re.sub(r" '", "-SQ-", string)
    string = re.sub("-DQ-", '"', string)
    string = re.sub("-SQ-", "'", string)
    string = re.sub(r"([,?!.]['\"])(\w)", "\g<1> \g<2>", string)
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    string = string.replace(" 'd", "'d")
    string = string.replace(" '", "'")
    string = string.replace(" n't", "n't")

    string = string.replace(" .", ".")
    string = string.replace(" ,", ",")
    string = string.replace(" !", "!")
    string = string.replace(" ?", "?")
    string = string.replace(" :", ":")
    string = string.replace(" ;", ";")

    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" ,'", ",'")
    string = string.replace(" .'", ".'")
    string = string.replace(" !'", "!'")
    string = string.replace(" ?'", "?'")
    string = string.replace("~", "")
    string = string.replace("---", "")
    string = string.replace("<", "")
    string = string.replace(">", "")
    string = string.replace("#", "")

    string = string.replace(', "', ',"')
    string = string.replace('. "', '."')
    string = string.replace('! "', '!"')
    string = string.replace('? "', '?"')
    string = string.replace('"" ', '" "')
    string = string.replace('• • •', '')

    # sensitive word process
    string = string.replace("f ** k", "fuck")
    string = string.replace("f ** king", "fucking")
    string = string.replace("f ** ked", "fucked")
    string = string.replace("c ** k", "cock")
    string = string.replace("br ** sts", "breasts")
    string = string.replace("n ** ples", "nipples")
    string = string.replace("ni ** les", "nipples")
    string = string.replace("a ** hole", "asshole")
    string = string.replace("ass ** le", "asshole")
    string = string.replace("p ** sy", "pussy")
    string = string.replace("pu ** y", "pussy")
    string = string.replace("na ** d", "naked")
    string = string.replace("nak * d", "naked")
    string = string.replace("cli ** x", "climax")
    string = string.replace("h * ps", "hips")
    string = string.replace("c * ck", "cock")
    string = string.replace("coc ** ne", "cocaine")
    string = string.replace("*", "")

    string = re.sub("    "," ",string)
    string = re.sub("   "," ",string)
    string = re.sub("  "," ",string)

    return string


def lambada_dataset_preprocess(input_file, output_file):
    sentences = []
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                line = lambada_detokenizer(line)
                split_sentence_list = line.split()
                final_word = split_sentence_list[-1]
                context = split_sentence_list[:-1]
                new_sentence = ' '.join(context) + '\t' + ' ' + final_word
                sentences.append(new_sentence)
                count += 1
    print('read {} file finished!\n total count = {}'.format(input_file, count))

    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                f.write(sentence)
                f.write('\n')
                count -= 1
    print('write {} file finished!\n total count = {}'.format(output_file, count))


def get_gold_answer_id(gold_answer, candidate_answer_list):
    id_ = 0
    for candidate in candidate_answer_list:
        if gold_answer == candidate:
            return id_
        id_ += 1


def get_passage_string(passage_string, candidate_answer, final_sentence, gold_answer_id):
    """
    concat each candidate answer to the rest_sentence
    Args:
        candidate_answer (list): store each candidate answers
        final_sentence (str): the 21st sentence string with "XXXXX"
        gold_answer_id (int): the id of correct answer.

    return:
        candidate_passage (list): the length of candidate_sentence equals to length of candidate_answer.
    """
    candidate_passage = []
    for answer in candidate_answer:
        passage = passage_string + "  " + final_sentence
        passage = passage.replace(" XXXXX", "\t XXXXX")
        final_passage = passage.replace("XXXXX", answer)
        whole_passage = final_passage + "\t" + str(gold_answer_id)
        candidate_passage.append(whole_passage)

    return candidate_passage


def cbt_dataset_preprocess(input_file, output_file):
    passages = []
    candidate_passage_list = []
    passage_string = ""
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                single_sentence = line.split(' ', 1)
                line_id = int(single_sentence[0])
                string = single_sentence[1]

                if line_id == 21:
                    string = string.replace("\t\t", "\t")
                    mini_string = string.split("\t")
                    candidate_answer = mini_string[-1]
                    candidate_answer_list = candidate_answer.split("|")
                    gold_answer_id = get_gold_answer_id(mini_string[-2], candidate_answer_list)
                    candidate_passage = get_passage_string(passage_string,
                                                           candidate_answer_list,
                                                           mini_string[0],
                                                           gold_answer_id)
                    assert len(candidate_passage) == 10
                    count += 10

                else:
                    passage_string = passage_string + " " + string
            else:
                passages.append(candidate_passage)
                candidate_passage_list = []
                passage_string = ""

    print('read {} file finished!\n total count = {}'.format(input_file, count))

    with open(output_file, 'w', encoding='utf-8') as f:
        for passage in passages:
            for candidate_passage in passage:
                candidate_passage = candidate_passage.replace(" \t ", "\t ")
                candidate_passage = candidate_passage.strip()
                f.write(candidate_passage)
                f.write("\n")
                count -= 1

    print('write {} file finished!\n total count = {}'.format(output_file, count))


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" .", ".")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def wikitext_dataset_preprocess(input_file, output_file):
    dataset_test = []
    passage = []
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                if line.startswith('=') and line.endswith('=') and len(passage) != 0:
                    dataset_test.append(passage)
                    count += 1
                    passage = []
                elif line.startswith('=') and line.endswith('='):
                    continue
                else:
                    passage.append(line)
    print('read {} file finished!\n total count = {}'.format(input_file, count))

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in dataset_test:
            text = ""
            for sentence in line:
                sentence = wikitext_detokenizer(sentence)
                text = text + " " + sentence
            text = text.strip()
            f.write(text)
            f.write("\n")
    print('write {} file finished!\n total count = {}'.format(output_file, count))


def ptb_detokenizer(string):
    string = string.replace(" '", "'")
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" n't", "n't")
    string = string.replace(" N ", "1 ")
    string = string.replace("$ 1", "$1")
    string = string.replace("# 1", "#1")
    string = string.replace("\/abc", "")
    string = string.replace("\/ua", "")

    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")

    string = string.replace(" 's", "'s")

    return string


def ptb_dataset_preprocess(input_file, output_file):
    sentences = []
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                line = ptb_detokenizer(line)
                sentences.append(line)
                count += 1
    print('read {} file finished!\n total count = {}'.format(input_file, count))

    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                f.write(sentence)
                f.write("\n")
                count -= 1
    print('write {} file finished!\n total count = {}'.format(output_file, count))


def onebw_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" --", "")
    string = string.replace("--", "")
    string = string.replace("? ? ?", " ?")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" 't", "'t")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    string = string.replace(" '", "'")
    string = string.replace(" n't", "n't")
    string = string.replace("$ 1", "$1")
    string = string.replace("# 1", "#1")

    return string


def test_length(string):
    string_list = string.split()
    return len(string_list)


def onebw_dataset_preprocess(condition, input_file, output_file):
    sentences = []
    count = 0
    if condition.lower() == "test":
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)
                    count += 1
        print('read {} file finished!\n total count = {}'.format(input_file, count))

        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    sentence = onebw_detokenizer(sentence)
                    f.write(sentence)
                    f.write("\n")
                    count -= 1
        print('write {} file finished!\n total count = {}'.format(output_file, count))

    elif condition.lower() == "train":
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    line = onebw_detokenizer(line)
                    length = test_length(line)
                    if length > 10 and length < 60:
                        sentences.append(line)
                        count += 1
        print('read finished! count = ', count)

        sample_result_list = random.sample(range(0, count), 30000)
        sample_result_list.sort()
        count_sample = 0
        choiced_sentence = ""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(len(sample_result_list)):
                choiced_sentence = sentences[sample_result_list[i]]
                f.write(choiced_sentence)
                f.write("\n")
                count_sample += 1
        print('write finished! ', count_sample)

    else:
        raise ValueError("condition error support: [train, test]")


def coqa_dataset_preprocess(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    stories = []
    instances = []
    end_sep = [',', '.', ';']
    question_before_sep = " "
    question_after_sep = " A: "
    answer_sep = " A:\t"

    for i, dialog in enumerate(source_data["data"]):
        story = dialog["story"].replace("\n", "")
        stories.append(story)

        concat_ = ""
        concat_ += story
        for question, answer in zip(dialog["questions"], dialog["answers"]):
            question = question["input_text"]
            answer = answer["input_text"]
            concat_ += question_before_sep
            concat_ += question
            tmp = concat_ + question_after_sep
            concat_ += answer_sep
            concat_ += answer
            instances.append(concat_)
            concat_ = tmp + answer
            if concat_[-1] not in end_sep:
                concat_ += "."
        instances.append("")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(len(instances)):
            if instances[i]:
                f.write(instances[i])
                f.write("\n")

    print('write {} file finished!\n total count = {}'.format(output_file, len(instances)))


def wmt14_en_fr_preprocess(input_file, output_file):
    input_file = input_file + "/newstest2014-fren-ref"
    output_file = output_file + "/wmt14"
    language = ['.en.sgm', '.fr.sgm']
    count = 0
    # en-fr
    with open(input_file + language[0], "r", encoding='utf-8') as english, \
            open(input_file + language[1], "r", encoding='utf-8') as french, \
            open(output_file + '.en_fr.txt', "a", encoding='utf-8') as enfr_f, \
            open(output_file + '.fr_en.txt', "a", encoding='utf-8') as fren_f:
        line_id = 0
        for en, fr in zip(english, french):
            line_id += 1
            if (en[:7] == '<seg id'):
                print("=" * 20, "\n", line_id, "\n", "=" * 20)
                en_start = en.find('>', 0)
                en_end = en.find('</seg>', 0)
                print(en[en_start + 1:en_end])
                en_ = en[en_start + 1:en_end]

                fr_start = fr.find('>', 0)
                fr_end = fr.find('</seg>', 0)
                print(fr[fr_start + 1:fr_end])
                fr_ = fr[fr_start + 1:fr_end]

                en_fr_str = en_ + "\t" + fr_ + "\n"
                enfr_f.write(en_fr_str)
                fr_en_str = fr_ + "\t" + en_ + "\n"
                fren_f.write(fr_en_str)
                count += 1

    print('write {} file finished!\n total count = {}'.format(output_file + '.en_fr.txt', count))
    print('write {} file finished!\n total count = {}'.format(output_file + '.fr_en.txt', count))
