# Copyright 2021 Huawei Technologies Co., Ltd
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
"""flask server, Serving client"""

import os
import time
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_apscheduler import APScheduler
from mindspore_serving.client import Client
from tokenization_jieba import JIEBATokenizer

cur_dir = os.path.abspath(os.path.dirname(__file__))
tokenizer_path = os.path.join(cur_dir, "tokenizer")
tokenizer = JIEBATokenizer(os.path.join(tokenizer_path, "vocab.vocab"), os.path.join(tokenizer_path, "vocab.model"))
end_token = tokenizer.eot_id

app = Flask(__name__)


def generate(input_sentence):
    """nerate sentence with given input_sentence"""
    client = Client("localhost", 5500, "pangu", "predict")

    print(f"----------------------------- begin {input_sentence} ---------", flush=True)
    tokens = tokenizer.tokenize(input_sentence)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_len = len(input_ids)

    seq_length = 1024
    generate_number = 0
    end_flag = False

    time_start = time.time()
    while generate_number < 50:
        if len(input_ids) >= seq_length - 1:
            break

        time0 = time.time()
        instance = {"input_tokens": np.array([input_ids])}
        target = client.infer([instance])
        target = int(target[0]["add_token"])
        print(f"request '{input_sentence}' add token {generate_number}: {target}, "
              f"time cost {(time.time() - time0) * 1000}ms", flush=True)
        if target == end_token:
            if len(input_ids) == input_len:
                continue
            end_flag = True
            break

        input_ids.append(target)
        generate_number += 1

        outputs = input_ids[input_len:]
        return_tokens = tokenizer.convert_ids_to_tokens(outputs)
        reply = "".join(return_tokens)
        if reply:
            break

    print(f"time cost {(time.time() - time_start) * 1000}ms, request '{input_sentence}' get reply '{reply}'"
          f" end flag{end_flag}", flush=True)

    return reply, end_flag


@app.route('/query')
def do_query():
    s = request.args.get('u')
    output_sentence, end_flag = generate(s)
    return jsonify(ok=True, rsvp=output_sentence, end_flag=end_flag)


@app.route('/')
def index():
    return render_template("index.html")


class Config():
    JOBS = [
        {'id': 'job',
         'func': 'do_query',
         'args': '',
         'trigger': {'type': 'cron', 'second': '*/1'}}
    ]


if __name__ == '__main__':
    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
