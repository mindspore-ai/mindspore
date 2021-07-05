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

import time
from flask import Flask, request, jsonify, render_template
from flask_apscheduler import APScheduler
from mindspore_serving.client import Client

app = Flask(__name__)


def generate(input_sentence):
    """nerate sentence with given input_sentence"""
    client = Client("localhost:5500", "pangu", "predict")

    print(f"----------------------------- begin {input_sentence} ---------", flush=True)
    instance = {"input_sentence": input_sentence}
    time_start = time.time()
    result = client.infer(instance)
    reply = result[0]["output_sentence"]

    print(f"time cost {(time.time() - time_start) * 1000}ms, request '{input_sentence}' get reply '{reply}'",
          flush=True)

    return reply


@app.route('/query')
def do_query():
    s = request.args.get('u')
    output_sentence = generate(s)
    return jsonify(ok=True, rsvp=output_sentence)


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
