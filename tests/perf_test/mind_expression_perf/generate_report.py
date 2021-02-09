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
import sys
import re
import json
import os
import time
import openpyxl as opx


def parse_arguments():
    print(sys.argv)
    me_report_path = sys.argv[1]
    log_path = sys.argv[2]
    n_iter = sys.argv[3]
    out = sys.argv[4]
    assert n_iter.isdigit()
    return me_report_path, log_path, int(n_iter), out


def extract_by_keyword(doc, keyword, pattern):
    rst = []
    for i, s in enumerate(doc):
        if keyword in s:
            p = re.findall(pattern, s)
            print("L%d: extracted %s from '%s'" % (i, p, s.strip()))
            rst.extend(p)
    return rst


def process_log(fname, log_path, n_iter, keyword, pattern):
    rnt = {}
    for i in range(1, 1+n_iter):
        fname_path = os.path.join(log_path, fname % i)
        with open(fname_path) as f:
            print("\nLoading %s" % fname_path)
            rst = extract_by_keyword(f, keyword, pattern)
        rnt[fname % i] = rst
    return rnt


def summarize(func):
    def wrapper(*args, **kwargs):
        log = func(*args, **kwargs)
        times = list(log.items())
        times.sort(key=lambda x: x[1])
        min_file, min_time = times[0]
        avg = sum(map(lambda x: x[1], times)) / len(times)
        log["min_time"] = min_time
        log["min_file"] = min_file
        log["avg_time"] = avg
        return log
    return wrapper


@summarize
def process_bert_log(log_path, n_iter):
    fname = "bert%d.log"
    total = process_log(fname, log_path, n_iter, "TotalTime", r"\d+.\d+")
    task = process_log(fname, log_path, n_iter, "task_emit", r"\d+.\d+")
    log = {}
    for fname in total:
        log[fname] = float(total[fname][0]) - float(task[fname][0])
    return log


@summarize
def process_resnet_log(log_path, n_iter):
    fname = "resnet%d.log"
    total = process_log(fname, log_path, n_iter, "TotalTime", r"\d+.\d+")
    task = process_log(fname, log_path, n_iter, "task_emit", r"\d+.\d+")
    log = {}
    for fname in total:
        log[fname] = float(total[fname][0]) - float(task[fname][0])
    return log


@summarize
def process_gpt_log(log_path, n_iter):
    fname = "gpt%d.log"
    total = process_log(fname, log_path, n_iter, "TotalTime", r"\d+.\d+")
    task = process_log(fname, log_path, n_iter, "task_emit", r"\d+.\d+")
    log = {}
    for fname in total:
        log[fname] = float(total[fname][0]) - float(task[fname][0])
    return log


@summarize
def process_reid_log(log_path, n_iter):
    log = {}
    for i in range(8):
        fname = "reid_%d_"+str(i)+".log"
        total = process_log(fname, log_path, n_iter, "TotalTime", r"\d+.\d+")
        task = process_log(fname, log_path, n_iter, "task_emit", r"\d+.\d+")
        for fname in total:
            log[fname] = float(total[fname][0]) - float(task[fname][0])
    return log


def write_to_me_report(log, me_report_path):
    wb = opx.load_workbook(me_report_path)
    sheet = wb["Sheet"]
    idx = sheet.max_row + 1
    date = time.strftime('%m%d', time.localtime())
    sheet['A%d' % idx] = date
    sheet['B%d' % idx] = round(log["reid"]["min_time"], 2)
    sheet['C%d' % idx] = round(log["bert"]["min_time"], 2)
    sheet['D%d' % idx] = round(log['resnet']["min_time"], 2)
    sheet['E%d' % idx] = round(log['gpt']["min_time"], 2)
    wb.save(me_report_path)


def generate_report():
    me_report_path, log_path, n_iter, out = parse_arguments()
    log_data = {}
    bert_log = process_bert_log(log_path, n_iter)
    resnet_log = process_resnet_log(log_path, n_iter)
    gpt_log = process_gpt_log(log_path, n_iter)
    reid_log = process_reid_log(log_path, n_iter)
    log_data["bert"] = bert_log
    log_data["resnet"] = resnet_log
    log_data["gpt"] = gpt_log
    log_data["reid"] = reid_log
    with open(out, "w") as f:
        json.dump(log_data, f, indent=2)
    write_to_me_report(log_data, me_report_path)


if __name__ == "__main__":
    generate_report()
