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
import os
import sys
import json
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import openpyxl as opx


def parse_arguments():
    log_path = sys.argv[1]
    log_data = sys.argv[2]
    me_report = sys.argv[3]
    n_days = sys.argv[4]
    assert n_days.isdigit()
    return log_path, log_data, me_report, int(n_days)


def read_data(log_data, me_report_path, n_days):
    with open(log_data) as f:
        log = json.load(f)

    wb = opx.load_workbook(me_report_path)
    sheet = wb["Sheet"]
    n_row = sheet.max_row
    date = [cell[0].value for cell in sheet["A2":"A%d" % n_row]]
    reid_data = [float(cell[0].value) for cell in sheet["B2":"B%d" % n_row]]
    bert_data = [float(cell[0].value) for cell in sheet["C2":"C%d" % n_row]]
    resnet_data = [float(cell[0].value) for cell in sheet["D2":"D%d" % n_row]]
    gpt_data = [float(cell[0].value) for cell in sheet["E43":"E%d" % n_row]]
    if n_days > 0:
        date = date[-n_days:]
        reid_data = reid_data[-n_days:]
        bert_data = bert_data[-n_days:]
        resnet_data = resnet_data[-n_days:]
        gpt_data = gpt_data[-n_days:]

    return log, date, reid_data, bert_data, resnet_data, gpt_data


def draw_figure(x_data, y_data, labels, title, out, height=24, width=8, tick_space=2):
    print("Generating figure to: %s" % out)
    plt.figure(figsize=(height, width))
    for y, label in zip(y_data, labels):
        x = x_data[-len(y):]
        n_data = len(x)
        assert len(x) == len(
            y), "assume len(x) == len(y), while %d != %d" % (len(x), len(y))
        plt.plot(x, y, linewidth=2, marker='o', markersize=5, label=label)
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_space))
        for i in range(n_data):
            if i % 2 == 0:
                plt.text(x[i], y[i], y[i], ha='center',
                         va='bottom', fontsize=8)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Time(s)")
    plt.grid()
    plt.legend()
    plt.savefig(out)


def generate_report(log, labels, log_path):
    for label in labels:
        fname = log[label]["min_file"]
        fname_path = os.path.join(log_path, fname)
        out_path = os.path.join(log_path, "reports", label+"_me.log")
        print("Generating report to: %s" % out_path)
        os.system("grep -A 230 'TotalTime = ' %s > %s" %
                  (fname_path, out_path))


def process_data():
    log_path, log_data, me_report, n_days = parse_arguments()
    log, date, reid_data, bert_data, resnet_data, gpt_data = read_data(
        log_data, me_report, n_days)
    draw_figure(date,
                [reid_data, bert_data, gpt_data],
                ["ReID", "BERT", "GPT"],
                "ReID&BERT&GPT",
                os.path.join(log_path, "reports", "reid_bert_gpt.png")
                )
    draw_figure(date, [resnet_data], ["ResNet"], "ResNet",
                os.path.join(log_path, "reports", "resnet.png"))
    generate_report(log, list(log.keys()), log_path)


if __name__ == "__main__":
    process_data()
