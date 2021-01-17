/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <math.h>
#include <getopt.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <utility>
#include "src/net_runner.h"
#include "include/context.h"
#include "src/utils.h"
#include "src/data_loader.h"
#include "src/accuracy_monitor.h"

static unsigned int seed = time(NULL);

std::vector<int> FillInputDataUtil(const mindspore::session::TrainLoopCallBackData &cb_data,
                                   const std::vector<DataLabelTuple> &dataset, bool serially) {
  static unsigned int idx = 1;
  int total_size = dataset.size();
  std::vector<int> labels_vec;

  auto inputs = cb_data.session_->GetInputs();
  char *input_data = reinterpret_cast<char *>(inputs.at(0)->MutableData());
  auto labels = reinterpret_cast<float *>(inputs.at(1)->MutableData());
  int batch_size = inputs.at(0)->shape()[0];
  int num_of_classes = inputs.at(1)->shape()[1];
  int data_size = inputs.at(0)->Size() / batch_size;
  MS_ASSERT(total_size > 0);
  MS_ASSERT(input_data != nullptr);
  std::fill(labels, labels + inputs.at(1)->ElementsNum(), 0.f);
  for (int i = 0; i < batch_size; i++) {
    if (serially) {
      idx = ++idx % total_size;
    } else {
      idx = rand_r(&seed) % total_size;
    }
    int label = 0;
    char *data = nullptr;
    std::tie(data, label) = dataset[idx];
    std::copy(data, data + data_size, input_data + i * data_size);
    labels[i * num_of_classes + label] = 1.0;  // Model expects labels in onehot representation
    labels_vec.push_back(label);
  }
  return labels_vec;
}

void DataLoader::StepBegin(const mindspore::session::TrainLoopCallBackData &cb_data) {
  FillInputDataUtil(cb_data, ds_->train_data(), false);
}

int AccuracyMonitor::EpochEnd(const mindspore::session::TrainLoopCallBackData &cb_data) {
  if ((cb_data.epoch_ + 1) % check_every_n_ != 0) return mindspore::session::RET_CONTINUE;

  float accuracy = 0.0;
  auto inputs = cb_data.session_->GetInputs();
  int batch_size = inputs.at(0)->shape()[0];
  int num_of_classes = ds_->num_of_classes();
  int tests = ds_->test_data().size() / batch_size;
  if (max_steps_ != -1 && tests > max_steps_) tests = max_steps_;
  cb_data.session_->Eval();
  for (int i = 0; i < tests; i++) {
    auto labels = FillInputDataUtil(cb_data, ds_->test_data(), false);
    cb_data.session_->RunGraph();
    auto outputs = cb_data.session_->GetPredictions();
    for (auto it = outputs.begin(); it != outputs.end(); ++it) {
      if (it->second->ElementsNum() == batch_size * num_of_classes) {
        auto scores = reinterpret_cast<float *>(it->second->MutableData());
        for (int b = 0; b < batch_size; b++) {
          int max_idx = 0;
          float max_score = scores[num_of_classes * b];
          for (int c = 1; c < num_of_classes; c++) {
            if (scores[num_of_classes * b + c] > max_score) {
              max_score = scores[num_of_classes * b + c];
              max_idx = c;
            }
          }
          if (labels[b] == max_idx) accuracy += 1.0;
        }
        break;
      }
    }
  }
  accuracy /= static_cast<float>(batch_size * tests);
  accuracies_.push_back(std::make_pair(cb_data.epoch_, accuracy));
  std::cout << cb_data.epoch_ + 1 << ":\tAccuracy is " << accuracy << std::endl;
  cb_data.session_->Train();
  return mindspore::session::RET_CONTINUE;
}
