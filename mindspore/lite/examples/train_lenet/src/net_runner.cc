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

#include "src/net_runner.h"
#include <math.h>
#include <getopt.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <utility>
#include "include/context.h"
#include "include/train/loss_monitor.h"
#include "include/train/ckpt_saver.h"
#include "include/train/lr_scheduler.h"
#include "include/train/classification_train_accuracy_monitor.h"
#include "src/utils.h"
#include "src/data_loader.h"
#include "src/accuracy_monitor.h"

using mindspore::session::TrainLoopCallBack;
using mindspore::session::TrainLoopCallBackData;

static unsigned int seed = time(NULL);

// Definition of callback function after forwarding operator.
bool after_callback(const std::vector<mindspore::tensor::MSTensor *> &after_inputs,
                    const std::vector<mindspore::tensor::MSTensor *> &after_outputs,
                    const mindspore::CallBackParam &call_param) {
  printf("%s\n", call_param.node_name.c_str());
  for (size_t i = 0; i < after_inputs.size(); i++) {
    int num2p = (after_inputs.at(i)->ElementsNum());
    printf("in%zu(%d): ", i, num2p);
    if (num2p > 10) num2p = 10;
    if (after_inputs.at(i)->data_type() == mindspore::kNumberTypeInt32) {
      auto d = reinterpret_cast<int *>(after_inputs.at(i)->MutableData());
      for (int j = 0; j < num2p; j++) printf("%d, ", d[j]);
    } else {
      auto d = reinterpret_cast<float *>(after_inputs.at(i)->MutableData());
      for (int j = 0; j < num2p; j++) printf("%f, ", d[j]);
    }
    printf("\n");
  }
  for (size_t i = 0; i < after_outputs.size(); i++) {
    auto d = reinterpret_cast<float *>(after_outputs.at(i)->MutableData());
    int num2p = (after_outputs.at(i)->ElementsNum());
    printf("ou%zu(%d): ", i, num2p);
    if (num2p > 10) num2p = 10;
    for (int j = 0; j < num2p; j++) printf("%f, ", d[j]);
    printf("\n");
  }
  return true;
}

NetRunner::~NetRunner() {
  if (loop_ != nullptr) delete loop_;
}

void NetRunner::InitAndFigureInputs() {
  mindspore::lite::Context context;
  context.device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = mindspore::lite::NO_BIND;
  context.device_list_[0].device_info_.cpu_device_info_.enable_float16_ = false;
  context.device_list_[0].device_type_ = mindspore::lite::DT_CPU;
  context.thread_num_ = 2;

  loop_ = mindspore::session::TrainLoop::CreateTrainLoop(ms_file_, &context);
  session_ = loop_->train_session();
  MS_ASSERT(nullptr != session_);

  auto inputs = session_->GetInputs();
  MS_ASSERT(inputs.size() > 1);
  data_index_ = 0;
  label_index_ = 1;
  batch_size_ = inputs[data_index_]->shape()[0];
  data_size_ = inputs[data_index_]->Size() / batch_size_;  // in bytes
  if (verbose_) {
    std::cout << "data size: " << data_size_ << std::endl << "batch size: " << batch_size_ << std::endl;
  }
}

float NetRunner::CalculateAccuracy(int max_tests) {
  AccuracyMonitor test_am(&ds_, 1, max_tests);
  test_am.EpochEnd(TrainLoopCallBackData(true, 0, session_, loop_));
  return 0.0;
}

int NetRunner::InitDB() {
  if (data_size_ != 0) ds_.set_expected_data_size(data_size_);
  int ret = ds_.Init(data_dir_, DS_MNIST_BINARY);
  num_of_classes_ = ds_.num_of_classes();
  if (ds_.test_data().size() == 0) {
    std::cout << "No relevant data was found in " << data_dir_ << std::endl;
    MS_ASSERT(ds_.test_data().size() != 0);
  }

  return ret;
}

int NetRunner::TrainLoop() {
  struct mindspore::lite::StepLRLambda step_lr_lambda(100, 0.9);
  mindspore::lite::LRScheduler step_lr_sched(mindspore::lite::StepLRLambda, static_cast<void *>(&step_lr_lambda), 100);

  mindspore::lite::LossMonitor lm(100);
  // mindspore::lite::ClassificationTrainAccuracyMonitor am(10);
  mindspore::lite::CkptSaver cs(1000, std::string("lenet"));
  AccuracyMonitor test_am(&ds_, 500, 10);
  DataLoader dl(&ds_);

  loop_->Train(cycles_, std::vector<TrainLoopCallBack *>{&dl, &lm, &test_am, &cs, &step_lr_sched});
  return 0;
}

int NetRunner::Main() {
  InitAndFigureInputs();

  InitDB();

  TrainLoop();

  CalculateAccuracy();

  if (cycles_ > 0) {
    auto trained_fn = ms_file_.substr(0, ms_file_.find_last_of('.')) + "_trained_" + std::to_string(cycles_) + ".ms";
    session_->SaveToFile(trained_fn);
  }
  return 0;
}

void NetRunner::Usage() {
  std::cout << "Usage: net_runner -f <.ms model file> -d <data_dir> [-c <num of training cycles>] "
            << "[-v (verbose mode)] [-s <save checkpoint every X iterations>]" << std::endl;
}

bool NetRunner::ReadArgs(int argc, char *argv[]) {
  int opt;
  while ((opt = getopt(argc, argv, "f:e:d:s:ihc:v")) != -1) {
    switch (opt) {
      case 'f':
        ms_file_ = std::string(optarg);
        break;
      case 'e':
        cycles_ = atoi(optarg);
        break;
      case 'd':
        data_dir_ = std::string(optarg);
        break;
      case 'v':
        verbose_ = true;
        break;
      case 's':
        save_checkpoint_ = atoi(optarg);
        break;
      case 'h':
      default:
        Usage();
        return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  NetRunner nr;

  if (nr.ReadArgs(argc, argv)) {
    nr.Main();
  } else {
    return -1;
  }
  return 0;
}
