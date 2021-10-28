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
#include <stdio.h>
#include <malloc.h>
#include <cstring>
#include <chrono>
#include <iostream>
#include <fstream>
#include <utility>
#include "include/api/context.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/api/callback/loss_monitor.h"
#include "include/api/metrics/accuracy.h"
#include "include/api/callback/ckpt_saver.h"
#include "include/api/callback/train_accuracy.h"
#include "include/api/callback/lr_scheduler.h"
#include "src/utils.h"
#include "include/dataset/datasets.h"
#include "include/dataset/vision_lite.h"
#include "include/dataset/transforms.h"

using mindspore::AccuracyMetrics;
using mindspore::Model;
using mindspore::TrainAccuracy;
using mindspore::TrainCallBack;
using mindspore::TrainCallBackData;
using mindspore::dataset::Dataset;
using mindspore::dataset::Mnist;
using mindspore::dataset::SequentialSampler;
using mindspore::dataset::TensorOperation;
using mindspore::dataset::transforms::TypeCast;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::Resize;

constexpr int kPrintNum = 10;
constexpr float kScalePoint = 255.0f;
constexpr int kBatchSize = 2;
constexpr int kNCHWDims = 4;
constexpr int kNCHWCDim = 2;
constexpr int kPrintTimes = 100;
constexpr int kSaveEpochs = 3;
constexpr float kGammaFactor = 0.7f;
constexpr static int kElem2Print = 10;

class Rescaler : public mindspore::TrainCallBack {
 public:
  explicit Rescaler(float scale) : scale_(scale) {
    if (scale_ == 0) {
      scale_ = 1.0;
    }
  }
  ~Rescaler() override = default;
  void StepBegin(const mindspore::TrainCallBackData &cb_data) override {
    auto inputs = cb_data.model_->GetInputs();
    auto *input_data = reinterpret_cast<float *>(inputs.at(0).MutableData());
    for (int k = 0; k < inputs.at(0).ElementNum(); k++) input_data[k] /= scale_;
  }

 private:
  float scale_ = 1.0;
};

/* This is an example of a user defined Callback to measure memory and latency of execution */
class Measurement : public mindspore::TrainCallBack {
 public:
  explicit Measurement(unsigned int epochs)
      : epochs_(epochs), time_avg_(std::chrono::duration<double, std::milli>(0)) {}
  ~Measurement() override = default;
  void EpochBegin(const mindspore::TrainCallBackData &cb_data) override {
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  mindspore::CallbackRetValue EpochEnd(const mindspore::TrainCallBackData &cb_data) override {
    end_time_ = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration<double, std::milli>(end_time_ - start_time_);
    time_avg_ += time;
    return mindspore::kContinue;
  }
  void End(const mindspore::TrainCallBackData &cb_data) override {
    if (epochs_ > 0) {
      std::cout << "AvgRunTime: " << time_avg_.count() / epochs_ << " ms" << std::endl;
    }

    struct mallinfo info = mallinfo();
    std::cout << "Total allocation: " << info.arena + info.hblkhd << std::endl;
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_time_;
  std::chrono::duration<double, std::milli> time_avg_;
  unsigned int epochs_;
};

NetRunner::~NetRunner() {
  if (model_ != nullptr) {
    delete model_;
  }
  if (graph_ != nullptr) {
    delete graph_;
  }
}

void NetRunner::InitAndFigureInputs() {
  auto context = std::make_shared<mindspore::Context>();
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  cpu_context->SetEnableFP16(enable_fp16_);
  context->MutableDeviceInfo().push_back(cpu_context);

  graph_ = new mindspore::Graph();
  MS_ASSERT(graph_ != nullptr);

  auto status = mindspore::Serialization::Load(ms_file_, mindspore::kMindIR, graph_);
  if (status != mindspore::kSuccess) {
    std::cout << "Error " << status << " during serialization of graph " << ms_file_;
    MS_ASSERT(status != mindspore::kSuccess);
  }

  auto cfg = std::make_shared<mindspore::TrainCfg>();
  if (enable_fp16_) {
    cfg.get()->optimization_level_ = mindspore::kO2;
  }

  model_ = new mindspore::Model();
  MS_ASSERT(model_ != nullptr);

  status = model_->Build(mindspore::GraphCell(*graph_), context, cfg);
  if (status != mindspore::kSuccess) {
    std::cout << "Error " << status << " during build of model " << ms_file_;
    MS_ASSERT(status != mindspore::kSuccess);
  }

  acc_metrics_ = std::shared_ptr<AccuracyMetrics>(new AccuracyMetrics);
  MS_ASSERT(acc_metrics_ != nullptr);
  model_->InitMetrics({acc_metrics_.get()});

  auto inputs = model_->GetInputs();
  MS_ASSERT(inputs.size() >= 1);
  auto nhwc_input_dims = inputs.at(0).Shape();

  batch_size_ = nhwc_input_dims.at(0);
  h_ = nhwc_input_dims.at(1);
  w_ = nhwc_input_dims.at(kNCHWCDim);
}

float NetRunner::CalculateAccuracy(int max_tests) {
  test_ds_ = Mnist(data_dir_ + "/test", "all");
  TypeCast typecast_f(mindspore::DataType::kNumberTypeFloat32);
  Resize resize({h_, w_});
  test_ds_ = test_ds_->Map({&resize, &typecast_f}, {"image"});

  TypeCast typecast(mindspore::DataType::kNumberTypeInt32);
  test_ds_ = test_ds_->Map({&typecast}, {"label"});
  test_ds_ = test_ds_->Batch(batch_size_, true);

  model_->Evaluate(test_ds_, {});
  std::cout << "Accuracy is " << acc_metrics_->Eval() << std::endl;

  return 0.0;
}

int NetRunner::InitDB() {
  train_ds_ = Mnist(data_dir_ + "/train", "all", std::make_shared<SequentialSampler>(0, 0));

  TypeCast typecast_f(mindspore::DataType::kNumberTypeFloat32);
  Resize resize({h_, w_});
  train_ds_ = train_ds_->Map({&resize, &typecast_f}, {"image"});

  TypeCast typecast(mindspore::DataType::kNumberTypeInt32);
  train_ds_ = train_ds_->Map({&typecast}, {"label"});

  train_ds_ = train_ds_->Batch(batch_size_, true);

  if (verbose_) {
    std::cout << "DatasetSize is " << train_ds_->GetDatasetSize() << std::endl;
  }
  if (train_ds_->GetDatasetSize() == 0) {
    std::cout << "No relevant data was found in " << data_dir_ << std::endl;
    MS_ASSERT(train_ds_->GetDatasetSize() != 0);
  }
  return 0;
}

int NetRunner::TrainLoop() {
  mindspore::LossMonitor lm(kPrintTimes);
  mindspore::TrainAccuracy am(1);

  mindspore::CkptSaver cs(kSaveEpochs, std::string("lenet"));
  Rescaler rescale(kScalePoint);
  Measurement measure(epochs_);

  if (virtual_batch_ > 0) {
    model_->Train(epochs_, train_ds_, {&rescale, &lm, &cs, &measure});
  } else {
    struct mindspore::StepLRLambda step_lr_lambda(1, kGammaFactor);
    mindspore::LRScheduler step_lr_sched(mindspore::StepLRLambda, static_cast<void *>(&step_lr_lambda), 1);
    model_->Train(epochs_, train_ds_, {&rescale, &lm, &cs, &am, &step_lr_sched, &measure});
  }

  return 0;
}

int NetRunner::Main() {
  InitAndFigureInputs();

  InitDB();

  TrainLoop();

  CalculateAccuracy();

  if (epochs_ > 0) {
    auto trained_fn = ms_file_.substr(0, ms_file_.find_last_of('.')) + "_trained.ms";
    mindspore::Serialization::ExportModel(*model_, mindspore::kFlatBuffer, trained_fn, mindspore::kNoQuant, false);
    trained_fn = ms_file_.substr(0, ms_file_.find_last_of('.')) + "_infer.ms";
    mindspore::Serialization::ExportModel(*model_, mindspore::kFlatBuffer, trained_fn, mindspore::kNoQuant, true);
  }
  return 0;
}

void NetRunner::Usage() {
  std::cout << "Usage: net_runner -f <.ms model file> -d <data_dir> [-e <num of training epochs>] "
            << "[-v (verbose mode)] [-s <save checkpoint every X iterations>]" << std::endl;
}

bool NetRunner::ReadArgs(int argc, char *argv[]) {
  int opt;
  while ((opt = getopt(argc, argv, "f:e:d:s:ihc:vob:")) != -1) {
    switch (opt) {
      case 'f':
        ms_file_ = std::string(optarg);
        break;
      case 'e':
        epochs_ = atoi(optarg);
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
      case 'o':
        enable_fp16_ = true;
        break;
      case 'b':
        virtual_batch_ = atoi(optarg);
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
