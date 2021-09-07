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
#include <getopt.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <utility>
#include "include/context.h"
#include "include/dataset/datasets.h"
#include "include/dataset/transforms.h"
#include "include/dataset/vision_lite.h"
#include "include/train/accuracy_metrics.h"
#include "include/train/ckpt_saver.h"
#include "include/train/classification_train_accuracy_monitor.h"
#include "include/train/loss_monitor.h"
#include "include/train/lr_scheduler.h"
#include "include/train/train_cfg.h"
#include "include/train/train_session.h"
#include "src/utils.h"

using mindspore::dataset::Dataset;
using mindspore::dataset::Mnist;
using mindspore::dataset::SequentialSampler;
using mindspore::dataset::TensorOperation;
using mindspore::dataset::transforms::TypeCast;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::Resize;
using mindspore::lite::AccuracyMetrics;
using mindspore::lite::Model;
using mindspore::session::TrainLoopCallBack;
using mindspore::session::TrainLoopCallBackData;

constexpr int kPrintNum = 10;
constexpr float kScalePoint = 255.0f;
constexpr int kBatchSize = 2;
constexpr int kNCHWDims = 4;
constexpr int kNCHWCDim = 2;
constexpr int kPrintTimes = 100;
constexpr int kSaveSteps = 1000;
constexpr float kGammaFactor = 0.7f;
constexpr static int kElem2Print = 10;

class Rescaler : public mindspore::session::TrainLoopCallBack {
 public:
  explicit Rescaler(float scale) : scale_(scale) {
    if (scale_ == 0) {
      scale_ = 1.0;
    }
  }
  ~Rescaler() override = default;
  void StepBegin(const mindspore::session::TrainLoopCallBackData &cb_data) override {
    auto inputs = cb_data.session_->GetInputs();
    auto *input_data = reinterpret_cast<float *>(inputs.at(0)->MutableData());
    for (int k = 0; k < inputs.at(0)->ElementsNum(); k++) input_data[k] /= scale_;
  }

 private:
  float scale_ = 1.0;
};

class Measurement : public mindspore::session::TrainLoopCallBack {
 public:
  explicit Measurement(unsigned int epochs)
      : epochs_(epochs), time_avg_(std::chrono::duration<double, std::milli>(0)) {}
  ~Measurement() override = default;
  void EpochBegin(const mindspore::session::TrainLoopCallBackData &cb_data) override {
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  int EpochEnd(const mindspore::session::TrainLoopCallBackData &cb_data) override {
    end_time_ = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration<double, std::milli>(end_time_ - start_time_);
    time_avg_ += time;
    return mindspore::session::RET_CONTINUE;
  }
  void End(const mindspore::session::TrainLoopCallBackData &cb_data) override {
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

// Definition of verbose callback function after forwarding operator.
bool after_callback(const std::vector<mindspore::tensor::MSTensor *> &after_inputs,
                    const std::vector<mindspore::tensor::MSTensor *> &after_outputs,
                    const mindspore::CallBackParam &call_param) {
  printf("%s\n", call_param.node_name.c_str());
  for (size_t i = 0; i < after_inputs.size(); i++) {
    int num2p = (after_inputs.at(i)->ElementsNum());
    printf("in%zu(%d): ", i, num2p);
    if (num2p > kPrintNum) num2p = kPrintNum;
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
    if (num2p > kElem2Print) {
      num2p = kElem2Print;
    }
    for (int j = 0; j < num2p; j++) printf("%f, ", d[j]);
    printf("\n");
  }
  return true;
}

NetRunner::~NetRunner() {
  if (loop_ != nullptr) delete loop_;
  if (session_ != nullptr) delete session_;
}

void NetRunner::InitAndFigureInputs() {
  mindspore::lite::Context context;
  context.device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = mindspore::lite::NO_BIND;
  context.device_list_[0].device_info_.cpu_device_info_.enable_float16_ = enable_fp16_;
  context.device_list_[0].device_type_ = mindspore::lite::DT_CPU;
  context.thread_num_ = 2;

  mindspore::lite::TrainCfg train_cfg;
  train_cfg.mix_precision_cfg_.is_raw_mix_precision_ = is_raw_mix_precision_;
  session_ = mindspore::session::TrainSession::CreateTrainSession(ms_file_, &context, true, &train_cfg);
  MS_ASSERT(session_ != nullptr);

  session_->SetupVirtualBatch(virtual_batch_);
  loop_ = mindspore::session::TrainLoop::CreateTrainLoop(session_);

  if (verbose_) {
    loop_->SetKernelCallBack(nullptr, after_callback);
  }
  acc_metrics_ = std::shared_ptr<AccuracyMetrics>(new AccuracyMetrics);

  loop_->Init({acc_metrics_.get()});

  auto inputs = session_->GetInputs();
  MS_ASSERT(inputs.size() > 1);
  auto nhwc_input_dims = inputs.at(0)->shape();
  MS_ASSERT(nhwc_input_dims.size() == kNCHWDims);
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

  Rescaler rescale(kScalePoint);

  loop_->Eval(test_ds_.get(), std::vector<TrainLoopCallBack *>{&rescale});
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
  mindspore::lite::LossMonitor lm(kPrintTimes);
  mindspore::lite::ClassificationTrainAccuracyMonitor am(1);

  mindspore::lite::CkptSaver cs(kSaveSteps, std::string("lenet"));
  Rescaler rescale(kScalePoint);
  Measurement measure(epochs_);

  if (virtual_batch_ > 0) {
    loop_->Train(epochs_, train_ds_.get(), std::vector<TrainLoopCallBack *>{&rescale, &lm, &cs, &am, &measure});
  } else {
    struct mindspore::lite::StepLRLambda step_lr_lambda(1, kGammaFactor);
    mindspore::lite::LRScheduler step_lr_sched(mindspore::lite::StepLRLambda, static_cast<void *>(&step_lr_lambda), 1);
    loop_->Train(epochs_, train_ds_.get(),
                 std::vector<TrainLoopCallBack *>{&rescale, &lm, &cs, &am, &step_lr_sched, &measure});
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
    session_->Export(trained_fn);
  }
  return 0;
}

void NetRunner::Usage() {
  std::cout << "Usage: net_runner -f <.ms model file> -d <data_dir> [-e <num of training epochs>] "
            << "[-v (verbose mode)] [-s <save checkpoint every X iterations>]" << std::endl;
}

bool NetRunner::ReadArgs(int argc, char *argv[]) {
  int opt;
  while ((opt = getopt(argc, argv, "f:e:d:s:ihc:vmob:")) != -1) {
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
      case 'm':
        is_raw_mix_precision_ = true;
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
