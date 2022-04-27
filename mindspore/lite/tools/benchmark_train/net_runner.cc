/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/benchmark_train/net_runner.h"
#include "tools/benchmark_train/net_train.h"
#include <getopt.h>
#include <malloc.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <utility>
#include <chrono>
#include "include/api/types.h"
#include "include/api/context.h"
#include "include/api/serialization.h"
#include "include/api/callback/loss_monitor.h"
#include "include/api/metrics/accuracy.h"
#include "include/api/callback/ckpt_saver.h"
#include "include/api/callback/train_accuracy.h"
#include "include/api/callback/lr_scheduler.h"
#include "include/dataset/datasets.h"
#include "include/dataset/vision_lite.h"
#include "include/dataset/transforms.h"
#include "include/api/cfg.h"
#include "include/api/net.h"

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

constexpr int kNCHWCDim = 2;
constexpr int kPrintTimes = 100;
constexpr float kBetta1 = 0.9f;
constexpr float kBetta2 = 0.999f;

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
      : time_avg_(std::chrono::duration<double, std::milli>(0)), epochs_(epochs) {}
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

mindspore::Status NetRunner::InitAndFigureInputs() {
  auto context = std::make_shared<mindspore::Context>();
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  cpu_context->SetEnableFP16(enable_fp16_);
  context->MutableDeviceInfo().push_back(cpu_context);

  graph_ = new (std::nothrow) mindspore::Graph(mindspore::Graph::Type::kExpressionGraph);
  if (graph_ == nullptr) {
    std::cout << "Cannot allocate graph" << std::endl;
    return mindspore::kLiteMemoryFailed;
  }
  auto status = mindspore::Serialization::Load(ms_file_, mindspore::kMindIR, graph_);
  if (status != mindspore::kSuccess) {
    std::cout << "Error " << status << " during serialization of graph " << ms_file_;
    return status;
  }
  auto net = std::make_unique<mindspore::Net>(*graph_);
  auto input_shape = net->InputShape(0);
  auto label_shape = net->OutputShape(0);
  auto inputM = mindspore::NN::Input(input_shape);
  auto labelM = mindspore::NN::Input(label_shape);
  auto label = labelM->Create("label");
  auto input = inputM->Create("input");

  auto cfg = std::make_shared<mindspore::TrainCfg>();
  if (enable_fp16_) {
    cfg.get()->optimization_level_ = mindspore::kO2;
  }

  model_ = new (std::nothrow) mindspore::Model();
  if (model_ == nullptr) {
    std::cout << "model allocation failed" << std::endl;
    return mindspore::kLiteMemoryFailed;
  }
  mindspore::SoftMaxCrossEntropyCfg softmax_ce_cfg;
  softmax_ce_cfg.reduction = "none";
  auto netWithLoss = mindspore::NN::GraphWithLoss(graph_, mindspore::NN::SoftmaxCrossEntropy(softmax_ce_cfg));
  mindspore::AdamConfig AdamCfg;
  AdamCfg.beta1_ = kBetta1;
  AdamCfg.beta2_ = kBetta2;
  AdamCfg.eps_ = 1e-8;
  AdamCfg.learning_rate_ = 1e-2;
  auto optimizer = mindspore::NN::Adam(net->trainable_params(), AdamCfg);
  status = model_->Build(mindspore::GraphCell(*netWithLoss), optimizer, {input, label}, context, cfg);
  if (status != mindspore::kSuccess) {
    std::cout << "Error " << status << " during build of model " << ms_file_ << std::endl;
    return status;
  }
  delete graph_;
  graph_ = nullptr;
  auto inputs = model_->GetInputs();
  if (inputs.size() < 1) {
    return mindspore::kLiteError;
  }
  auto nhwc_input_dims = inputs.at(0).Shape();
  batch_size_ = nhwc_input_dims.at(0);
  h_ = nhwc_input_dims.at(1);
  w_ = nhwc_input_dims.at(kNCHWCDim);
  return mindspore::kSuccess;
}

int NetRunner::CompareOutput(const std::vector<mindspore::MSTensor> &outputs) {
  std::cout << "================ Comparing Forward Output data ================" << std::endl;
  float total_bias = 0;
  int total_size = 0;
  bool has_error = false;
  int i = 1;
  for (auto &tensor : outputs) {
    std::cout << "output is tensor " << tensor.Name() << "\n";
    auto output = tensor.Data();
    size_t size;
    std::string output_file = flags_->data_file_ + std::to_string(i) + ".bin";
    auto bin_buf = std::unique_ptr<float[]>(mindspore::lite::NetTrain::ReadFileBuf(output_file.c_str(), &size));
    if (bin_buf == nullptr) {
      MS_LOG(ERROR) << "ReadFile return nullptr";
      std::cout << "ReadFile return nullptr" << std::endl;
      return mindspore::kLiteNullptr;
    }
    if (size != tensor.DataSize()) {
      MS_LOG(ERROR) << "Output buffer and output file differ by size. Tensor size: " << tensor.DataSize()
                    << ", read size: " << size;
      std::cout << "Output buffer and output file differ by size. Tensor size: " << tensor.DataSize()
                << ", read size: " << size << std::endl;
      return mindspore::kLiteError;
    }
    float bias = mindspore::lite::NetTrain::CompareData<float>(bin_buf.get(), tensor.ElementNum(),
                                                               reinterpret_cast<const float *>(output.get()));
    if (bias >= 0) {
      total_bias += bias;
      total_size++;
    } else {
      has_error = true;
      break;
    }
    i++;
  }

  if (!has_error) {
    float mean_bias;
    if (total_size != 0) {
      mean_bias = total_bias / total_size * kPrintTimes;
    } else {
      mean_bias = 0;
    }

    std::cout << "Mean bias of all nodes/tensors: " << mean_bias << "%"
              << " threshold is:" << this->flags_->accuracy_threshold_ << std::endl;
    std::cout << "=======================================================" << std::endl << std::endl;

    if (mean_bias > this->flags_->accuracy_threshold_) {
      MS_LOG(INFO) << "Mean bias of all nodes/tensors is too big: " << mean_bias << "%";
      std::cout << "Mean bias of all nodes/tensors is too big: " << mean_bias << "%" << std::endl;
      return mindspore::kLiteError;
    } else {
      return mindspore::kSuccess;
    }
  } else {
    MS_LOG(ERROR) << "Error in CompareData";
    std::cout << "Error in CompareData" << std::endl;
    std::cout << "=======================================================" << std::endl << std::endl;
    return mindspore::kSuccess;
  }
}

void NetRunner::CheckSum(const mindspore::MSTensor &tensor, std::string node_type, int id, std::string in_out) {
  constexpr int kPrintLen = 4;
  int tensor_size = tensor.ElementNum();
  const void *data = tensor.Data().get();
  const float *fdata = reinterpret_cast<const float *>(data);
  mindspore::DataType type = tensor.DataType();
  std::cout << node_type << " " << in_out << id << std::endl;
  std::cout << "tensor name: " << tensor.Name() << std::endl;
  if ((tensor_size) == 0 || (data == nullptr)) {
    std::cout << "Empty tensor" << std::endl;
    return;
  }
  switch (type) {
    case mindspore::DataType::kNumberTypeFloat32:
      std::cout << "sum=" << mindspore::lite::TensorSum<float>(data, tensor_size) << std::endl;
      std::cout << "data: ";
      for (int i = 0; i <= kPrintLen && i < tensor_size; i++) {
        std::cout << static_cast<float>(fdata[i]) << ", ";
      }
      std::cout << std::endl;
      break;
    case mindspore::DataType::kNumberTypeInt32:
      std::cout << "sum=" << mindspore::lite::TensorSum<int>(data, tensor_size) << std::endl;
      break;
    default:
      std::cout << "unsupported type:" << static_cast<int>(type) << std::endl;
      break;
  }
}

int NetRunner::InitCallbackParameter() {
  // after callback
  after_call_back_ = [&](const std::vector<mindspore::MSTensor> &after_inputs,
                         const std::vector<mindspore::MSTensor> &after_outputs,
                         const mindspore::MSCallBackParam &call_param) {
    if (after_inputs.empty()) {
      MS_LOG(INFO) << "The num of after inputs is empty";
    }
    if (after_outputs.empty()) {
      MS_LOG(INFO) << "The num of after outputs is empty";
    }
    if (flags_->layer_checksum_) {
      for (size_t i = 0; i < after_inputs.size(); i++) {
        CheckSum(after_inputs.at(i), call_param.node_type, i, "in");
      }
      for (size_t i = 0; i < after_outputs.size(); i++) {
        CheckSum(after_outputs.at(i), call_param.node_type, i, "out");
      }
      std::cout << std::endl;
    }
    return true;
  };
  return false;
}

int NetRunner::RunOnce() {
  auto inputs = model_->GetInputs();
  std::vector<mindspore::MSTensor> output;
  auto status = LoadInput(&inputs);
  if (status != mindspore::kSuccess) {
    std::cout << "cannot load data";
    return status;
  }
  model_->SetTrainMode(true);
  model_->RunStep(nullptr, nullptr);
  model_->SetTrainMode(false);
  model_->Predict(inputs, &output, nullptr, nullptr);
  return CompareOutput(output);
}

int NetRunner::LoadInput(std::vector<mindspore::MSTensor> *ms_inputs) {
  auto status = ReadInputFile(ms_inputs);
  if (status != mindspore::kSuccess) {
    std::cout << "Read Input File error, " << status << std::endl;
    MS_LOG(ERROR) << "Read Input File error, " << status;
    return status;
  }
  return mindspore::kSuccess;
}

int NetRunner::ReadInputFile(std::vector<mindspore::MSTensor> *ms_inputs) {
  if (ms_inputs->empty()) {
    std::cout << "no inputs to input" << std::endl;
    return mindspore::kLiteError;
  }
  for (size_t i = 0; i < ms_inputs->size(); i++) {
    auto cur_tensor = ms_inputs->at(i);
    if (cur_tensor == nullptr) {
      std::cout << "empty tensor " << i << std::endl;
      MS_LOG(ERROR) << "empty tensor " << i;
    }
    size_t size;
    std::string file_name = flags_->in_data_file_ + std::to_string(i + 1) + ".bin";
    auto bin_buf = std::unique_ptr<float[]>(mindspore::lite::NetTrain::ReadFileBuf(file_name.c_str(), &size));
    if (bin_buf == nullptr) {
      MS_LOG(ERROR) << "ReadFile return nullptr";
      std::cout << "ReadFile return nullptr" << std::endl;
      return mindspore::kLiteNullptr;
    }
    auto tensor_data_size = cur_tensor.DataSize();
    if (size != tensor_data_size) {
      std::cout << "Input binary file size error, required: " << tensor_data_size << ", in fact: " << size
                << " ,file_name: " << file_name.c_str() << std::endl;
      MS_LOG(ERROR) << "Input binary file size error, required: " << tensor_data_size << ", in fact: " << size
                    << " ,file_name: " << file_name.c_str();
      return mindspore::kLiteError;
    }
    auto input_data = cur_tensor.MutableData();
    memcpy(input_data, bin_buf.get(), tensor_data_size);
  }
  return mindspore::kSuccess;
}

int NetRunner::Main() {
  ms_file_ = flags_->model_file_;
  InitCallbackParameter();
  auto status = InitAndFigureInputs();
  if (status != mindspore::kSuccess) {
    std::cout << "failed to initialize network" << std::endl;
    return status.StatusCode();
  }
  return RunOnce();
}

int CallBack(mindspore::lite::NetTrainFlags *flags) {
  NetRunner nr(flags);
  return nr.Main();
}

int init = mindspore::lite::NetTrain::SetNr(CallBack);
