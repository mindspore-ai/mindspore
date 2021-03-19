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

#ifndef MINDSPORE_LITE_EXAMPLES_TRAIN_LENET_SRC_NET_RUNNER_H_
#define MINDSPORE_LITE_EXAMPLES_TRAIN_LENET_SRC_NET_RUNNER_H_

#include <tuple>
#include <iomanip>
#include <map>
#include <vector>
#include <memory>
#include <string>
#include "include/train/train_loop.h"
#include "include/train/accuracy_metrics.h"
#include "include/ms_tensor.h"
#include "include/datasets.h"

using mindspore::dataset::Dataset;
using mindspore::lite::AccuracyMetrics;

class NetRunner {
 public:
  int Main();
  bool ReadArgs(int argc, char *argv[]);
  ~NetRunner();

 private:
  void Usage();
  void InitAndFigureInputs();
  int InitDB();
  int TrainLoop();
  float CalculateAccuracy(int max_tests = 0);
  float GetLoss() const;
  mindspore::tensor::MSTensor *SearchOutputsForSize(size_t size) const;

  mindspore::session::TrainSession *session_ = nullptr;
  mindspore::session::TrainLoop *loop_ = nullptr;

  std::shared_ptr<Dataset> train_ds_;
  std::shared_ptr<Dataset> test_ds_;
  std::shared_ptr<AccuracyMetrics> acc_metrics_;

  std::string ms_file_ = "";
  std::string data_dir_ = "";
  unsigned int epochs_ = 10;
  bool verbose_ = false;
  int save_checkpoint_ = 0;
  int batch_size_ = 32;
  int h_ = 32;
  int w_ = 32;
};

#endif  // MINDSPORE_LITE_EXAMPLES_TRAIN_LENET_SRC_NET_RUNNER_H_
