/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_EXAMPLES_UNIFIED_API_SRC_NET_RUNNER_H_
#define MINDSPORE_LITE_EXAMPLES_UNIFIED_API_SRC_NET_RUNNER_H_

#include <tuple>
#include <iomanip>
#include <map>
#include <vector>
#include <memory>
#include <string>
#include "include/api/model.h"
#include "include/api/graph.h"
#include "include/api/metrics/accuracy.h"
#include "include/dataset/datasets.h"

using mindspore::AccuracyMetrics;
using mindspore::dataset::Dataset;

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

  mindspore::Model *model_ = nullptr;
  mindspore::Graph *graph_ = nullptr;

  std::shared_ptr<Dataset> train_ds_;
  std::shared_ptr<Dataset> test_ds_;
  std::shared_ptr<AccuracyMetrics> acc_metrics_;

  std::string ms_file_ = "";
  std::string data_dir_ = "";
  unsigned int epochs_ = 10;
  bool verbose_ = false;
  bool enable_fp16_ = false;
  int virtual_batch_ = -1;
  int save_checkpoint_ = 0;
  int batch_size_ = 32;
  int h_ = 32;
  int w_ = 32;
};

#endif  // MINDSPORE_LITE_EXAMPLES_UNIFIED_API_SRC_NET_RUNNER_H_
