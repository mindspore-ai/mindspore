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
#include <string>
#include "include/train_session.h"
#include "include/train/train_loop.h"
#include "include/ms_tensor.h"
#include "src/dataset.h"

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
  std::vector<int> FillInputData(const std::vector<DataLabelTuple> &dataset, bool is_train_set = false) const;
  float CalculateAccuracy(int max_tests = -1);
  float GetLoss() const;
  mindspore::tensor::MSTensor *SearchOutputsForSize(size_t size) const;

  DataSet ds_;
  mindspore::session::TrainSession *session_ = nullptr;
  mindspore::session::TrainLoop *loop_ = nullptr;

  std::string ms_file_ = "";
  std::string data_dir_ = "";
  size_t data_size_ = 0;
  size_t batch_size_ = 0;
  unsigned int cycles_ = 100;
  int data_index_ = 0;
  int label_index_ = -1;
  int num_of_classes_ = 0;
  bool verbose_ = false;
  int save_checkpoint_ = 0;
  static unsigned int seed_;
};

#endif  // MINDSPORE_LITE_EXAMPLES_TRAIN_LENET_SRC_NET_RUNNER_H_
