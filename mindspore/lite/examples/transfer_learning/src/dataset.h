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

#ifndef MINDSPORE_LITE_EXAMPLES_TRANSFER_LEARNING_SRC_DATASET_H_
#define MINDSPORE_LITE_EXAMPLES_TRANSFER_LEARNING_SRC_DATASET_H_

#include <tuple>
#include <string>
#include <vector>

using DataLabelTuple = std::tuple<char *, int>;
using FileTuple = std::tuple<int, std::string>;

enum database_type { DS_CIFAR10_BINARY = 0, DS_MNIST_BINARY, DS_OTHER };

char *ReadFile(const std::string &file, size_t *size);  // utility function

class DataSet {
 public:
  DataSet() {}
  ~DataSet();

  int Init(const std::string &data_base_directory, database_type type = DS_OTHER);

  const std::vector<DataLabelTuple> &train_data() const { return train_data_; }
  const std::vector<DataLabelTuple> &test_data() const { return test_data_; }
  const std::vector<DataLabelTuple> &val_data() const { return val_data_; }
  unsigned int num_of_classes() { return num_of_classes_; }
  void set_expected_data_size(unsigned int expected_data_size) { expected_data_size_ = expected_data_size; }
  unsigned int expected_data_size() { return expected_data_size_; }

 private:
  std::vector<FileTuple> ReadDir(const std::string dpath);
  void InitializeBMPFoldersDatabase(std::string dpath);

  std::vector<DataLabelTuple> train_data_;
  std::vector<DataLabelTuple> test_data_;
  std::vector<DataLabelTuple> val_data_;
  unsigned int num_of_classes_ = 0;
  unsigned int expected_data_size_ = 0;
};

#endif  // MINDSPORE_LITE_EXAMPLES_TRANSFER_LEARNING_SRC_DATASET_H_
