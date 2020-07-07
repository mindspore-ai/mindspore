/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_DEBUG_TENSOR_DATA_H_
#define MINDSPORE_CCSRC_DEBUG_TENSOR_DATA_H_

#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include "ir/tensor.h"

namespace mindspore {
class TensorData {
 private:
  mindspore::tensor::TensorPtr tensor_ptr;
  std::string name;
  size_t slot;
  int execution_order;

 public:
  TensorData() : slot(0), execution_order(-1) {}

  TensorData(const TensorData &obj) {
    std::cout << "Copy Constructor" << std::endl;
    this->name = obj.name;
    this->execution_order = obj.execution_order;
    this->slot = obj.slot;
    this->tensor_ptr = obj.tensor_ptr;
  }

  ~TensorData() {}

  std::string GetName() { return this->name; }

  mindspore::tensor::TensorPtr GetTensor() { return this->tensor_ptr; }

  size_t GetSlot() { return this->slot; }

  int GetExecutionOrder() { return this->execution_order; }

  int SetExecutionOrder(int execution_order) {
    this->execution_order = execution_order;
    return true;
  }

  int SetName(const std::string &name) {
    this->name = name;
    return true;
  }

  bool SetTensor(mindspore::tensor::TensorPtr out_tensor) {
    this->tensor_ptr = out_tensor;
    return true;
  }

  bool SetSlot(size_t slot) {
    this->slot = slot;
    return true;
  }
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_TENSOR_DATA_H_
