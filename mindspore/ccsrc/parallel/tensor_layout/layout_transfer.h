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

#ifndef MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_LAYOUT_TRANSFER_H_
#define MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_LAYOUT_TRANSFER_H_

#include <string>
#include "parallel/status.h"
#include "parallel/tensor_layout/tensor_layout.h"

namespace mindspore {
namespace parallel {

class LayoutTransfer {
 public:
  LayoutTransfer() = default;
  virtual ~LayoutTransfer() = 0;
  std::string ToString() const;
  Status Init(const TensorLayout& from_in, const TensorLayout& to_in);
  TensorLayout from_in() const { return from_in_; }
  TensorLayout to_in() const { return to_in_; }

 protected:
  bool IsSameTensorShape() const { return from_in_.IsSameTensorShape(to_in_); }
  bool IsSameDeviceArrangement() const { return from_in_.IsSameDeviceArrangement(to_in_); }

  TensorLayout from_in_;
  TensorLayout to_in_;

 private:
  virtual Status CheckValidTransfer() = 0;
};

}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_LAYOUT_TRANSFER_H_
