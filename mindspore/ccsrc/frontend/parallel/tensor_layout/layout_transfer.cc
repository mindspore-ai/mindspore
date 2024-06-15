/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include <functional>
#include <algorithm>
#include <set>
#include <vector>
#include "frontend/parallel/tensor_layout/layout_transfer.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/tensor_layout/prime_generator.h"
#include "frontend/parallel/tensor_layout/layout_utils.h"

namespace mindspore {
namespace parallel {
constexpr size_t INVALID_TENSOR_RANK = 9999;

std::string LayoutTransfer::ToString() const {
  std::ostringstream buffer;
  buffer << std::endl << std::string("from_in_ tensor layout:" + from_in_.ToString());
  buffer << std::endl << std::string("to_in_ tensor layout:" + to_in_.ToString());
  return buffer.str();
}

LayoutTransfer::~LayoutTransfer() = default;

Status LayoutTransfer::Init(const TensorLayout &from_in, const TensorLayout &to_in) {
  from_in_ = from_in;
  to_in_ = to_in;
  MS_LOG(DEBUG) << "LayoutTransfer " << this->ToString();
  Status status = CheckValidTransfer();
  return status;
}
}  // namespace parallel
}  // namespace mindspore
