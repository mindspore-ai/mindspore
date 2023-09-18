/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_PYTHON_SRC_COMMON_PYBIND_H_
#define MINDSPORE_LITE_PYTHON_SRC_COMMON_PYBIND_H_

#include <vector>
#include <algorithm>
#include <memory>
#include "include/api/types.h"
#include "src/common/log_adapter.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace mindspore::lite {
namespace py = pybind11;
using MSTensorPtr = std::shared_ptr<MSTensor>;
static inline std::vector<MSTensor> MSTensorPtrToMSTensor(const std::vector<MSTensorPtr> &tensors_ptr) {
  std::vector<MSTensor> tensors;
  for (auto &item : tensors_ptr) {
    if (item == nullptr) {
      MS_LOG(ERROR) << "Tensor object cannot be nullptr";
      return {};
    }
    tensors.push_back(*item);
  }
  return tensors;
}

static inline std::vector<MSTensorPtr> MSTensorToMSTensorPtr(const std::vector<MSTensor> &tensors) {
  std::vector<MSTensorPtr> tensors_ptr;
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(tensors_ptr),
                 [](auto &item) { return std::make_shared<MSTensor>(item); });
  return tensors_ptr;
}
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_PYTHON_SRC_COMMON_PYBIND_H_
