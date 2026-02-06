/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/utils/pipeline/pipeline_utils.h"

#include "ir/tensor.h"
#include "ir/dtype/number.h"
#include "utils/log_adapter.h"
#include "include/utils/tensor_py.h"

namespace mindspore {
namespace {
std::string ToOrdinal(const size_t &i) {
  constexpr size_t kIndex1 = 1;
  constexpr size_t kIndex2 = 2;
  constexpr size_t kIndex3 = 3;
  auto suffix = "th";
  if (i == kIndex1) {
    suffix = "st";
  } else if (i == kIndex2) {
    suffix = "nd";
  } else if (i == kIndex3) {
    suffix = "rd";
  }
  return std::to_string(i) + suffix;
}

bool CheckArgValid(const py::handle &arg) {
  if (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg)) {
    auto vector_arg = py::cast<py::list>(arg);
    return std::all_of(vector_arg.begin(), vector_arg.end(), CheckArgValid);
  }

  if (py::isinstance<py::dict>(arg)) {
    auto dict_arg = py::cast<py::dict>(arg);
    return std::all_of(dict_arg.begin(), dict_arg.end(), [](const auto &pair) { return CheckArgValid(pair.second); });
  }

  if (tensor::IsTensorPy(arg)) {
    auto tensor = tensor::ConvertToTensor(arg);
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->data_type() == kNumberTypeBool) {
      MS_LOG(INFO) << "It is not recommended to use a tensor of bool data type as network input, which may cause "
                   << "operator compilation failure. For more details, please refer to the FAQ at "
                   << "https://mindspore.cn/search?[AddN]%20input(kNumberTypeBool.";
    }
  }

  return py::isinstance<py::int_>(arg) || py::isinstance<py::float_>(arg) || py::isinstance<py::none>(arg) ||
         py::isinstance<Number>(arg) || py::isinstance<py::str>(arg) || tensor::IsTensorPy(arg) ||
         py::isinstance<tensor::CSRTensor>(arg) || py::isinstance<tensor::COOTensor>(arg);
}
}  // namespace

namespace pipeline {
void CheckArgsValid(const py::object &source, const py::tuple &args) {
  if (!IS_OUTPUT_ON(mindspore::kInfo)) {
    return;
  }
  for (size_t i = 0; i < args.size(); i++) {
    if (!CheckArgValid(args[i])) {
      MS_LOG(INFO) << "The " << ToOrdinal(i + 1) << " arg type is " << py::str(args[i].get_type()).cast<std::string>()
                   << ", value is '" << py::str(args[i]).cast<std::string>() << "'.";
    }
  }
}
}  // namespace pipeline
}  // namespace mindspore
