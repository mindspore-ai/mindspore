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

#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
namespace {
void ErrorReport(const std::vector<NumberContainer> &values) {
  std::ostringstream oss;
  oss << "The value should be int32, int64 or ValueAny, but got[";
  size_t length = values.size();
  for (size_t i = 0; i < length; ++i) {
    const auto &value = values[i].value_;
    if (value->isa<ValueAny>()) {
      oss << "ValueAny";
    } else {
      oss << value->DumpText();
    }
    if (i < length - 1) {
      oss << ", ";
    } else {
      oss << "].";
    }
  }
  MS_LOG(ERROR) << oss.str();
}

std::vector<ValuePtr> ConvertInt(const std::vector<NumberContainer> &values) {
  std::vector<ValuePtr> value_vec{};
  for (const auto &v : values) {
    auto value = v.value_;
    if (value->isa<Int32Imm>()) {
      auto x = std::make_shared<Int64Imm>(static_cast<int64_t>(value->cast<Int32ImmPtr>()->value()));
      value_vec.push_back(std::move(x));
    } else if (value->isa<Int64Imm>() || value->isa<ValueAny>()) {
      value_vec.push_back(std::move(value));
    } else {
      ErrorReport(values);
      break;
    }
  }
  return value_vec;
}
}  // namespace

ValuePtr CreatePyIntTuple(const std::vector<NumberContainer> &values) {
  auto value_vec = ConvertInt(values);
  return std::make_shared<ValueTuple>(std::move(value_vec));
}

ValuePtr CreatePyIntList(const std::vector<NumberContainer> &values) {
  auto value_vec = ConvertInt(values);
  return std::make_shared<ValueList>(std::move(value_vec));
}
}  // namespace ops
}  // namespace mindspore
