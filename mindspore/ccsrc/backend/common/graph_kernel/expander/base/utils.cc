/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/expander/base/utils.h"

#include <algorithm>
#include <string>
#include <vector>

#include "utils/check_convert_utils.h"

namespace mindspore::graphkernel::expander {
bool CheckAllFormatsSame(const DefaultIrBuilder *ib) {
  auto inputs = ib->inputs();
  if (inputs.empty()) {
    return true;
  }
  const auto &fmt_0 = inputs[0]->GetFormat();
  for (size_t i = 1; i < inputs.size(); i++) {
    MS_LOG_INFO << i << "th format: " << inputs[i]->GetFormat();
    if (inputs[i]->GetFormat() != fmt_0) {
      MS_LOG(INFO) << "The " << i << "th format: " << inputs[i]->GetFormat() << " is not same as 0th format: " << fmt_0
                   << " of op " << ib->name();
      return false;
    }
  }
  return true;
}

bool CheckAttrs(const DefaultIrBuilder *ib, const std::vector<std::string> &attrs) {
  for (auto &a : attrs) {
    if (ib->attrs().count(a) == 0) {
      MS_LOG(INFO) << "attr " << a << " dose not exist. Op: " << ib->name();
      return false;
    }
  }
  return true;
}

bool CheckSupportFormat(const DefaultIrBuilder *ib, const std::vector<std::vector<std::string>> &formats_list) {
  for (auto &formats : formats_list) {
    if (formats.size() != ib->inputs().size()) {
      continue;
    }
    bool match = true;
    for (size_t i = 0; i < formats.size(); i++) {
      if (ib->inputs()[i]->GetFormat() != formats[i]) {
        match = false;
        break;
      }
    }
    if (match) {
      return true;
    }
  }
  MS_LOG(INFO) << "unsupported format for op " << ib->name();
  return false;
}

ShapeVector ExpandDimsInferShape(const ShapeVector &shape, const std::vector<int64_t> &axis) {
  ShapeVector new_shape = shape;
  for (auto x : axis) {
    int64_t rank = static_cast<int64_t>(new_shape.size());
    if (x > rank || x < -rank - 1) {
      MS_LOG(EXCEPTION) << "ExpandDims attr 'axis' value " << x << " is out of range of [" << (-rank - 1) << ", "
                        << rank << "]";
    }
    if (x >= 0) {
      (void)new_shape.insert(new_shape.cbegin() + x, 1LL);
    } else {
      (void)new_shape.insert(new_shape.cbegin() + (x + rank + 1), 1LL);
    }
  }
  return new_shape;
}

std::vector<int64_t> GetAxisList(const ValuePtr &value) {
  std::vector<int64_t> result;
  auto get_int_value = [](const ValuePtr &value) -> int64_t {
    return value->isa<Int64Imm>() ? GetValue<int64_t>(value) : static_cast<int64_t>(GetValue<int>(value));
  };
  if (value->isa<ValueSequence>()) {
    const auto &vals = value->cast<ValueSequencePtr>()->value();
    (void)std::transform(vals.begin(), vals.end(), std::back_inserter(result), get_int_value);
  } else if (value->isa<tensor::Tensor>()) {
    result = CheckAndConvertUtils::CheckTensorIntValue("axes value", value, "GetAxisList");
  } else {
    result.push_back(get_int_value(value));
  }
  return result;
}
}  // namespace mindspore::graphkernel::expander
