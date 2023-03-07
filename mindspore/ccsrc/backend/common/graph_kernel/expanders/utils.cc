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
#include "backend/common/graph_kernel/expanders/utils.h"

#include <algorithm>
#include <string>
#include <vector>
#include <unordered_set>

#include "backend/common/graph_kernel/model/lite_graph.h"
#include "backend/common/graph_kernel/model/node.h"

namespace mindspore::graphkernel::expanders {
constexpr int OFFSET1 = 1;
constexpr int OFFSET2 = 2;
constexpr int OFFSET3 = 3;
constexpr int OFFSET4 = 4;
inner::LiteGraphPtr OpDesc::Run(const BaseInfoList &inputs, const BaseInfoList &outputs, const inner::DAttrs &attrs,
                                const std::string &processor) {
  this->inputs_info_ = inputs;
  this->outputs_info_ = outputs;
  this->attrs_ = attrs;
  this->processor_ = processor;
  if (std::any_of(validators_.begin(), validators_.end(),
                  [this](const std::unique_ptr<Validator> &v) { return !(v->Check(*this)); })) {
    return nullptr;
  }
  Init();
  if (!this->CheckInputs()) {
    return nullptr;
  }
  for (auto &inp : inputs) {
    (void)gb.Parameter(inp);
  }
  auto result = this->Expand(gb.Get()->inputs());
  gb.SetOutputs(result);
  if (!this->CheckOutputs()) {
    return nullptr;
  }
  return gb.Get();
}

bool OpDesc::CheckOutputs() {
  // check the output shape/type/format are same as the original basic node's output.
  const NodePtrList &outputs = gb.Get()->GetOutputs();
  if (outputs.size() != this->outputs_info_.size()) {
    MS_LOG(INFO) << "the output num was not equal to the original output num : " << outputs.size() << " vs "
                 << outputs_info_.size();
    return false;
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i]->shape != outputs_info_[i].shape) {
      std::ostringstream oss;
      oss << "Op " << this->name_ << "'s output shape [";
      for (auto s : outputs[i]->shape) {
        oss << s << ",";
      }
      oss << "] is wrong. expect: [";
      for (auto s : outputs_info_[i].shape) {
        oss << s << ",";
      }
      oss << "]";
      MS_LOG(INFO) << oss.str();
      return false;
    }
    if (outputs[i]->type != outputs_info_[i].type) {
      MS_LOG(INFO) << "Op " << this->name_ << "'s output type [" << outputs[i]->type << "] is wrong, expect: ["
                   << outputs_info_[i].type << "]";
      return false;
    }
    if (outputs[i]->format != outputs_info_[i].format) {
      MS_LOG(INFO) << "Op " << this->name_ << "'s output format [" << outputs[i]->format << "] is wrong, expect: ["
                   << outputs_info_[i].format << "]";
      return false;
    }
  }
  return true;
}

std::vector<int64_t> GetAxisList(const ValuePtr &value) {
  std::vector<int64_t> result;
  auto get_int_value = [](const ValuePtr &value) -> int64_t {
    return value->isa<Int64Imm>() ? GetValue<int64_t>(value) : static_cast<int64_t>(GetValue<int>(value));
  };
  if (value->isa<ValueSequence>()) {
    const auto &vals = value->cast<ValueSequencePtr>()->value();
    (void)std::transform(vals.begin(), vals.end(), std::back_inserter(result), get_int_value);
  } else {
    result.push_back(get_int_value(value));
  }
  return result;
}

std::vector<int64_t> InferShapeFromFractalnz(const std::vector<int64_t> &fractal) {
  std::vector<int64_t> shape;
  size_t dims = fractal.size();
  size_t batch = dims - OFFSET4;
  for (size_t i = 0; i < batch; i++) {
    shape.push_back(fractal[i]);
  }
  shape.push_back(fractal[dims - OFFSET3] * fractal[dims - OFFSET2]);
  shape.push_back(fractal[dims - OFFSET4] * fractal[dims - OFFSET1]);
  return shape;
}

std::vector<int64_t> GetReducedOriShape(const std::vector<int64_t> &shape, const std::vector<int64_t> &axis) {
  std::vector<int64_t> reduced_ori_shape;
  std::unordered_set<int64_t> axis_set(axis.begin(), axis.end());
  for (size_t i = 0; i < shape.size(); i++) {
    if (axis_set.count(SizeToLong(i)) > 0) {
      reduced_ori_shape.push_back(1);
    } else {
      reduced_ori_shape.push_back(shape[i]);
    }
  }
  return reduced_ori_shape;
}

std::vector<int64_t> ToFracZAxis(const std::vector<int64_t> &ori_shape, const std::vector<int64_t> &ori_axis) {
  std::vector<int64_t> frac_z_axis = ori_axis;
  int64_t shape_len = SizeToLong(ori_shape.size());
  for (size_t i = 0; i < frac_z_axis.size(); i++) {
    int64_t axis_index = (frac_z_axis[i] + shape_len) % shape_len;
    if (axis_index == shape_len - OFFSET1) {
      frac_z_axis[i] = axis_index - OFFSET1;
      frac_z_axis.push_back(axis_index + OFFSET2);
    } else if (axis_index == shape_len - OFFSET2) {
      frac_z_axis[i] = axis_index + OFFSET1;
      frac_z_axis.push_back(axis_index + OFFSET2);
    } else {
      frac_z_axis[i] = axis_index;
    }
  }
  return frac_z_axis;
}
}  // namespace mindspore::graphkernel::expanders
