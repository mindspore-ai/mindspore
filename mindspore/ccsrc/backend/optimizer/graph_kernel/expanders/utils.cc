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
#include "backend/optimizer/graph_kernel/expanders/utils.h"

#include <algorithm>
#include <string>
#include <vector>

#include "backend/optimizer/graph_kernel/model/lite_graph.h"
#include "backend/optimizer/graph_kernel/model/node.h"

namespace mindspore {
namespace opt {
namespace expanders {
graphkernel::LiteGraphPtr OpExpander::Run(const BaseInfoList &inputs, const BaseInfoList &outputs,
                                          const graphkernel::DAttrs &attrs, const std::string &processor) {
  this->inputs_info_ = inputs;
  this->outputs_info_ = outputs;
  this->attrs_ = attrs;
  this->processor_ = processor;
  if (std::any_of(validators_.begin(), validators_.end(),
                  [this](const std::unique_ptr<Validator> &v) { return !(v->Check(*this)); })) {
    return nullptr;
  }
  if (!this->CheckInputs()) {
    return nullptr;
  }
  for (auto &inp : inputs) {
    (void)gb.Parameter(inp);
  }
  auto result = this->Expand();
  gb.SetOutputs(result);
  if (!this->CheckOutputs()) {
    return nullptr;
  }
  return gb.Get();
}

bool OpExpander::CheckOutputs() {
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
      oss << "Op " << this->op_ << "'s output shape [";
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
      MS_LOG(INFO) << "Op " << this->op_ << "'s output type [" << outputs[i]->type << "] is wrong, expect: ["
                   << outputs_info_[i].type << "]";
      return false;
    }
    if (outputs[i]->format != outputs_info_[i].format) {
      MS_LOG(INFO) << "Op " << this->op_ << "'s output format [" << outputs[i]->format << "] is wrong, expect: ["
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
  if (value->isa<ValueSequeue>()) {
    const auto &vals = value->cast<ValueSequeuePtr>()->value();
    (void)std::transform(vals.begin(), vals.end(), std::back_inserter(result), get_int_value);
  } else {
    result.push_back(get_int_value(value));
  }
  return result;
}
}  // namespace expanders
}  // namespace opt
}  // namespace mindspore
