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
#include "backend/optimizer/graph_kernel/model/op_node.h"

#include <sstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "backend/optimizer/graph_kernel/model/node.h"

namespace mindspore {
namespace opt {
namespace graphkernel {
void PrimOp::Infer(const NodePtrList &inputs, const DAttrs &attrs) {
  this->shape = InferShape(inputs, attrs);
  this->type = InferType(inputs, attrs);
  this->format = InferFormat(inputs, attrs);
  this->attrs_ = attrs;
  SetInputs(inputs);
}

void PrimOp::Dump(std::ostringstream &os) const {
  DumpTensor(os);
  os << " = " << this->op_ << "(";
  for (size_t i = 0; i < inputs_.size(); i++) {
    inputs_[i]->DumpTensor(os);
    if (i != inputs_.size() - 1) os << ", ";
  }
  os << ")";
  std::ostringstream attr_os;
  bool has_attr = false;
  std::set<std::string> black_list = {"IsFeatureMapInputList", "IsFeatureMapOutput", "output_names", "input_names"};
  for (auto attr : attrs_) {
    if (attr.second != nullptr && black_list.count(attr.first) == 0) {
      if (has_attr) {
        attr_os << ", ";
      } else {
        has_attr = true;
      }
      attr_os << attr.first << ": " << attr.second->ToString();
    }
  }
  if (has_attr) {
    os << "  // attr {" << attr_os.str() << "}";
  }
}

void ElemwiseOp::Infer(const NodePtrList &inputs, const DAttrs &attrs) {
  PrimOp::Infer(inputs, attrs);
  auto IsBroadcast = [this](const NodePtrList &inputs) -> bool {
    for (auto &ref : inputs) {
      if (ref->shape.size() != this->shape.size()) return true;
      for (size_t i = 0; i < this->shape.size(); ++i) {
        if (ref->shape[i] != this->shape[i]) return true;
      }
    }
    return false;
  };
  compute_type_ = IsBroadcast(inputs) ? BROADCAST : ELEMWISE;
}

DShape ReduceOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  auto axis = GetValue<std::vector<int64_t>>(attrs.find("axis")->second);
  auto keepdims = GetValue<bool>(attrs.find("keep_dims")->second);
  if (keepdims) {
    DShape new_shape = inputs[0]->shape;
    for (auto x : axis) {
      new_shape[x] = 1;
    }
    return new_shape;
  }
  DShape new_shape;
  const auto &input_shape = inputs[0]->shape;
  for (size_t i = 0; i < input_shape.size(); i++) {
    if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
      new_shape.emplace_back(input_shape[i]);
    }
  }
  if (new_shape.empty()) {
    new_shape.emplace_back(1);
  }
  return new_shape;
}
}  // namespace graphkernel
}  // namespace opt
}  // namespace mindspore
