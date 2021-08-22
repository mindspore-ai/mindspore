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

#include <math.h>
#include <sstream>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <numeric>

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

template <typename TM, typename TD>
tensor::TensorPtr CalcByOperator(const NodePtrList &inputs, const std::string &op, TypeId tid) {
  std::vector<TM> inputs_tm;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_tm), [](const NodePtr &i) {
    return *static_cast<TM *>(std::static_pointer_cast<graphkernel::ConstTensorNode>(i)->data()->data_c());
  });

  std::unordered_map<std::string, std::function<TM(const std::vector<TM> &)>> func_map;
  func_map["Add"] = [](const std::vector<TM> &n) { return n[0] + n[1]; };
  func_map["Sub"] = [](const std::vector<TM> &n) { return n[0] - n[1]; };
  func_map["Mul"] = [](const std::vector<TM> &n) { return n[0] * n[1]; };
  func_map["RealDiv"] = [](const std::vector<TM> &n) { return n[0] / n[1]; };
  func_map["Neg"] = [](const std::vector<TM> &n) { return -n[0]; };
  func_map["Reciprocal"] = [](const std::vector<TM> &n) { return TM(1) / n[0]; };
  func_map["Log"] = [](const std::vector<TM> &n) { return log(n[0]); };
  func_map["Exp"] = [](const std::vector<TM> &n) { return exp(n[0]); };
  func_map["Abs"] = [](const std::vector<TM> &n) { return n[0] < TM(0) ? (-n[0]) : n[0]; };
  func_map["Sqrt"] = [](const std::vector<TM> &n) { return sqrt(n[0]); };
  func_map["Rsqrt"] = [](const std::vector<TM> &n) { return TM(1) / sqrt(n[0]); };

  if (func_map.find(op) == func_map.end()) return nullptr;
  return std::make_shared<tensor::Tensor>(static_cast<TD>(func_map[op](inputs_tm)), TypeIdToType(tid));
}

NodePtr PrimOp::InferValue(const NodePtrList &inputs, const DAttrs &attrs, const std::string &op) {
  for (auto i : inputs) {
    if (i->NodeType() != NType::Value) return nullptr;
  }
  TypeId output_type = InferType(inputs, attrs);
  tensor::TensorPtr res = nullptr;
  switch (output_type) {
    case TypeId::kNumberTypeUInt8: {
      res = CalcByOperator<uint8_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeInt8: {
      res = CalcByOperator<int8_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeInt16: {
      res = CalcByOperator<int16_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeInt32: {
      res = CalcByOperator<int32_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeInt64: {
      res = CalcByOperator<int64_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeUInt16: {
      res = CalcByOperator<uint16_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeUInt32: {
      res = CalcByOperator<uint32_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeUInt64: {
      res = CalcByOperator<uint64_t, int64_t>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeFloat16: {
      res = CalcByOperator<float16, double>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeFloat32: {
      res = CalcByOperator<float, double>(inputs, op, output_type);
      break;
    }
    case TypeId::kNumberTypeFloat64: {
      res = CalcByOperator<double, double>(inputs, op, output_type);
      break;
    }
    default:
      return nullptr;
  }
  return res == nullptr ? nullptr : std::make_shared<ConstTensorNode>(res);
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

DShape BroadcastToOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  return GetValue<std::vector<int64_t>>(attrs.find("shape")->second);
}

DShape ReshapeOp::InferShape(const NodePtrList &inputs, const DAttrs &attrs) {
  auto new_shape = GetValue<std::vector<int64_t>>(attrs.find("shape")->second);
  auto origin_shape = inputs[0]->shape;
  for (size_t i = 0; i < new_shape.size(); i++) {
    if (new_shape[i] == -1) {
      auto origin_product = std::accumulate(origin_shape.begin(), origin_shape.end(), 1, std::multiplies<int64_t>());
      auto new_product = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int64_t>());
      new_shape[i] = origin_product / new_product * (-1);
      break;
    }
  }
  return new_shape;
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
