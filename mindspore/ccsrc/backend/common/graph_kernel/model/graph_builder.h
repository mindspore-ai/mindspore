/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_GRAPH_BUILDER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_GRAPH_BUILDER_H_

#include <vector>
#include <memory>
#include <string>
#include "backend/common/graph_kernel/model/lite_graph.h"

namespace mindspore::graphkernel::inner {
class GraphBuilder : public LiteGraph::GraphBuilderBase {
 public:
  explicit GraphBuilder(const std::string &name = "") : GraphBuilderBase(name) {}
  ~GraphBuilder() = default;
  NodePtr Add(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("Add", {lhs, rhs}); }
  NodePtr Sub(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("Sub", {lhs, rhs}); }
  NodePtr Mul(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("Mul", {lhs, rhs}); }
  NodePtr Div(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("RealDiv", {lhs, rhs}); }
  NodePtr Greater(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("Greater", {lhs, rhs}); }
  NodePtr Less(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("Less", {lhs, rhs}); }
  NodePtr GreaterEqual(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("GreaterEqual", {lhs, rhs}); }
  NodePtr LessEqual(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("LessEqual", {lhs, rhs}); }
  NodePtr Equal(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("Equal", {lhs, rhs}); }
  NodePtr LogicalOr(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("LogicalOr", {lhs, rhs}); }
  NodePtr Assign(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("Assign", {lhs, rhs}); }
  NodePtr Select(const NodePtr &cond, const NodePtr &lhs, const NodePtr &rhs) const {
    return Emit("Select", {cond, lhs, rhs});
  }
  NodePtr MatMul(const NodePtr &lhs, const NodePtr &rhs, const TypeId &type_id = kNumberTypeFloat16,
                 const bool &transpose_a = false, const bool &transpose_b = false) const {
    return Emit("MatMul", {lhs, rhs},
                {{"transpose_a", MakeValue(transpose_a)},
                 {"transpose_x1", MakeValue(transpose_a)},
                 {"transpose_b", MakeValue(transpose_b)},
                 {"transpose_x2", MakeValue(transpose_b)},
                 {"dst_type", TypeIdToType(type_id)}});
  }
  NodePtr Neg(const NodePtr &input) const { return Emit("Neg", {input}); }
  NodePtr Exp(const NodePtr &input) const { return Emit("Exp", {input}); }
  NodePtr Abs(const NodePtr &input) const { return Emit("Abs", {input}); }
  NodePtr Log(const NodePtr &input) const { return Emit("Log", {input}); }
  NodePtr Sqrt(const NodePtr &input) const { return Emit("Sqrt", {input}); }
  NodePtr Tanh(const NodePtr &input) const { return Emit("Tanh", {input}); }
  NodePtr IsInf(const NodePtr &input) const { return Emit("IsInf", {input}); }
  NodePtr IsNan(const NodePtr &input) const { return Emit("IsNan", {input}); }
  NodePtr StridedSlice(const NodePtr &input, const std::vector<int64_t> &begin, const std::vector<int64_t> &end,
                       const std::vector<int64_t> &strides) const {
    return Emit("StridedSlice", {input},
                {{"begin", MakeValue(begin)},
                 {"end", MakeValue(end)},
                 {"strides", MakeValue(strides)},
                 {"shrink_axis_mask", MakeValue(static_cast<int64_t>(0))},
                 {"begin_mask", MakeValue(static_cast<int64_t>(0))},
                 {"ellipsis_mask", MakeValue(static_cast<int64_t>(0))},
                 {"new_axis_mask", MakeValue(static_cast<int64_t>(0))},
                 {"end_mask", MakeValue(static_cast<int64_t>(0))}});
  }
  NodePtr TensorScatterAdd(const NodePtr &input, const NodePtr &indices, const NodePtr &update) const {
    return Emit("TensorScatterAdd", {input, indices, update});
  }
  NodePtr Custom(const NodePtrList &inputs, const NodeBase &baseinfo, const std::string &func_name,
                 const std::string &func_type, const std::string &func_source_str, const size_t &inplace_assign_output,
                 const std::string &func_compile_attrs) {
    std::string write_from_output_to_input = "0 " + std::to_string(inplace_assign_output);
    return Op("Custom", baseinfo, inputs,
              {{"func_name", MakeValue(func_name)},
               {"func_type", MakeValue(func_type)},
               {"func_source_str", MakeValue(func_source_str)},
               {"inplace_assign_output", MakeValue(write_from_output_to_input)},
               {"func_compile_attrs", MakeValue(func_compile_attrs)}});
  }
  NodePtr Cast(const NodePtr &input, const TypeId &type_id) const {
    return Emit("Cast", {input}, {{"dst_type", TypeIdToType(type_id)}});
  }
  NodePtr Shape(const NodePtr &input) const { return Emit("Shape", {input}); }
  NodePtr Reshape(const NodePtr &input, const ShapeVector &shape) const;
  NodePtr BroadcastTo(const NodePtr &input, const ShapeVector &shape) const;
  NodePtr Gather(const NodePtr &param, const NodePtr &indice, const int64_t &axis) const;
  NodePtr Concat(const NodePtrList &inputs, const int64_t &axis) const;
  NodePtr Transpose(const NodePtr &input, const ShapeVector &perm) const;

  NodePtr ReduceSum(const NodePtr &input, const std::vector<int64_t> &axis, const bool &keep_dims = false) const;
  NodePtr ReduceMax(const NodePtr &input, const std::vector<int64_t> &axis, const bool &keep_dims = false) const;
  NodePtr ReduceMin(const NodePtr &input, const std::vector<int64_t> &axis, const bool &keep_dims = false) const;

  template <typename T>
  NodePtr Const(T input, const TypeId &type_id) const {
    tensor::TensorPtr const_input;
    switch (type_id) {
      case kNumberTypeBool:
        const_input = std::make_shared<tensor::Tensor>(static_cast<bool>(input), TypeIdToType(type_id));
        break;
      case kNumberTypeInt:
      case kNumberTypeInt8:
      case kNumberTypeInt16:
      case kNumberTypeInt32:
      case kNumberTypeInt64:
        const_input = std::make_shared<tensor::Tensor>(static_cast<int64_t>(input), TypeIdToType(type_id));
        break;
      case kNumberTypeUInt:
      case kNumberTypeUInt8:
      case kNumberTypeUInt16:
      case kNumberTypeUInt32:
      case kNumberTypeUInt64:
        const_input = std::make_shared<tensor::Tensor>(static_cast<uint64_t>(input), TypeIdToType(type_id));
        break;
      case kNumberTypeFloat:
      case kNumberTypeFloat16:
      case kNumberTypeFloat32:
      case kNumberTypeFloat64:
        const_input = std::make_shared<tensor::Tensor>(static_cast<double>(input), TypeIdToType(type_id));
        break;
      default:
        MS_LOG(EXCEPTION) << "The input data type should be int, uint, float or bool, But Get :"
                          << TypeIdToString(type_id);
    }
    return Value(const_input);
  }

  NodePtr TupleGetItem(const NodePtr &input, int64_t index) const;
};
}  // namespace mindspore::graphkernel::inner
#endif
