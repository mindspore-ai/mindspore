/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/dvm/kernel_mod_impl/dvm_kernel_build.h"
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include "kernel/ascend/dvm/kernel_mod_impl/dvm_kernel_mod.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "ir/graph_utils.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "primitive/math_ops.h"
#include "primitive/other_ops.h"  // collective communication operations
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "utils/check_convert_utils.h"
#include "plugin/ascend/res_manager/collective/dvm_collective_comm_lib.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"

namespace mindspore {
namespace kernel {
namespace {
enum OpType {
  OP_UNARY,
  OP_BINARY,
  OP_RESHAPE,
  OP_BROADCAST,
  OP_CAST,
  OP_NEG,
  OP_SELECT,
  OP_RSQRT,
  OP_ASSIGN,
  OP_ELEMENY,
  OP_REDUCE,
  OP_SLICE,
  OP_MATMUL,
  OP_GMM,
  OP_COMM,
  OP_TG,
  OP_COPY,
  OPTypeEnd
};

ShapeVector GetAxisList(const AnfNodePtr &axis_input) {
  ValuePtr value = nullptr;
  if (axis_input->isa<ValueNode>()) {
    value = axis_input->cast<ValueNodePtr>()->value();
  } else if (axis_input->isa<Parameter>()) {
    value = axis_input->cast<ParameterPtr>()->abstract()->BuildValue();
  }
  if (value == nullptr) {
    MS_LOG(EXCEPTION) << "ReduceOp axis input is not Value.";
  }
  if (value->isa<ValueAny>()) {
    MS_LOG(EXCEPTION) << "ReduceOp axis input is ValueAny.";
  }
  ShapeVector result;
  if (value->isa<ValueSequence>()) {
    result = GetValue<ShapeVector>(value);
  } else if (value->isa<tensor::Tensor>()) {
    result = TensorValueToVector<int64_t>(value->cast<tensor::TensorPtr>());
  } else {
    result.push_back(GetValue<int64_t>(value));
  }
  return result;
}

static std::unordered_map<std::string, std::pair<OpType, int>> op_type_map = {
  {"Abs", {OP_UNARY, dvm::UnaryOpType::kAbs}},
  {"Exp", {OP_UNARY, dvm::UnaryOpType::kExp}},
  {"Log", {OP_UNARY, dvm::UnaryOpType::kLog}},
  {"Sqrt", {OP_UNARY, dvm::UnaryOpType::kSqrt}},
  {"Round", {OP_UNARY, dvm::UnaryOpType::kRound}},
  {"Floor", {OP_UNARY, dvm::UnaryOpType::kFloor}},
  {"Ceil", {OP_UNARY, dvm::UnaryOpType::kCeil}},
  {"Trunc", {OP_UNARY, dvm::UnaryOpType::kTrunc}},
  {"Neg", {OP_NEG, 0}},
  {"Cast", {OP_CAST, 0}},
  {"Add", {OP_BINARY, dvm::BinaryOpType::kAdd}},
  {"Sub", {OP_BINARY, dvm::BinaryOpType::kSub}},
  {"Mul", {OP_BINARY, dvm::BinaryOpType::kMul}},
  {"Div", {OP_BINARY, dvm::BinaryOpType::kDiv}},
  {"RealDiv", {OP_BINARY, dvm::BinaryOpType::kDiv}},
  {"Greater", {OP_BINARY, dvm::BinaryOpType::kGreater}},
  {"Maximum", {OP_BINARY, dvm::BinaryOpType::kMaximum}},
  {"Minimum", {OP_BINARY, dvm::BinaryOpType::kMinimum}},
  {"Pow", {OP_BINARY, dvm::BinaryOpType::kPow}},
  {"BroadcastTo", {OP_BROADCAST, 0}},
  {"GreaterEqual", {OP_BINARY, dvm::BinaryOpType::kGreaterEqual}},
  {"Less", {OP_BINARY, dvm::BinaryOpType::kLess}},
  {"LessEqual", {OP_BINARY, dvm::BinaryOpType::kLessEqual}},
  {"Equal", {OP_BINARY, dvm::BinaryOpType::kEqual}},
  {"NotEqual", {OP_BINARY, dvm::BinaryOpType::kNotEqual}},
  {"Reciprocal", {OP_UNARY, dvm::UnaryOpType::kReciprocal}},
  {"Reshape", {OP_RESHAPE, 0}},
  {"Select", {OP_SELECT, 0}},
  {"LogicalNot", {OP_UNARY, dvm::UnaryOpType::kLogicalNot}},
  {"LogicalOr", {OP_BINARY, dvm::BinaryOpType::kLogicalOr}},
  {"LogicalAnd", {OP_BINARY, dvm::BinaryOpType::kLogicalAnd}},
  {"Rsqrt", {OP_RSQRT, 0}},
  {"Assign", {OP_ASSIGN, 0}},
  {"ElemAny", {OP_ELEMENY, 0}},
  {"IsFinite", {OP_UNARY, dvm::UnaryOpType::kIsFinite}},
  {"ReduceSum", {OP_REDUCE, dvm::ReduceOpType::kSum}},
  {"Slice", {OP_SLICE, 0}},
  {"StridedSlice", {OP_SLICE, 1}},
  {ops::kNameMatMul, {OP_MATMUL, 0}},
  {ops::kNameBatchMatMul, {OP_MATMUL, 0}},
  {ops::kNameGroupedMatmul, {OP_GMM, 0}},
  {"TupleGetItem", {OP_TG, 0}},
  {"AllReduce", {OP_COMM, 0}},
};

TypeId GetValueNodeType(const AnfNodePtr &node) {
  auto valuenode = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(valuenode);
  auto value = valuenode->value();
  MS_EXCEPTION_IF_NULL(value);
  TypeId type_id = TypeId::kTypeUnknown;
  if (value->isa<tensor::Tensor>()) {
    auto input_tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(input_tensor);
    type_id = input_tensor->data_type();
  } else if (value->isa<Scalar>()) {
    auto value_type = value->type();
    MS_EXCEPTION_IF_NULL(value_type);
    type_id = value_type->type_id();
  }
  if (type_id != TypeId::kNumberTypeFloat32 && type_id != TypeId::kNumberTypeFloat16 &&
      type_id != TypeId::kNumberTypeInt32 && type_id != TypeId::kNumberTypeBFloat16) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Data type of scalar value input only supports float, but got: "
                                      << TypeIdToString(type_id) << " node: " << node->fullname_with_scope();
  }
  return type_id;
}

template <typename T>
T GetScalarFromNode(const AnfNodePtr &node) {
  auto valuenode = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(valuenode);
  auto value_ptr = valuenode->value();
  if (value_ptr->isa<Scalar>()) {
    auto input_scalar = value_ptr->cast<ScalarPtr>();
    MS_EXCEPTION_IF_NULL(input_scalar);
    if constexpr (std::is_same_v<T, dvm::Float16>) {
      return T(GetValue<float16>(input_scalar).int_value());
    } else if constexpr (std::is_same_v<T, dvm::BFloat16>) {
      return T(GetValue<bfloat16>(input_scalar).int_value());
    } else {
      return GetValue<T>(input_scalar);
    }
  } else {
    auto input_tensor = value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(input_tensor);
    if constexpr (std::is_same_v<T, dvm::Float16>) {
      return T(static_cast<float16 *>(input_tensor->data_c())->int_value());
    } else if constexpr (std::is_same_v<T, dvm::BFloat16>) {
      return T(static_cast<bfloat16 *>(input_tensor->data_c())->int_value());
    } else {
      return *(static_cast<T *>(input_tensor->data_c()));
    }
  }
}

class OpBuilder {
 public:
  OpBuilder(dvm::Kernel *kernel, const AnfNodePtrList &outputs, std::unordered_map<AnfNodePtr, ShapeRefPtr> *shapes_ref,
            std::vector<ShapeVector> *shapes_ref_source, bool empty_input)
      : kernel_(kernel), shapes_ref_(shapes_ref), shapes_ref_source_(shapes_ref_source), empty_input_(empty_input) {
    for (const auto &node : outputs) {
      outputs_[node] = nullptr;
    }
  }
  ~OpBuilder() = default;

  void Emit(const AnfNodePtr &anf_node) {
    auto node = anf_node->cast<CNodePtr>();
    auto prim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    auto op_type = op_type_map.find(prim_name);
    if (op_type == op_type_map.end()) {
      MS_LOG(EXCEPTION) << "unsupported fused op: " << prim_name;
    }
    switch (op_type->second.first) {
      case OP_CAST: {
        auto dst_dtype = AnfAlgo::GetOutputDeviceDataType(node, 0);
        auto op = EmitCast(GetInput(node->input(kIndex1)), dst_dtype);
        EmitOp(anf_node, op);
        break;
      }
      case OP_SELECT: {
        auto op = kernel_->Select(GetInput(node->input(kIndex1)), GetInput(node->input(kIndex2)),
                                  GetInput(node->input(kIndex3)));
        EmitOp(anf_node, op);
        break;
      }
      case OP_UNARY: {
        auto op = kernel_->Unary(op_type->second.second, GetInput(node->input(kIndex1)));
        EmitOp(anf_node, op);
        break;
      }
      case OP_RSQRT: {
        auto sqrt_op = kernel_->Unary(dvm::UnaryOpType::kSqrt, GetInput(node->input(kIndex1)));
        auto op = kernel_->Unary(dvm::UnaryOpType::kReciprocal, sqrt_op);
        EmitOp(anf_node, op);
        break;
      }
      case OP_RESHAPE: {
        auto shape_ref = CacheShape(node, kIndex2);
        auto op = kernel_->Reshape(GetInput(node->input(kIndex1)), shape_ref);
        EmitOp(anf_node, op);
        break;
      }
      case OP_BINARY: {
        auto op = EmitBinaryOp(node, op_type->second.second);
        EmitOp(anf_node, op);
        break;
      }
      case OP_BROADCAST: {
        HandlerBroadcastOp(node);
        break;
      }
      case OP_NEG: {
        auto obj = GetInput(node->input(1));
        auto op = kernel_->Binary(dvm::BinaryOpType::kMul, obj, -1);
        EmitOp(anf_node, op);
        break;
      }
      case OP_ASSIGN: {
        HandlerAssignOp(node);
        break;
      }
      case OP_ELEMENY: {
        auto dst_dtype = AnfAlgo::GetOutputDeviceDataType(node, 0);
        auto op = kernel_->ElemAny(EmitCast(GetInput(node->input(kIndex1)), dst_dtype));
        EmitOp(anf_node, op);
        break;
      }
      case OP_REDUCE: {
        HandlerReduceOp(node, prim, op_type->second.second);
        break;
      }
      case OP_SLICE: {
        HandlerSliceOp(node, op_type->second.second);
        break;
      }
      case OP_MATMUL: {
        HandlerMatMulOp(node, prim, prim_name);
        break;
      }
      case OP_GMM: {
        HandlerGroupedMatmulOp(node, prim);
        break;
      }
      case OP_COMM: {
        HandlerCollectiveCommOp(node);
        break;
      }
      case OP_TG: {
        EmitOp(node, GetInput(node->input(kIndex1)));
        break;
      }
      case OP_COPY: {
        EmitOp(anf_node, kernel_->Copy(GetInput(node->input(kIndex1))));
        break;
      }
      default:
        MS_LOG(EXCEPTION) << op_type->second << " is unsupported op type.";
        break;
    }
  }

  template <typename T>
  dvm::NDObject *EmitBinaryScalarOp(const AnfNodePtr &node, const AnfNodePtr &scalar_node, int binary_type,
                                    const bool rhs_val) {
    auto scalar = GetScalarFromNode<T>(scalar_node);
    return rhs_val ? kernel_->Binary(binary_type, GetInput(node), scalar)
                   : kernel_->Binary(binary_type, scalar, GetInput(node));
  }

  dvm::NDObject *EmitBinaryOp(const CNodePtr &node, int binary_type) {
    AnfNodePtr inputs[] = {node->input(kIndex1), node->input(kIndex2)};
    int scalar_index = -1;
    for (int i = 0; i < 2; i++) {
      auto shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(inputs[i]->Shape())[kShape];
      auto size = std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies{});
      if (inputs[i]->isa<ValueNode>()) {
        scalar_index = i;
        if (size != 1) {
          MS_LOG_WITH_NODE(EXCEPTION, node)
            << "In GraphKernel the input node " << inputs[i]->fullname_with_scope() << " of "
            << node->fullname_with_scope() << " should have a size of 1, but get " << size;
        }
      }
    }
    dvm::NDObject *op = nullptr;
    if (scalar_index != -1) {
      auto scalar_node = inputs[scalar_index];
      auto common_node = inputs[scalar_index ^ 1];
      auto type_id = GetValueNodeType(scalar_node);
      switch (type_id) {
        case kNumberTypeFloat32:
          return EmitBinaryScalarOp<float>(common_node, scalar_node, binary_type, scalar_index == kIndex1);
        case kNumberTypeFloat16:
          return EmitBinaryScalarOp<dvm::Float16>(common_node, scalar_node, binary_type, scalar_index == kIndex1);
        case kNumberTypeInt32:
          return EmitBinaryScalarOp<int32_t>(common_node, scalar_node, binary_type, scalar_index == kIndex1);
        case kNumberTypeBFloat16:
          return EmitBinaryScalarOp<dvm::BFloat16>(common_node, scalar_node, binary_type, scalar_index == kIndex1);
        default:
          MS_LOG(EXCEPTION) << "Some input of node " << node->fullname_with_scope() << " has unsupported dtype "
                            << TypeIdToString(type_id);
      }
    } else {
      op = kernel_->Binary(binary_type, GetInput(inputs[0]), GetInput(inputs[1]));
    }
    return op;
  }

  dvm::NDObject *GetLoad(const AnfNodePtr &node) {
    auto it = inputs_.find(node);
    return (it == inputs_.end() ? nullptr : it->second);
  }

  dvm::NDObject *GetStore(const AnfNodePtr &node) {
    auto it = outputs_.find(node);
    return (it == outputs_.end() ? nullptr : it->second);
  }

 private:
  void HandlerAssignOp(const AnfNodePtr &anf_node) {
    const CNodePtr &node = anf_node->cast<CNodePtr>();
    auto out_type = AnfAlgo::GetOutputDeviceDataType(node, kIndex0);
    auto input_obj = GetInput(node->input(kIndex2));
    // %0 = Assign(p0, p1)
    // %1 = Div(p2, %0)
    // %2 = Add(%1, p0)
    // The second input of Div should be Load(p1), not Store(%0), precision error due to wrong local memory address
    // The second input of Add should be Load(p1), not Load(p0)
    ops_map_[anf_node] = input_obj;
    ops_map_[node->input(kIndex1)] = input_obj;
    auto input2 = EmitCast(input_obj, out_type);
    // store the second input of assign to the output of subgraph
    // the output addr of subgraph equals to the corresponding parameter addr of subgraph
    if (outputs_.find(node) != outputs_.end()) {
      outputs_[anf_node] = kernel_->Store(nullptr, input2);
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "AssignOp " << node->fullname_with_scope()
                                        << " is not in graph kernel 's outputs.";
    }
  }

  void HandlerReduceOp(const CNodePtr &node, const PrimitivePtr &prim, int op_type) {
    auto keep_dims_attr = prim->GetAttr(kAttrKeepDims);
    MS_EXCEPTION_IF_NULL(keep_dims_attr);
    auto keep_dims = GetValue<bool>(keep_dims_attr);
    auto shape_ref = CacheAxis(node, node->input(2));
    auto op = kernel_->Reduce(op_type, GetInput(node->input(1)), shape_ref, keep_dims);
    EmitOp(node, op);
  }

  void HandlerBroadcastOp(const CNodePtr &node) {
    auto input = node->input(kIndex1);
    auto shape_ref = CacheShape(node, kIndex2);
    auto op =
      input->isa<ValueNode>() ? EmitScalarBroadcast(input, shape_ref) : kernel_->Broadcast(GetInput(input), shape_ref);
    EmitOp(node, op);
  }

  void HandlerSliceOp(const CNodePtr &node, int op_type) {
    auto input = node->input(1);
    auto start_ref = CacheAxis(node, node->input(kIndex2));
    if (op_type) {
      auto end_ref = CacheAxis(node, node->input(kIndex3));
      auto step_ref = CacheAxis(node, node->input(kIndex4));
      auto op = kernel_->Copy(GetInput(input, start_ref, end_ref, step_ref));
      EmitOp(node, op);
    } else {
      auto size_ref = CacheAxis(node, node->input(kIndex3));
      auto op = kernel_->Copy(GetInput(input, start_ref, size_ref));
      EmitOp(node, op);
    }
  }

  void HandlerCollectiveCommOp(const CNodePtr &node) {
    auto prim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(prim);
    auto group_name = GetValue<std::string>(prim->GetAttr(kAttrGroup));
    auto comm_ptr = device::ascend::DvmCollectiveCommLib::GetInstance().GetCommunicator(group_name);
    auto op = kernel_->AllReduce(GetInput(node->input(kIndex1)), comm_ptr.get());
    EmitOp(node, op);
  }

  void HandlerMatMulOp(const CNodePtr &node, const PrimitivePtr &prim, const std::string &prim_name) {
    // Input: (prim, a, b)
    constexpr auto kMatMulInputNum = 3;
    constexpr auto kMatMulBiasAddInputNum = 4;
    if (node->size() == kMatMulInputNum || node->size() == kMatMulBiasAddInputNum) {
      auto transpose_a = GetValue<bool>(prim->GetAttr(kTransposeA));
      auto transpose_b = GetValue<bool>(prim->GetAttr(kTransposeB));
      auto op =
        kernel_->MatMul(GetInput(node->input(kIndex1)), GetInput(node->input(kIndex2)), transpose_a, transpose_b,
                        node->size() == kMatMulBiasAddInputNum ? GetInput(node->input(kIndex3)) : nullptr);
      EmitOp(node, op);
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "Input size of " << prim_name << " should be " << kMatMulInputNum
                                        << " but got " << node->size();
    }
  }

  void HandlerGroupedMatmulOp(const CNodePtr &node, const PrimitivePtr &prim) {
    auto transpose_a = GetValue<bool>(prim->GetAttr(kTransposeA));
    auto transpose_b = GetValue<bool>(prim->GetAttr(kTransposeB));
    auto group_type = GetValue<int64_t>(prim->GetAttr("group_type"));
    auto bias_node = node->input(kIndex3);
    dvm::NDObject *bias_op = nullptr;
    if (AnfAlgo::GetInputDeviceShape(node, kIndex2) != ShapeVector{0}) {
      bias_op = GetInput(bias_node);
    }
    auto op = kernel_->GroupedMatMul(GetInput(node->input(kIndex1)), GetInput(node->input(kIndex2)), transpose_a,
                                     transpose_b, bias_op, GetInput(node->input(kIndex8)), dvm::GroupType(group_type));
    EmitOp(node, op);
  }

  std::pair<dvm::ShapeRef *, dvm::DType> GetNodeShapeAndType(const AnfNodePtr &node) {
    // hit subgraph input
    auto type_id = AnfAlgo::GetOutputDeviceDataType(node, 0);
    auto iter = ms_type_map.find(type_id);
    if (iter == ms_type_map.end()) {
      MS_LOG_WITH_NODE(EXCEPTION, node) << node->ToString() << " 's type " << TypeIdToString(type_id)
                                        << " is unsupported data type.";
    }
    auto shape = AnfAlgo::GetOutputDeviceShape(node, 0);
    shapes_ref_source_->push_back(shape);
    (*shapes_ref_)[node] = std::make_shared<dvm::ShapeRef>(shapes_ref_source_->back());
    return {(*shapes_ref_)[node].get(), iter->second};
  }

  dvm::NDObject *GetInput(const AnfNodePtr &node) {
    auto it = ops_map_.find(node);
    if (it == ops_map_.end()) {
      dvm::NDObject *op = nullptr;
      if (node->isa<ValueNode>()) {
        shapes_ref_source_->push_back({1});
        (*shapes_ref_)[node] = std::make_shared<dvm::ShapeRef>(shapes_ref_source_->back());
        op = EmitScalarBroadcast(node, (*shapes_ref_)[node].get());
      } else if (node->isa<Parameter>()) {
        auto [shape_ref, type] = GetNodeShapeAndType(node);
        op = kernel_->Load(nullptr, shape_ref, type);
        inputs_[node] = op;
      } else {
        MS_LOG_WITH_NODE(EXCEPTION, node) << node->DebugString() << " is unsupported node type.";
      }
      ops_map_[node] = op;
      return op;
    }
    return it->second;
  }

  dvm::NDObject *GetInput(const AnfNodePtr &node, dvm::ShapeRef *start, dvm::ShapeRef *size) {
    auto it = ops_map_.find(node);
    if (it == ops_map_.end()) {
      dvm::NDObject *op = nullptr;
      if (node->isa<Parameter>()) {
        auto [shape_ref, type] = GetNodeShapeAndType(node);
        op = kernel_->SliceLoad(nullptr, shape_ref, start, size, type);
        inputs_[node] = op;
      } else {
        MS_LOG_WITH_NODE(EXCEPTION, node) << node->DebugString() << " is unsupported node type.";
      }
      ops_map_[node] = op;
      return op;
    }
    return it->second;
  }

  dvm::NDObject *GetInput(const AnfNodePtr &node, dvm::ShapeRef *start, dvm::ShapeRef *end, dvm::ShapeRef *step) {
    auto it = ops_map_.find(node);
    if (it == ops_map_.end()) {
      dvm::NDObject *op = nullptr;
      if (node->isa<Parameter>()) {
        auto [shape_ref, type] = GetNodeShapeAndType(node);
        op = kernel_->StridedSliceLoad(nullptr, shape_ref, start, end, step, type);
        inputs_[node] = op;
      } else {
        MS_LOG_WITH_NODE(EXCEPTION, node) << node->DebugString() << " is unsupported node type.";
      }
      ops_map_[node] = op;
      return op;
    }
    return it->second;
  }

  dvm::ShapeRef *CacheShape(const CNodePtr &node, size_t input_idx) {
    auto shape = AnfAlgo::GetOutputDeviceShape(node, 0);
    if (IsDynamic(shape)) {
      // Although param is subgraph input, there is no need to emit a Load op
      // for it, because it's value is only needed in infer shape
      auto param = node->input(input_idx);
      MS_EXCEPTION_IF_NULL(param);
      if (param->isa<Parameter>()) {
        (*shapes_ref_)[param] = std::make_shared<dvm::ShapeRef>();
        return (*shapes_ref_)[param].get();
      }
      // else, the shape input is a const input, but has -1 value and the real value of -1 can be
      //   inferred from the first input's shape at runtime.
      // e.g. Reshape(x, (1, -1)) or BroadcastTo(x, (-1, 3))
    }
    shapes_ref_source_->push_back(shape);
    (*shapes_ref_)[node] = std::make_shared<dvm::ShapeRef>(shapes_ref_source_->back());
    return (*shapes_ref_)[node].get();
  }

  dvm::ShapeRef *CacheAxis(const CNodePtr &node, const AnfNodePtr &axis_input) {
    if (IsDynamic(AnfAlgo::GetOutputDeviceShape(node, 0)) && axis_input->isa<Parameter>()) {
      (*shapes_ref_)[axis_input] = std::make_shared<dvm::ShapeRef>();
      return (*shapes_ref_)[axis_input].get();
    } else {
      auto shape = GetAxisList(axis_input);
      shapes_ref_source_->push_back(shape);
      (*shapes_ref_)[axis_input] = std::make_shared<dvm::ShapeRef>(shapes_ref_source_->back());
      return (*shapes_ref_)[axis_input].get();
    }
  }

  dvm::NDObject *EmitScalarBroadcast(const AnfNodePtr &node, dvm::ShapeRef *shape) {
    auto type_id = GetValueNodeType(node);
    auto v_type_id = ms_type_map[type_id];
    dvm::NDObject *op = nullptr;
    switch (type_id) {
      case kNumberTypeFloat32: {
        op = kernel_->Broadcast(GetScalarFromNode<float>(node), shape, v_type_id, empty_input_);
        break;
      }
      case kNumberTypeFloat16: {
        op = kernel_->Broadcast(GetScalarFromNode<dvm::Float16>(node), shape, v_type_id, empty_input_);
        break;
      }
      case kNumberTypeBFloat16: {
        op = kernel_->Broadcast(GetScalarFromNode<dvm::BFloat16>(node), shape, v_type_id, empty_input_);
        break;
      }
      case kNumberTypeInt32: {
        op = kernel_->Broadcast(GetScalarFromNode<int32_t>(node), shape, v_type_id, empty_input_);
        break;
      }
      default:
        MS_LOG(EXCEPTION) << "Some input of node " << node->fullname_with_scope() << " has unsupported dtype "
                          << TypeIdToString(type_id);
    }
    if (empty_input_) {
      empty_input_ = false;  // now we have a fake input
    }
    return op;
  }

  dvm::NDObject *EmitCast(dvm::NDObject *obj, TypeId dst_type) {
    auto it = ms_type_map.find(dst_type);
    if (it == ms_type_map.end()) {
      MS_LOG(EXCEPTION) << "Unsupported data type '" << TypeIdToString(dst_type) << "' for Cast";
    }
    if (kernel_->GetDType(obj) == it->second) {
      return obj;
    }
    return kernel_->Cast(obj, it->second);
  }

  void EmitOp(const AnfNodePtr &node, dvm::NDObject *obj) {
    ops_map_[node] = obj;
    if (outputs_.find(node) != outputs_.end()) {
      // hit subgraph output
      auto out_type = AnfAlgo::GetOutputDeviceDataType(node, 0);
      obj = EmitCast(obj, out_type);
      outputs_[node] = kernel_->Store(nullptr, obj);
    }
  }

  dvm::Kernel *kernel_;
  std::unordered_map<AnfNodePtr, dvm::NDObject *> inputs_;
  std::unordered_map<AnfNodePtr, dvm::NDObject *> outputs_;
  std::unordered_map<AnfNodePtr, dvm::NDObject *> ops_map_;
  std::unordered_map<AnfNodePtr, ShapeRefPtr> *shapes_ref_;
  std::vector<ShapeVector> *shapes_ref_source_;
  static std::unordered_map<dvm::DType, TypeId> v_type_map;
  static std::unordered_map<TypeId, dvm::DType> ms_type_map;
  bool empty_input_{false};
};

std::unordered_map<dvm::DType, TypeId> OpBuilder::v_type_map = {
  {dvm::DType::kFloat32, TypeId::kNumberTypeFloat32}, {dvm::DType::kFloat16, TypeId::kNumberTypeFloat16},
  {dvm::DType::kBool, TypeId::kNumberTypeBool},       {dvm::DType::kInt32, TypeId::kNumberTypeInt32},
  {dvm::DType::kInt64, TypeId::kNumberTypeInt64},     {dvm::DType::kBFloat16, TypeId::kNumberTypeBFloat16}};

std::unordered_map<TypeId, dvm::DType> OpBuilder::ms_type_map = {
  {TypeId::kNumberTypeFloat32, dvm::DType::kFloat32}, {TypeId::kNumberTypeFloat16, dvm::DType::kFloat16},
  {TypeId::kNumberTypeBool, dvm::DType::kBool},       {TypeId::kNumberTypeInt32, dvm::DType::kInt32},
  {TypeId::kNumberTypeInt64, dvm::DType::kInt64},     {TypeId::kNumberTypeBFloat16, dvm::DType::kBFloat16}};

size_t GetSubGraphNums(FuncGraphPtr graph_kernel) {
  auto output = graph_kernel->get_return()->cast<CNodePtr>()->input(1);
  MS_EXCEPTION_IF_NULL(output);
  auto maketuple = output->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(maketuple);
  auto prim = GetCNodePrimitive(maketuple->inputs().back());
  MS_EXCEPTION_IF_NULL(prim);
  auto value = prim->GetAttr("parallel_dim_info");
  auto info = GetValue<std::vector<size_t>>(value);
  return info[0] + 1;
}

FuncGraphPtr GetNodeFuncGraph(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto func_graph = common::AnfAlgo::GetNodeAttr<FuncGraphPtr>(node, kAttrFuncGraph);
  if (func_graph == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Can not get dvm func graph from node[" << node->fullname_with_scope() << "] "
                                      << node->DebugString();
  }
  return func_graph;
}

class DvmKernelBuilder {
 public:
  DvmKernelBuilder(const AnfNodePtr &node, bool is_dynamic) : node_(node), is_dynamic_(is_dynamic) {
    MS_EXCEPTION_IF_NULL(node_);
    kernel_name_ = common::AnfAlgo::GetCNodeName(node_);
    kernel_full_name_ = node_->fullname_with_scope();
  }
  ~DvmKernelBuilder() = default;

  virtual void BuildKernel(const FuncGraphPtr &graph, const CNodePtr &out_node,
                           const std::vector<AnfNodePtr> &outputs) = 0;

  void Construct(const FuncGraphPtr &graph) {
    MS_EXCEPTION_IF_NULL(graph);
    AnfNodePtr end_node = graph->get_return();
    MS_EXCEPTION_IF_NULL(end_node);
    auto ret_node = end_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(ret_node);
    auto out_node = ret_node->input(1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(out_node);
    std::vector<AnfNodePtr> outputs;
    if (IsPrimitiveCNode(out_node, prim::kPrimMakeTuple)) {
      auto tuple = out_node->cast<CNodePtr>();
      for (size_t i = 1; i < tuple->size(); ++i) {
        outputs.emplace_back(tuple->input(i));
      }
    } else {
      outputs.emplace_back(out_node);
    }
    BuildKernel(graph, out_node, outputs);
    CacheRefPair();
  }

  KernelModPtr Create() {
    auto cnode = node_->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto scope = cnode->fullname_with_scope();
    // FuncGraph --> Dvm Kernel
    auto func_graph = GetNodeFuncGraph(cnode);
    Construct(func_graph);
    if (kernel_mod_->EnableDump()) {
      kernel_mod_->DumpBuffer() << "[func graph] " << kernel_name_ << " " << kernel_full_name_ << "\n";
      DumpIR(kernel_mod_->DumpBuffer(), func_graph, false);
      kernel_mod_->DumpRefPair();
      kernel_mod_->DumpToFile();
    }
    if (!is_dynamic_) {
      // Static shape need codegen
      std::vector<ShapeVector> inputs_shape(cnode->size() - 1);
      for (size_t i = 0; i < inputs_shape.size(); ++i) {
        inputs_shape[i] = AnfAlgo::GetInputDeviceShape(cnode, i);
      }
      std::vector<ShapeVector> outputs_shape(kernel_mod_->GetOutputNum());
      for (size_t i = 0; i < outputs_shape.size(); ++i) {
        outputs_shape[i] = AnfAlgo::GetOutputDeviceShape(cnode, i);
      }
      kernel_mod_->CodeGen(inputs_shape, outputs_shape);
    } else {
      // Dynamic shape need create a prim to hold the infer shape function
      auto prim = std::make_shared<Primitive>(scope);
      prim->set_attr("infer_shape_functor", std::make_shared<DvmInfer>("dvm_infer_functor", kernel_mod_.get()));
      if (!std::static_pointer_cast<KernelMod>(kernel_mod_)->Init(prim, {}, {})) {
        MS_LOG(EXCEPTION) << "Initialize kernel module failed for node: " << scope;
      }
    }
    return kernel_mod_;
  }

 protected:
  const AnfNodePtr node_;
  std::string kernel_name_;
  std::string kernel_full_name_;
  bool is_dynamic_;
  DvmKernelModPtr kernel_mod_;

 private:
  void CacheRefPair() {
    auto info_ptr = node_->kernel_info();
    if (info_ptr == nullptr) {
      return;
    }
    auto kernel_info = dynamic_cast<device::KernelInfo *>(info_ptr);
    if (kernel_info == nullptr) {
      return;
    }
    kernel_mod_->CacheRefPair(kernel_info->out_in_ref_map());
  }
};

class SingleDvmKernelBuilder : public DvmKernelBuilder {
 public:
  SingleDvmKernelBuilder(const AnfNodePtr anf_node, bool is_dynamic) : DvmKernelBuilder(anf_node, is_dynamic) {}
  ~SingleDvmKernelBuilder() = default;

  void BuildKernel(const FuncGraphPtr &graph, const CNodePtr &out_node,
                   const std::vector<AnfNodePtr> &outputs) override {
    // Create kernel mod
    auto nodes = TopoSort(out_node);
    if (outputs.size() > 1) {
      nodes.pop_back();  // exclude maketuple
    }
    auto kernel_type = GetKernelType(&nodes);
    kernel_mod_ = std::make_shared<SingleDvmKernelMod>(kernel_type, kernel_name_, kernel_full_name_);
    auto inputs_type = AnfAlgo::GetAllInputDeviceTypes(node_);
    auto outputs_type = AnfAlgo::GetAllOutputDeviceTypes(node_);
    kernel_mod_->Initialize(inputs_type, outputs_type);
    const auto &params = graph->parameters();
    std::unordered_map<AnfNodePtr, ShapeRefPtr> shapes_ref;
    auto kernel_mod = std::static_pointer_cast<SingleDvmKernelMod>(kernel_mod_);
    OpBuilder builder(kernel_mod_->Kernel(), outputs, &shapes_ref, kernel_mod->ShapesSource(), params.empty());
    for (const auto &node : nodes) {
      if (node->isa<CNode>()) {
        builder.Emit(node);
      }
    }
    for (const auto &iter : shapes_ref) {
      kernel_mod->CacheShapeRef(iter.second);
    }
    // cache kernel's inputs and outputs from subgraph's inputs and outputs
    for (size_t i = 0; i < params.size(); ++i) {
      auto shape_iter = shapes_ref.find(params[i]);
      auto ref = shape_iter == shapes_ref.end() ? nullptr : shape_iter->second.get();
      kernel_mod->UpdateInputShapeRef(i, ref);
      if (auto load = builder.GetLoad(params[i]); load != nullptr) {
        kernel_mod->CacheLoad(load, i);
      }
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (auto store = builder.GetStore(outputs[i]); store != nullptr) {
        kernel_mod->CacheStore(store, i);
      }
    }
    kernel_mod->UpdateIO();
  }

 private:
  dvm::KernelType GetKernelType(std::vector<AnfNodePtr> *nodes) {
    auto iter = std::find_if(nodes->begin(), nodes->end(), [](const AnfNodePtr &node) {
      return IsPrimitiveCNode(node, prim::kPrimMatMul) || IsPrimitiveCNode(node, prim::kPrimBatchMatMul) ||
             IsPrimitiveCNode(node, prim::kPrimGroupedMatmul);
    });
    if (iter != nodes->end()) {
      std::rotate(nodes->begin(), iter, iter + 1);
      return is_dynamic_ ? dvm::KernelType::kDynMix : dvm::KernelType::kStaticMix;
    } else {
      return is_dynamic_ ? dvm::KernelType::kDynShape : dvm::KernelType::kStaticShape;
    }
  }
};

class ParallelDvmKernelBuilder : public DvmKernelBuilder {
 public:
  explicit ParallelDvmKernelBuilder(const AnfNodePtr anf_node, bool is_dynamic, size_t sub_graph_count)
      : DvmKernelBuilder(anf_node, is_dynamic), sub_graph_count_(sub_graph_count) {}
  ~ParallelDvmKernelBuilder() = default;

  void BuildKernel(const FuncGraphPtr &graph, const CNodePtr &out_node,
                   const std::vector<AnfNodePtr> &outputs) override {
    // Create kernel mod
    kernel_mod_ = std::make_shared<ParallelDvmKernelMod>(dvm::KernelType::kStaticParallel, kernel_name_,
                                                         kernel_full_name_, sub_graph_count_);
    auto inputs_type = AnfAlgo::GetAllInputDeviceTypes(node_);
    auto outputs_type = AnfAlgo::GetAllOutputDeviceTypes(node_);
    kernel_mod_->Initialize(inputs_type, outputs_type);
    const auto &output_groups = GetOutputGroups(sub_graph_count_, outputs);
    const auto &total_nodes = GetSubGraphs(output_groups);
    std::vector<OpBuilder> builders;
    builders.reserve(sub_graph_count_);
    const auto &params = graph->parameters();
    auto kernel_mod = std::static_pointer_cast<ParallelDvmKernelMod>(kernel_mod_);
    std::vector<std::unordered_map<AnfNodePtr, ShapeRefPtr>> shapes_ref(sub_graph_count_);
    for (size_t i = 0; i < sub_graph_count_; i++) {
      size_t param_input_num = std::count_if(total_nodes[i].begin(), total_nodes[i].end(),
                                             [](const AnfNodePtr &node) { return node->isa<Parameter>(); });
      builders.emplace_back(kernel_mod_->Kernel(), output_groups[i], &shapes_ref[i], kernel_mod->ShapesSource(i),
                            param_input_num == 0);
    }
    for (size_t i = 0; i < sub_graph_count_; i++) {
      const auto &nodes = total_nodes[i];
      for (const auto &node : nodes) {
        if (node->isa<CNode>()) {
          builders[i].Emit(node);
        }
      }
      if (i != sub_graph_count_ - 1) {
        (void)kernel_mod_->Kernel()->ParallelNext();
      }
    }
    for (size_t graph_idx = 0; graph_idx < sub_graph_count_; ++graph_idx) {
      for (const auto &iter : shapes_ref[graph_idx]) {
        kernel_mod->CacheShapeRef(iter.second);
      }
    }

    // cache kernel's inputs and outputs from subgraph's inputs and outputs
    for (size_t i = 0; i < params.size(); ++i) {
      for (size_t graph_idx = 0; graph_idx < sub_graph_count_; ++graph_idx) {
        auto shape_iter = shapes_ref[graph_idx].find(params[i]);
        auto ref = shape_iter == shapes_ref[graph_idx].end() ? nullptr : shape_iter->second.get();
        kernel_mod->UpdateInputShapeRef(i, ref);
        if (auto load = builders[graph_idx].GetLoad(params[i]); load != nullptr) {
          kernel_mod->CacheLoad(load, graph_idx, i);
        }
      }
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      for (size_t graph_idx = 0; graph_idx < sub_graph_count_; ++graph_idx) {
        if (auto store = builders[graph_idx].GetStore(outputs[i]); store != nullptr) {
          kernel_mod->CacheStore(store, graph_idx, i);
        }
      }
    }
    kernel_mod->UpdateIO();
  }

 private:
  const std::vector<std::vector<AnfNodePtr>> GetOutputGroups(size_t sub_graph_count,
                                                             const std::vector<AnfNodePtr> &outputs) {
    std::vector<std::vector<AnfNodePtr>> output_groups(sub_graph_count, std::vector<AnfNodePtr>());
    for (auto output : outputs) {
      auto output_prim = GetCNodePrimitive(output);
      MS_EXCEPTION_IF_NULL(output_prim);
      auto attrs = output_prim->attrs();
      if (attrs.find("parallel_dim_info") == attrs.end()) {
        MS_LOG(EXCEPTION) << "Can't find parallel_dim_info for parallel fusion, please check";
      }
      auto value = attrs["parallel_dim_info"];
      auto info = GetValue<std::vector<size_t>>(value);
      output_groups[info[0]].push_back(output);
    }
    return output_groups;
  }

  const std::vector<std::vector<AnfNodePtr>> GetSubGraphs(const std::vector<std::vector<AnfNodePtr>> &output_groups) {
    std::vector<std::vector<AnfNodePtr>> total_nodes(output_groups.size());
    for (size_t i = 0; i < output_groups.size(); i++) {
      const auto &output_group = output_groups[i];
      if (output_group.size() == 1) {
        auto output = output_group[0];
        auto subgraph = TopoSort(output);
        total_nodes[i] = subgraph;
      } else {
        auto maketuple = std::make_shared<CNode>(output_group, output_group[0]->func_graph());
        auto subgraph = TopoSort(maketuple);
        subgraph.pop_back();  // exclude maketuple
        total_nodes[i] = subgraph;
      }
    }
    return total_nodes;
  }

  size_t sub_graph_count_{0};
};
}  // namespace

KernelModPtr DvmOpBuild(const AnfNodePtr &anf_node) {
  static bool init = false;
  if (!init) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    bool enable_deterministic = ms_context->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON";
    bool enable_tuning = graphkernel::GraphKernelFlags::GetInstance().online_tuning > 0;
    if (enable_tuning && enable_deterministic) {
      enable_tuning = false;
      MS_LOG(WARNING) << "Since the result is required to be deterministic, online tuning is disabled.";
    }
    dvm::SetOnlineTuning(enable_tuning);
    dvm::SetDeterministic(enable_deterministic);
    init = true;
    MS_LOG(INFO) << "Set dvm deterministic " << enable_deterministic;
    MS_LOG(INFO) << "Set dvm online tuning " << enable_tuning;
  }
  MS_EXCEPTION_IF_NULL(anf_node);
  auto scope = anf_node->fullname_with_scope();
  MS_LOG(INFO) << "Start creating dvm kernel module for node: " << scope;
  auto func_graph = GetNodeFuncGraph(anf_node);
  std::shared_ptr<DvmKernelBuilder> kernel_builder{nullptr};
  auto is_dynamic = common::AnfAlgo::IsDynamicShape(anf_node);
  if (func_graph->has_attr(kAttrCompositeType) &&
      GetValue<std::string>(func_graph->get_attr(kAttrCompositeType)) == "parallel_fusion") {
    MS_EXCEPTION_IF_CHECK_FAIL(!is_dynamic, "Parallel fusion only supports static shape situations");
    auto sub_graph_count = GetSubGraphNums(func_graph);
    kernel_builder = std::make_shared<ParallelDvmKernelBuilder>(anf_node, is_dynamic, sub_graph_count);
  } else {
    kernel_builder = std::make_shared<SingleDvmKernelBuilder>(anf_node, is_dynamic);
  }
  auto kernel = kernel_builder->Create();
  MS_LOG(INFO) << "End creating dvm kernel module for node: " << scope;
  return kernel;
}
}  // namespace kernel
}  // namespace mindspore
