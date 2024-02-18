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
#include "plugin/device/ascend/kernel/dvm/dvm_kernel_build.h"
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include "plugin/device/ascend/kernel/dvm/dvm_kernel_mod.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"

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
};

static std::unordered_map<std::string, std::pair<OpType, int>> op_type_map = {
  {"Abs", {OP_UNARY, dvm::UnaryOpType::kAbs}},
  {"Exp", {OP_UNARY, dvm::UnaryOpType::kExp}},
  {"Log", {OP_UNARY, dvm::UnaryOpType::kLog}},
  {"Sqrt", {OP_UNARY, dvm::UnaryOpType::kSqrt}},
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
  {"IsFinite", {OP_UNARY, dvm::UnaryOpType::kIsFinite}}};

TypeId GetValueNodeType(const AnfNodePtr &node) {
  auto valuenode = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(valuenode);
  auto input_tensor = valuenode->value()->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto type_id = input_tensor->data_type();
  if (type_id != TypeId::kNumberTypeFloat32 && type_id != TypeId::kNumberTypeFloat16 &&
      type_id != TypeId::kNumberTypeInt32) {
    MS_LOG(EXCEPTION) << "Data type of scalar value input only supports float, but got: " << TypeIdToString(type_id)
                      << " node: " << node->fullname_with_scope();
  }
  return type_id;
}

template <typename T>
T GetScalarFromNode(const AnfNodePtr &node) {
  auto valuenode = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(valuenode);
  auto input_tensor = valuenode->value()->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  return TensorValueToVector<T>(input_tensor)[0];
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
        auto op = EmitCast(GetInput(node->input(1)), dst_dtype);
        EmitOp(anf_node, op);
        break;
      }
      case OP_SELECT: {
        auto op = kernel_->Select(GetInput(node->input(1)), GetInput(node->input(2)), GetInput(node->input(3)));
        EmitOp(anf_node, op);
        break;
      }
      case OP_UNARY: {
        auto op = kernel_->Unary(op_type->second.second, GetInput(node->input(1)));
        EmitOp(anf_node, op);
        break;
      }
      case OP_RSQRT: {
        auto sqrt_op = kernel_->Unary(dvm::UnaryOpType::kSqrt, GetInput(node->input(1)));
        auto op = kernel_->Unary(dvm::UnaryOpType::kReciprocal, sqrt_op);
        EmitOp(anf_node, op);
        break;
      }
      case OP_RESHAPE: {
        auto shape_ref = CacheShape(node, 2);
        auto op = kernel_->Reshape(GetInput(node->input(1)), shape_ref);
        EmitOp(anf_node, op);
        break;
      }
      case OP_BINARY: {
        auto op = EmitBinaryOp(node, op_type->second.second);
        EmitOp(anf_node, op);
        break;
      }
      case OP_BROADCAST: {
        auto input = node->input(1);
        auto shape_ref = CacheShape(node, 2);
        auto op = input->isa<ValueNode>() ? EmitScalarBroadcast(input, shape_ref)
                                          : kernel_->Broadcast(GetInput(input), shape_ref);
        EmitOp(anf_node, op);
        break;
      }
      case OP_NEG: {
        auto obj = GetInput(node->input(1));
        if (kernel_->GetDType(obj) == dvm::kInt32) {
          auto op = kernel_->Binary(dvm::BinaryOpType::kMul, obj, -1);
          EmitOp(anf_node, op);
        } else {
          auto op = kernel_->Binary(dvm::BinaryOpType::kMul, obj, -1.0f);
          EmitOp(anf_node, op);
        }
        break;
      }
      case OP_ASSIGN: {
        auto out_type = AnfAlgo::GetOutputDeviceDataType(node, 0);
        auto input2 = EmitCast(GetInput(node->input(2)), out_type);
        // store the second input of assign to the output of subgraph
        // the output addr of subgraph equals to the corresponding parameter addr of subgraph
        if (outputs_.find(node) != outputs_.end()) {
          ops_map_[anf_node] = kernel_->Store(nullptr, input2);
          outputs_[anf_node] = ops_map_[anf_node];
        } else {
          MS_LOG(EXCEPTION) << "AssignOp " << node->fullname_with_scope() << " is not in graph kernel 's outputs.";
        }
        break;
      }
      case OP_ELEMENY: {
        auto dst_dtype = AnfAlgo::GetOutputDeviceDataType(node, 0);
        auto op = kernel_->ElemAny(EmitCast(GetInput(node->input(1)), dst_dtype));
        EmitOp(anf_node, op);
        break;
      }
      default:
        MS_LOG(EXCEPTION) << op_type->second << " is unsupported op type.";
        break;
    }
  }

  dvm::NDObject *EmitBinaryOp(const CNodePtr &node, int binary_type) {
    AnfNodePtr inputs[] = {node->input(1), node->input(2)};
    int scalar_index = -1;
    for (int i = 0; i < 2; i++) {
      auto shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(inputs[i]->Shape())[kShape];
      auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies{});
      if (inputs[i]->isa<ValueNode>()) {
        scalar_index = i;
        if (size != 1) {
          MS_LOG(EXCEPTION) << "In GraphKernel the input node " << inputs[i]->fullname_with_scope() << " of "
                            << node->fullname_with_scope() << " should have a size of 1, but get " << size;
        }
      }
    }
    dvm::NDObject *op = nullptr;
    if (scalar_index != -1) {
      auto scalar_node = inputs[scalar_index];
      auto common_node = inputs[scalar_index ^ 1];
      auto type_id = GetValueNodeType(scalar_node);
      if (type_id == kNumberTypeFloat32) {
        auto scalar = GetScalarFromNode<float>(scalar_node);
        op = scalar_index ? kernel_->Binary(binary_type, GetInput(common_node), scalar)
                          : kernel_->Binary(binary_type, scalar, GetInput(common_node));
      } else if (type_id == kNumberTypeFloat16) {
        auto scalar = static_cast<float>(GetScalarFromNode<float16>(scalar_node));
        op = scalar_index ? kernel_->Binary(binary_type, GetInput(common_node), scalar)
                          : kernel_->Binary(binary_type, scalar, GetInput(common_node));
      } else if (type_id == kNumberTypeInt32) {
        auto scalar = GetScalarFromNode<int32_t>(scalar_node);
        op = scalar_index ? kernel_->Binary(binary_type, GetInput(common_node), scalar)
                          : kernel_->Binary(binary_type, scalar, GetInput(common_node));
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
  dvm::NDObject *GetInput(const AnfNodePtr &node) {
    auto it = ops_map_.find(node);
    if (it == ops_map_.end()) {
      dvm::NDObject *op = nullptr;
      if (node->isa<ValueNode>()) {
        shapes_ref_source_->push_back({1});
        (*shapes_ref_)[node] = std::make_shared<dvm::ShapeRef>(shapes_ref_source_->back());
        op = EmitScalarBroadcast(node, (*shapes_ref_)[node].get());
      } else if (node->isa<Parameter>()) {
        // hit subgraph input
        auto type_id = AnfAlgo::GetOutputDeviceDataType(node, 0);
        auto iter = ms_type_map.find(type_id);
        if (iter == ms_type_map.end()) {
          MS_LOG(EXCEPTION) << node->ToString() << " 's type " << TypeIdToString(type_id)
                            << " is unsupported data type.";
        }
        auto shape = AnfAlgo::GetOutputDeviceShape(node, 0);
        shapes_ref_source_->push_back(shape);
        (*shapes_ref_)[node] = std::make_shared<dvm::ShapeRef>(shapes_ref_source_->back());
        op = kernel_->Load(nullptr, (*shapes_ref_)[node].get(), iter->second);
        inputs_[node] = op;
      } else {
        MS_LOG(EXCEPTION) << node->DebugString() << " is unsupported node type.";
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
      if (!param->isa<Parameter>()) {
        MS_LOG(EXCEPTION) << "For " << node->fullname_with_scope() << ", input[" << (input_idx - 1)
                          << "] must be a Parameter, but got: " << param->ToString();
      }
      (*shapes_ref_)[param] = std::make_shared<dvm::ShapeRef>();
      return (*shapes_ref_)[param].get();
    }
    shapes_ref_source_->push_back(shape);
    (*shapes_ref_)[node] = std::make_shared<dvm::ShapeRef>(shapes_ref_source_->back());
    return (*shapes_ref_)[node].get();
  }

  dvm::NDObject *EmitScalarBroadcast(const AnfNodePtr &node, dvm::ShapeRef *shape) {
    auto type_id = GetValueNodeType(node);
    auto v_type_id = ms_type_map[type_id];
    dvm::NDObject *op = nullptr;
    if (type_id == kNumberTypeFloat32) {
      auto scalar = GetScalarFromNode<float>(node);
      op = kernel_->Broadcast(scalar, shape, v_type_id, empty_input_);
    } else if (type_id == kNumberTypeFloat16) {
      auto scalar = static_cast<float>(GetScalarFromNode<float16>(node));
      op = kernel_->Broadcast(scalar, shape, v_type_id, empty_input_);
    } else if (type_id == kNumberTypeInt32) {
      auto scalar = GetScalarFromNode<int32_t>(node);
      op = kernel_->Broadcast(scalar, shape, v_type_id, empty_input_);
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

std::unordered_map<dvm::DType, TypeId> OpBuilder::v_type_map = {{dvm::DType::kFloat32, TypeId::kNumberTypeFloat32},
                                                                {dvm::DType::kFloat16, TypeId::kNumberTypeFloat16},
                                                                {dvm::DType::kInt8, TypeId::kNumberTypeBool},
                                                                {dvm::DType::kInt32, TypeId::kNumberTypeInt32}};

std::unordered_map<TypeId, dvm::DType> OpBuilder::ms_type_map = {{TypeId::kNumberTypeFloat32, dvm::DType::kFloat32},
                                                                 {TypeId::kNumberTypeFloat16, dvm::DType::kFloat16},
                                                                 {TypeId::kNumberTypeBool, dvm::DType::kInt8},
                                                                 {TypeId::kNumberTypeInt32, dvm::DType::kInt32}};

class DvmKernelBuilder {
 public:
  DvmKernelBuilder() = default;
  ~DvmKernelBuilder() = default;

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
      end_node = out_node;
    } else {
      outputs.emplace_back(out_node);
    }
    const auto &params = graph->parameters();
    std::unordered_map<AnfNodePtr, ShapeRefPtr> shapes_ref;
    OpBuilder builder(kernel_mod_->Kernel(), outputs, &shapes_ref, kernel_mod_->ShapesSource(), params.empty());
    auto nodes = TopoSort(ret_node);
    for (const auto &node : nodes) {
      if (node == end_node) break;
      if (node->isa<CNode>()) {
        builder.Emit(node);
      }
    }
    for (const auto &iter : shapes_ref) {
      kernel_mod_->CacheShapeRef(iter.second);
    }
    // cache kernel's inputs and outputs from subgraph's inputs and outputs
    for (size_t i = 0; i < params.size(); ++i) {
      auto shape_iter = shapes_ref.find(params[i]);
      auto ref = shape_iter == shapes_ref.end() ? nullptr : shape_iter->second.get();
      kernel_mod_->UpdateInputShapeRef(i, ref);
      if (auto load = builder.GetLoad(params[i]); load != nullptr) {
        kernel_mod_->CacheLoad(load, i);
      }
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (auto store = builder.GetStore(outputs[i]); store != nullptr) {
        kernel_mod_->CacheStore(store, i);
      }
    }
    kernel_mod_->UpdateIO();
  }

  KernelModPtr Create(const AnfNodePtr &anf_node) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto scope = cnode->fullname_with_scope();
    MS_LOG(INFO) << "Start creating kernel module for node: " << scope;
    // Create kernel mod
    auto is_dynamic = common::AnfAlgo::IsDynamicShape(anf_node);
    kernel_mod_ = std::make_shared<DvmKernelMod>(is_dynamic);
    auto inputs_type = AnfAlgo::GetAllInputDeviceTypes(cnode);
    auto outputs_type = AnfAlgo::GetAllOutputDeviceTypes(cnode);
    kernel_mod_->Initialize(inputs_type, outputs_type);
    // FuncGraph --> Dvm Kernel
    auto func_graph = GetCNodeFuncGraph(cnode);
    Construct(func_graph);
    if (!is_dynamic) {
      // Static shape need codegen
      std::vector<ShapeVector> inputs_shape(inputs_type.size());
      for (size_t i = 0; i < inputs_type.size(); ++i) {
        inputs_shape[i] = AnfAlgo::GetInputDeviceShape(cnode, i);
      }
      std::vector<ShapeVector> outputs_shape(outputs_type.size());
      for (size_t i = 0; i < outputs_type.size(); ++i) {
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
    MS_LOG(INFO) << "End creating kernel module for node: " << scope;
    return kernel_mod_;
  }

 private:
  DvmKernelModPtr kernel_mod_;
};
}  // namespace

KernelModPtr DvmOpBuild(const AnfNodePtr &anf_node) {
  DvmKernelBuilder kernel_builder;
  return kernel_builder.Create(anf_node);
}
}  // namespace kernel
}  // namespace mindspore
