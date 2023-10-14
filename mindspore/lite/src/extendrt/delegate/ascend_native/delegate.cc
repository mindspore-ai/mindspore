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

#include "extendrt/delegate/ascend_native/delegate.h"
#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <set>
#include <functional>
#include "extendrt/delegate/ascend_native/ascend_native_kernel_registry.h"
#include "extendrt/delegate/ascend_native/ops/ascend_native_composite.h"
#include "extendrt/delegate/ops/copy.h"
#include "extendrt/kernel/ascend_native/ascend_native_composite_kernel.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "extendrt/delegate/ascend_native/ops/ascend_native_stub.h"
#include "extendrt/delegate/factory.h"
#include "include/common/utils/convert_utils.h"

#include "ops/encoder_layer.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "ops/use_past_embedding.h"
#include "ops/gather.h"
#include "ops/reshape.h"
#include "ops/cast.h"
#include "ops/not_equal.h"
#include "ops/tuple_get_item.h"
#include "ops/less.h"

namespace mindspore {

constexpr auto kAscendNativeProvider = "ascend_native";
namespace {
static inline kernel::InferTensor *anfTensorToTensorInfo(const common::KernelWithIndex &tensor_id) {
  auto [prev_node, index] = tensor_id;
  auto data_type = FuncGraphUtils::GetTensorDataType(tensor_id);
  auto tensor_val = FuncGraphUtils::GetConstNodeValue(prev_node);
  auto shape = FuncGraphUtils::GetTensorShape(tensor_id);
  auto name = FuncGraphUtils::GetTensorName(tensor_id);
  constexpr auto tensorrt_format = mindspore::Format::NCHW;
  const void *data = nullptr;
  size_t data_len = 0;
  if (tensor_val) {
    data = tensor_val->data_c();
    data_len = tensor_val->Size();
    shape = tensor_val->shape_c();
  } else {
    if (data_type == DataType::kObjectTypeTuple) {
      auto tuple_abs = prev_node->abstract()->cast<abstract::AbstractTuplePtr>();
      auto abs = tuple_abs->elements().at(index);
      data_type = static_cast<enum DataType>(abs->BuildType()->type_id());
      auto base_shape = abs->BuildShape();
      auto shape_ptr = base_shape->cast<abstract::ShapePtr>();
      if (shape_ptr != nullptr) {
        shape = shape_ptr->shape();
      }
    }
  }
  auto format = tensorrt_format;
  std::vector<int> t_shape;
  t_shape.resize(shape.size());
  std::transform(shape.begin(), shape.end(), t_shape.begin(), [](int64_t x) { return static_cast<int>(x); });
  auto t = kernel::InferTensor::CreateTensor(name, static_cast<TypeId>(data_type), t_shape, data, data_len);
  t->set_format(format);
  return t;
}

static inline BaseOperatorPtr CreateOperatorByCNode(const CNodePtr &cnode) {
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  auto kernel_name = prim->name();
  // Create PrimtiveC from map and create BaseOperator.
  ops::PrimitiveCPtr primc_ptr = nullptr;
  static auto primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  if (primc_fns.find(kernel_name) != primc_fns.end()) {
    primc_ptr = primc_fns[kernel_name]();
    (void)primc_ptr->SetAttrs(prim->attrs());
  }
  MS_EXCEPTION_IF_NULL(primc_ptr);
  static auto operator_fns = ops::OperatorRegister::GetInstance().GetOperatorMap();
  if (operator_fns.find(kernel_name) == operator_fns.end()) {
    MS_LOG(EXCEPTION) << "Cannot create BaseOperator for " << kernel_name;
  }
  auto base_operator = operator_fns[kernel_name](primc_ptr);
  MS_EXCEPTION_IF_NULL(base_operator);
  return base_operator;
}
}  // namespace

bool AscendNativeDelegate::IsSupport(const CNodePtr &cnode) {
  bool ret = false;
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  std::set<std::string> ops = {ops::kNameEncoderLayer, ops::kNameAddFusion, ops::kNameMatMulFusion, ops::kNameGather,
                               ops::kNameTupleGetItem};
  if (ops.find(prim->name()) != ops.end()) {
    if (prim->name() == ops::kNameMatMulFusion) {
      auto base_op = CreateOperatorByCNode(cnode);
      if (base_op == nullptr) {
        MS_LOG(WARNING) << "no op found for " << cnode->fullname_with_scope();
        return false;
      }
      auto primitive = std::make_shared<ops::MatMulFusion>(base_op->GetPrim());
      if (primitive == nullptr) {
        MS_LOG(WARNING) << "cannot create primitive for MatMulFusion";
        return false;
      }
      bool act = primitive->get_activation_type();
      if ((act == ActivationType::NO_ACTIVATION) && (cnode->inputs().size() == Num3)) {
        ret = true;
      }
    } else if (prim->name() == ops::kNameAddFusion) {
      auto shape1 = mindspore::BaseShapeToShape(cnode->input(1)->Shape());
      auto shape2 = mindspore::BaseShapeToShape(cnode->input(2)->Shape());
      auto in1 = std::reduce(shape1.begin(), shape1.end(), 1.0, std::multiplies<int64_t>());
      auto in2 = std::reduce(shape2.begin(), shape2.end(), 1.0, std::multiplies<int64_t>());
      if (in1 == in2) ret = true;
    } else {
      ret = true;
    }
  }
  return ret;
}

void AscendNativeDelegate::ReplaceNodes(const std::shared_ptr<FuncGraph> &graph) {
  auto nodes = TopoSort(graph->get_return());
  // for all the nodes in the graph, call the delegate isDelegateNode and CreateKernel interface to create kernels
  helper_ = std::make_shared<SubGraphHelper>(graph);
  for (auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsSupport(cnode)) continue;
    // consider tuple only if parent is supported.
    auto prim = GetCNodePrimitive(cnode);
    if (prim->name() == ops::kNameTupleGetItem) {
      auto parent = cnode->input(1);
      if (parent->isa<CNode>() && !IsSupport(parent->cast<CNodePtr>())) continue;
    }
    // check all node inputs belong to same subgraph
    int sbg = helper_->CheckAllInputInSameSg(cnode);
    // if yes add node to subgraph
    if (sbg >= 0) {
      helper_->AddToSubGraph(sbg, cnode);
    } else {
      helper_->AddSubGraph(cnode);
    }
  }
  for (int i = 0; i < helper_->SubGroupNum(); i++) {
    ReplaceSubGraph(graph, i);
  }
  helper_->FixAllNodes(nodes);
}

void AscendNativeDelegate::ReplaceSubGraph(const std::shared_ptr<FuncGraph> &graph, int idx) {
  auto composite_prim = std::make_shared<ops::AscendNativeComposite>();
  if (composite_prim == nullptr) {
    MS_LOG(ERROR) << "failed to create custom node";
    return;
  }
  composite_prim->Init(idx);
  auto composite_prim_c = composite_prim->GetPrim();
  CNodePtr composite_node = graph->NewCNode(composite_prim_c, helper_->GetSbgInputs(idx));
  composite_node->set_fullname_with_scope("composite_" + std::to_string(idx));
  helper_->SetCNode(idx, composite_node);
  // abstract handled later
}

bool AscendNativeDelegate::IsDelegateNode(const std::shared_ptr<AnfNode> &node) {
  auto cnode = node->cast<CNodePtr>();
  auto copy_prim = std::make_shared<Primitive>(ops::kNameCopy);
  auto composite_prim = std::make_shared<Primitive>(ops::kNameAscendNativeComposite);
  if ((cnode != nullptr) && (IsPrimitiveCNode(cnode, composite_prim) || IsPrimitiveCNode(cnode, copy_prim)))
    return true;
  return false;
}

void AscendNativeDelegate::CreateInputKernelTensors(const CNodePtr &cnode,
                                                    std::vector<kernel::InferTensor *> *input_tensors,
                                                    std::shared_ptr<DelegateAllocator> allocator) {
  input_tensors->clear();
  auto input_nodes = FuncGraphUtils::GetNodeInputs(cnode);
  for (auto &tensor_id : input_nodes) {
    auto it = std::find_if(kernel_list_.begin(), kernel_list_.end(),
                           [&tensor_id](const KernelWithIndexAndTensor &k) { return k.kernel_index == tensor_id; });
    // tensor already created - use the same tensor
    if (it != kernel_list_.end()) {
      input_tensors->push_back(it->tensor_info);
    } else {
      auto tensor_info = anfTensorToTensorInfo(tensor_id);
      if (tensor_info == nullptr) {
        MS_LOG(ERROR) << "failed to get tensor info";
        return;
      }
      input_tensors->push_back(tensor_info);
      kernel_list_.push_back(KernelWithIndexAndTensor(tensor_id, tensor_info));
    }
  }
}

void AscendNativeDelegate::CreateOutputKernelTensors(const CNodePtr &cnode,
                                                     std::vector<kernel::InferTensor *> *output_tensors,
                                                     std::shared_ptr<DelegateAllocator> allocator) {
  output_tensors->clear();
  auto output_num = AnfUtils::GetOutputTensorNum(cnode);
  for (size_t output_idx = 0; output_idx < output_num; ++output_idx) {
    common::KernelWithIndex tensor_id = {cnode, output_idx};
    auto it = std::find_if(kernel_list_.begin(), kernel_list_.end(),
                           [&tensor_id](const KernelWithIndexAndTensor &k) { return k.kernel_index == tensor_id; });
    if (it != kernel_list_.end()) {
      output_tensors->push_back(it->tensor_info);
    } else {
      auto tensor_info = anfTensorToTensorInfo(tensor_id);
      output_tensors->push_back(tensor_info);
      kernel_list_.push_back(KernelWithIndexAndTensor(tensor_id, tensor_info));
    }
  }
}

std::shared_ptr<kernel::BaseKernel> AscendNativeDelegate::CreateKernel(const std::shared_ptr<AnfNode> &node) {
  // step I - Convert to cnode
  if (!node->isa<CNode>()) {
    MS_LOG(ERROR) << "AscendNativeDelegate::CreateKernel not a cnode";
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "AscendNativeDelegate::CreateKernel cnode is nullptr";
    return nullptr;
  }
  auto stream = ascend_native::CreateStream();
  auto allocator = std::make_shared<DelegateAllocator>(stream);
  // step II - Prepare kernel attributes
  std::vector<kernel::InferTensor *> input_tensors;
  CreateInputKernelTensors(cnode, &input_tensors, allocator);
  std::vector<kernel::InferTensor *> output_tensors;
  CreateOutputKernelTensors(cnode, &output_tensors, allocator);
  kernel::InferPrimitive primitive;
  primitive.base_operator = CreateOperatorByCNode(cnode);
  primitive.cnode = cnode;
  auto kernel_name = cnode->fullname_with_scope();
  auto node_type = primitive.base_operator->name();

  if (node_type == ops::kNameAscendNativeComposite) {
    auto kernel = std::make_shared<kernel::AscendNativeCompositeKernel>(input_tensors, output_tensors, primitive,
                                                                        ascend_native_ctx_.get(), stream, kernel_name);
    int idx = static_cast<int>(GetValue<int64_t>(primitive.base_operator->GetAttr("group")));
    auto func_graph = helper_->GetSbg(idx)->func_graph();
    kernel->set_func_graph(func_graph);
    kernel->set_stream(stream);
    return kernel;
  } else {
    // create base kernel for debug
    auto orig_node_type = node_type;
    // step III - Create Ascend native Kernel
    auto &plugin_factory = kernel::AscendNativeRegistrationFactory::Get();
    if (node_type != ops::kNameCopy) {
      node_type = ops::kNameAscendNativeStub;
    }
    if (plugin_factory.HasKey(node_type)) {
      kernel::AscendNativeBaseKernel *ascend_native_op = plugin_factory.GetCreator(node_type)(
        input_tensors, output_tensors, primitive, ascend_native_ctx_.get(), stream, node_type);
      if (ascend_native_op == nullptr) {
        return nullptr;
      }
      auto ker = std::shared_ptr<kernel::AscendNativeBaseKernel>(ascend_native_op);
      if (!ker->IsWeightInputHanledInner()) {
        auto in_tensors = ker->in_tensors();
        for (auto &t : in_tensors) {
          if (t->IsConst() && t->device_data() != nullptr) {
            t->set_device_data(ascend_native::MallocCopy(t->data(), t->Size(), const_cast<void *>(stream)));
          }
        }
      }
      if (node_type == "AscendNativeStub") {
        ker->set_name(orig_node_type);
      } else {
        ker->set_name(kernel_name);
      }
      ker->set_stream(stream);
      return ker;
    } else {
      MS_LOG(WARNING) << "Unsupported op type for ascend native. kernel name:" << kernel_name << " type:" << node_type;
      return nullptr;
    }
  }
}

void AscendNativeDelegate::CopyTensors(InferTensor *t_src, InferTensor *t_dst, const void *stream) const {
  auto dst = t_dst->device_data();
  auto elem = t_src->Size();
  bool t_is_float = (t_src->data_type() == kNumberTypeFloat || t_src->data_type() == kNumberTypeFloat32);
  if (t_is_float) {
    ascend_native::CopyHostFp32ToDeviceFp16(t_src->data(), &dst, elem, const_cast<void *>(stream));
  } else {
    int elem_size = mindspore::lite::DataTypeSize(t_src->data_type());
    switch (elem_size) {
      case Num4:
        ascend_native::CopyHostFp32ToDeviceFp32(t_src->data(), &dst, elem, const_cast<void *>(stream));
        break;
      case Num2:
        ascend_native::CopyHostFp16ToDeviceFp16(t_src->data(), &dst, elem, const_cast<void *>(stream));
        break;
      case Num1:
        ascend_native::CopyHostFp16ToDeviceFp16(t_src->data(), &dst, elem / 2, const_cast<void *>(stream));
        break;
      default:
        MS_LOG(ERROR) << "no supported size " << elem_size;
    }
  }
  t_dst->set_device_data(dst);
}

std::shared_ptr<kernel::BaseKernel> AscendNativeDelegate::CreateKernel(const kernel::KernelSpec &spec,
                                                                       const std::vector<InferTensor *> &inputs,
                                                                       const std::vector<InferTensor *> &outputs,
                                                                       const InferContext *ctx) const {
  // step I - Convert to cnode
  auto cnode = spec.cnode;
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "AscendNativeDelegate::CreateKernel cnode is nullptr";
    return nullptr;
  }
  // step II - Prepare kernel attributes
  auto kernel_name = cnode->fullname_with_scope();
  auto stream = ascend_native::CreateStream();
  if (stream == nullptr) {
    MS_LOG(ERROR) << "fail to create stream for kernel " << kernel_name;
    return nullptr;
  }
  kernel::InferPrimitive primitive;
  primitive.base_operator = spec.primitive;
  primitive.cnode = cnode;
  auto node_type = primitive.base_operator->name();
  if (node_type == ops::kNameAscendNativeComposite) {
    auto kernel = std::make_shared<kernel::AscendNativeCompositeKernel>(inputs, outputs, primitive,
                                                                        ascend_native_ctx_.get(), stream, kernel_name);
    int idx = static_cast<int>(GetValue<int64_t>(primitive.base_operator->GetAttr("group")));
    auto func_graph = helper_->GetSbg(idx)->func_graph();
    kernel->set_func_graph(func_graph);
    kernel->set_stream(stream);
    return kernel;
  } else {
    // step III - Create Ascend native Kernel
    auto &plugin_factory = kernel::AscendNativeRegistrationFactory::Get();
    if (plugin_factory.HasKey(node_type)) {
      kernel::AscendNativeBaseKernel *ascend_native_op =
        plugin_factory.GetCreator(node_type)(inputs, outputs, primitive, ascend_native_ctx_.get(), stream, node_type);
      if (ascend_native_op == nullptr) {
        return nullptr;
      }
      auto ker = std::shared_ptr<kernel::AscendNativeBaseKernel>(ascend_native_op);
      if (!ker->IsWeightInputHanledInner()) {
        auto in_tensors = ker->in_tensors();
        for (auto &t : in_tensors) {
          if (t->IsConst() && t->device_data() != nullptr) {
            CopyTensors(t, t, stream);
          }
        }
      }
      ker->set_name(kernel_name);
      ker->set_stream(stream);
      return ker;
    } else {
      MS_LOG(ERROR) << "Unsupported op type for ascend native. kernel name:" << kernel_name << " type:" << node_type;
      return nullptr;
    }
  }
}

ExtendDelegate *AscendDelegateCreator(const std::shared_ptr<Context> &, const ConfigInfos &) {
  return &AscendNativeDelegate::Instance();
}
REG_DELEGATE(kAscend, kAscendNativeProvider, AscendDelegateCreator);

}  // namespace mindspore
