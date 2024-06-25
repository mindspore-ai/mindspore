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

#include "backend/common/expander/fallback/fallback_irbuilder.h"
#include <algorithm>
#include <vector>
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "utils/anf_utils.h"
#include "include/common/expander/core/emitter.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/convert_utils.h"
#include "include/backend/kernel_info.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace expander {
class InferHostAndDevice : public CppInfer {
 public:
  InferHostAndDevice(const SelectKernelFunc &func, bool *result)
      : select_kernel_func_(func), select_kernel_result_(*result) {}
  ~InferHostAndDevice() = default;

  void Infer(const NodePtr &node) override {
    CppInfer::Infer(node);
    if (select_kernel_result_) {
      SetKernelInfo(node->get());
    }
  }

  void SetDynamicShapeAttr(const CNodePtr &cnode) {
    if (common::AnfAlgo::IsNodeInputDynamicShape(cnode)) {
      common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), cnode);
    }
    if (common::AnfAlgo::IsNodeOutputDynamicShape(cnode)) {
      common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), cnode);
    }
  }

  void SetKernelInfo(const AnfNodePtr &node) {
    if (AnfUtils::IsRealCNodeKernel(node)) {
      auto cnode = node->cast<CNodePtr>();
      SetDynamicShapeAttr(cnode);
      select_kernel_result_ = select_kernel_func_(cnode);
      if (!select_kernel_result_) {
        MS_LOG(INFO) << "Select kernel for " << cnode->fullname_with_scope()
                     << " unsuccessful. node:" << cnode->DebugString();
      }
    } else {
      // virtual cnode or Value node
      node->set_kernel_info(std::make_shared<device::KernelInfo>());
      auto vnode = node->cast<ValueNodePtr>();
      if (vnode != nullptr) {
        auto tensor = vnode->value()->cast<tensor::TensorPtr>();
        if (tensor != nullptr) {
          kernel::KernelBuildInfo::KernelBuildInfoBuilder info_builder;
          info_builder.SetOutputsFormat({kOpFormat_DEFAULT});
          info_builder.SetOutputsDeviceType({tensor->Dtype()->type_id()});
          AnfAlgo::SetSelectKernelBuildInfo(info_builder.Build(), node.get());
        }
      }
    }
  }

 protected:
  SelectKernelFunc select_kernel_func_;
  bool &select_kernel_result_;
};

FallbackIRBuilder::FallbackIRBuilder(const std::string &name, const FuncGraphPtr &fg, const SelectKernelFunc &func)
    : IrEmitter(fg, std::make_shared<InferHostAndDevice>(func, &success_),
                std::make_shared<Scope>(std::string("Expand/_") + name)),
      name_(name) {}
AnfNodePtr FallbackIRBuilder::Run(const CNodePtr &cnode, const IRBuilderHandle &handle) {
  inputs_.resize(cnode->size() - 1);
  (void)std::transform(cnode->weak_inputs().cbegin() + 1, cnode->weak_inputs().cend(), inputs_.begin(),
                       [this](const AnfNodeWeakPtr &no) { return this->NewIrNode(no.lock()); });
  attrs_ptr_ = &(GetCNodePrimitive(cnode)->attrs());
  auto outputs = handle.func(this);
  if (!success_ || outputs.empty()) {
    MS_LOG(DEBUG) << "Exec func result: success=" << success_ << ", outputs.empty()=" << outputs.empty();
    return nullptr;
  }
  if (outputs.size() > 1) {
    return this->MakeTuple(outputs)->get();
  }
  return outputs[0]->get();
}

ValuePtr FallbackIRBuilder::GetAttr(const std::string &attr) const {
  auto iter = attrs_ptr_->find(attr);
  if (iter != attrs_ptr_->end()) {
    return iter->second;
  }
  MS_LOG(WARNING) << "The attr " << attr << " does not exist in op " << name_;
  return nullptr;
}

int64_t FallbackIRBuilder::GetSize(const NodePtr &node) const {
  auto shape = GetShape(node);
  return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
}
// DEF_PURE_SHAPE_CALC's name cannot be same now
DEF_PURE_SHAPE_CALC(g_dyn_size_2)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray { return {{abstract::ShapeSize(inputs.at(0))}}; })
  .SetInfer([](const ShapeArray &, const HashSet<size_t> &) -> ShapeVector { return {1}; });

NodePtr FallbackIRBuilder::DynSize(const NodePtr &node) {
  if (!IsDynamic(GetShape(node))) {
    return Tensor(GetSize(node), kInt64);
  }
  return SequenceToTensor(ShapeCalc(g_dyn_size_2, {node})[0]);
}

NodePtr FallbackIRBuilder::DynSize(const NodePtr &node, const TypePtr &type) {
  return Cast(SequenceToTensor(DynSize(node)), type);
}

NodePtr FallbackIRBuilder::SequenceToTensor(const NodePtr &node, const TypePtr &dtype) {
  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractSequence>()) {
    if (node->input_type() == InputType::kConstant) {
      return Tensor(GetIntList(node), dtype);
    }
    if (abs->isa<abstract::AbstractTuple>()) {
      return Emit(kTupleToTensorOpName, {node, Value(static_cast<int64_t>(dtype->type_id()))});
    } else {
      return Emit(kListToTensorOpName, {node, Value(dtype)});
    }
  }
  return node;
}

std::vector<int64_t> FallbackIRBuilder::GetIntList(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->data_sync();
    return CheckAndConvertUtils::CheckTensorIntValue("tensor", value, "bprop");
  } else {
    return CheckAndConvertUtils::CheckIntOrTupleInt("value", value, "bprop");
  }
}

std::vector<int64_t> FallbackIRBuilder::GetIntList(const NodePtr &node) {
  auto value = node->BuildValue();
  MS_EXCEPTION_IF_NULL(value);
  return GetIntList(value);
}
}  // namespace expander
}  // namespace mindspore
