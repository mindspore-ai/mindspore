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
    : Emitter(fg, std::make_shared<InferHostAndDevice>(func, &success_),
              std::make_shared<Scope>(std::string("Expand/_") + name)),
      name_(name) {}
AnfNodePtr FallbackIRBuilder::Run(const CNodePtr &cnode, const IRBuilderHandle &handle) {
  inputs_.resize(cnode->size() - 1);
  (void)std::transform(cnode->inputs().cbegin() + 1, cnode->inputs().cend(), inputs_.begin(),
                       [this](const AnfNodePtr &no) { return this->NewNode(no); });
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

void FallbackIRBuilder::ConvertConstInputToTensorInput(const PrimitivePtr &p, NodePtrList *inputs_ptr) {
  static const PrimitiveSet nochange_prims = {prim::kPrimMakeTuple, prim::kPrimTupleGetItem, prim::kPrimDepend,
                                              prim::kPrimStack};
  if (nochange_prims.find(p) != nochange_prims.end()) {
    return;
  }
  auto &inputs = *inputs_ptr;
  for (size_t i = 0; i < inputs.size(); i++) {
    if (!inputs[i]->isa<ValueNode>()) {
      continue;
    }
    const auto &value = inputs[i]->get<ValueNodePtr>()->value();
    if (value->isa<Scalar>()) {
      inputs[i] = EmitValue(ScalarToTensor(value->cast<ScalarPtr>()));
    } else if (value->isa<ValueTuple>()) {
      inputs[i] = EmitValue(opt::CreateTupleTensor(value->cast<ValueTuplePtr>()));
    } else if (value->isa<ValueList>()) {
      inputs[i] = EmitValue(opt::CreateTupleTensor(std::make_shared<ValueTuple>(value->cast<ValueListPtr>()->value())));
    }
  }
}

NodePtr FallbackIRBuilder::EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) {
  auto new_inputs = inputs;
  ConvertConstInputToTensorInput(prim, &new_inputs);
  return Emitter::EmitOp(prim, new_inputs);
}
}  // namespace expander
}  // namespace mindspore
