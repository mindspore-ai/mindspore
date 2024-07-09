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
#include "backend/common/graph_kernel/expander/mindir_adapter/mindir_emitter.h"
#include <algorithm>
#include "ir/primitive.h"
#include "backend/common/graph_kernel/expander/mindir_adapter/anf_node_holder.h"
#include "backend/common/graph_kernel/model/op_register.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore::graphkernel::expander {
NodePtr MindirEmitter::EmitOp(MetaOp op, const NodePtrList &args, const NodePtrDict &kargs) {
  auto func = emit_functions[static_cast<int>(op)];
  if (func == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "The emit function of op " << MetaOpStr[static_cast<int>(op)] << "("
                               << static_cast<int>(op) << ") is nullptr.";
  }
  return func(this, MetaOpStr[static_cast<int>(op)], args, kargs);
}

NodePtrList MindirEmitter::Inputs(const CNodePtr &cnode) {
  NodePtrList result(cnode->size() - 1);
  auto &inputs = cnode->inputs();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto node = inputs[i];
    // Tensor has no device type in build_info, so skip
    if (node->isa<ValueNode>()) {
      if (node->kernel_info() != nullptr) {
        auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
        if (build_info != nullptr) {
          auto outputs_format = build_info->GetAllOutputFormats();
          auto node_input_format = AnfAlgo::GetInputFormat(cnode, i - 1);
          // the format fetched from Tensor itself may be ND, but the format fetched from cnode's input may be
          // DefaultFormat, the ND format is a wrong format which will pad the Tensor's shape from 2D to 4D,
          // and cause later operator got a wrong input shape
          if (outputs_format.size() == 1 && outputs_format[0] != node_input_format) {
            MS_LOG(INFO) << "For node[" << cnode->fullname_with_scope() << "], inputs[" << (i - 1)
                         << "] node: " << node->DebugString() << ", update its format from '" << outputs_format[0]
                         << "' to '" << node_input_format << "'";
            outputs_format[0] = node_input_format;
            build_info->SetOutputsFormat(outputs_format);
          }
        }
      }
      result[i - 1] = NewNode(node);
      continue;
    }
    auto p = func_graph_->add_parameter();
    p->set_abstract(node->abstract());
    // Parameter has no output type in build info, so create one from input of cnode
    auto kernel_info = std::make_shared<device::KernelInfo>();
    p->set_kernel_info(kernel_info);
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    MS_EXCEPTION_IF_NULL(builder);
    builder->SetOutputsFormat({AnfAlgo::GetInputFormat(cnode, i - 1)});
    builder->SetOutputsDeviceType({AnfAlgo::GetInputDeviceDataType(cnode, i - 1)});
    kernel_info->set_select_kernel_build_info(builder->Build());
    result[i - 1] = NewNode(p);
  }
  infer_->HandleInputs(result);
  return result;
}

NodePtr MindirEmitter::DefaultEmitFunc(const std::string &op_name, const NodePtrList &args, const NodePtrDict &kargs) {
  auto prim = std::make_shared<Primitive>(op_name);
  if (!kargs.empty()) {
    HashMap<std::string, ValuePtr> attrs;
    for (auto &[k, v] : kargs) {
      attrs[k] = v->GetValue();
    }
    (void)prim->SetAttrs(attrs);
  }
  AnfNodePtrList inputs(args.size() + 1);
  inputs[0] = NewValueNode(prim);
  (void)std::transform(args.cbegin(), args.cend(), inputs.begin() + 1,
                       [](const NodePtr &node) { return node->as<AnfNodePtr>(); });
  auto node = NewNode(NewCNode(inputs));
  infer_->InferOp(node, prim, args);
  return node;
}

NodePtr MindirEmitter::ReduceEmitFunc(const std::string &op_name, const NodePtrList &args, const NodePtrDict &kargs) {
  constexpr size_t keep_dims_idx = 2;
  constexpr size_t skip_mode_idx = 3;
  auto prim = std::make_shared<Primitive>(op_name);
  HashMap<std::string, ValuePtr> attrs;
  attrs["keep_dims"] = args[keep_dims_idx]->GetValue();
  if (args.size() > skip_mode_idx) {
    attrs["skip_mode"] = args[skip_mode_idx]->GetValue();
  }
  (void)prim->SetAttrs(attrs);
  NodePtrList new_args{args[0], args[1]};
  AnfNodePtrList inputs(new_args.size() + 1);
  inputs[0] = NewValueNode(prim);
  (void)std::transform(new_args.cbegin(), new_args.cend(), inputs.begin() + 1,
                       [](const NodePtr &node) { return node->as<AnfNodePtr>(); });
  auto node = NewNode(NewCNode(inputs));
  infer_->InferOp(node, prim, new_args);
  return node;
}
}  // namespace mindspore::graphkernel::expander
