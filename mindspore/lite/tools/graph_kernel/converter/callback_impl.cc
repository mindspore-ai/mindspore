/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "tools/graph_kernel/converter/callback_impl.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>
#include "mindspore/core/ops/core_ops.h"
#include "ir/dtype.h"
#include "ir/func_graph.h"
#include "utils/anf_utils.h"
#include "include/common/utils/utils.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
// register the callback object
#ifndef ENABLE_CLOUD_FUSION_INFERENCE
GRAPH_KERNEL_CALLBACK_REGISTER(CallbackImpl);
#endif

ShapeVector CallbackImpl::GetInputShape(const AnfNodePtr &node, size_t i) { return GetInputInferShape(node, i); }

ShapeVector CallbackImpl::GetOutputShape(const AnfNodePtr &node, size_t i) { return GetOutputInferShape(node, i); }

ShapeVector CallbackImpl::GetInputInferShape(const AnfNodePtr &node, size_t i) {
  KernelWithIndex kernel_with_index = AnfUtils::VisitKernel(node->cast<CNodePtr>()->input(i + 1), 0);
  return GetOutputInferShape(kernel_with_index.first, kernel_with_index.second);
}

ShapeVector CallbackImpl::GetOutputInferShape(const AnfNodePtr &node, size_t i) {
  MS_EXCEPTION_IF_NULL(node);
  auto base_shape = node->Shape();
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::Shape>()) {
    if (i == 0) {
      return base_shape->cast<abstract::ShapePtr>()->shape();
    }
    MS_LOG(EXCEPTION) << "The node " << node->DebugString() << " is a single output node but got index [" << i << "]";
  } else if (base_shape->isa<abstract::TupleShape>()) {
    auto tuple_shape = base_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    if (i >= tuple_shape->size()) {
      MS_LOG(EXCEPTION) << "Output index " << i << " is larger than output number " << tuple_shape->size()
                        << " in node " << node->DebugString();
    }
    auto b_shp = (*tuple_shape)[i];
    if (b_shp->isa<abstract::Shape>()) {
      return b_shp->cast<abstract::ShapePtr>()->shape();
    } else if (b_shp->isa<abstract::NoShape>()) {
      return ShapeVector();
    } else {
      MS_LOG(EXCEPTION) << "The output type of ApplyKernel index:" << i
                        << " should be a NoShape , ArrayShape or a TupleShape, but it is " << base_shape->ToString()
                        << " node :" << node->DebugString();
    }
  } else if (base_shape->isa<abstract::NoShape>()) {
    return ShapeVector();
  }
  MS_LOG(EXCEPTION) << "The output type of ApplyKernel should be a NoShape , ArrayShape or a TupleShape, but it is "
                    << base_shape->ToString() << " node : " << node->DebugString();
}

TypeId CallbackImpl::GetInputType(const AnfNodePtr &node, size_t i) { return GetInputInferType(node, i); }

TypeId CallbackImpl::GetOutputType(const AnfNodePtr &node, size_t i) { return GetOutputInferType(node, i); }

TypeId CallbackImpl::GetInputInferType(const AnfNodePtr &node, size_t i) {
  KernelWithIndex kernel_with_index = AnfUtils::VisitKernel(node->cast<CNodePtr>()->input(i + 1), 0);
  return GetOutputInferType(kernel_with_index.first, kernel_with_index.second);
}

TypeId CallbackImpl::GetOutputInferType(const AnfNodePtr &node, size_t i) {
  MS_EXCEPTION_IF_NULL(node);
  TypePtr type_ptr = node->Type();
  MS_EXCEPTION_IF_NULL(type_ptr);
  if (type_ptr->isa<Tuple>()) {
    auto tuple_ptr = type_ptr->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_ptr);
    if (i >= tuple_ptr->size()) {
      MS_LOG(EXCEPTION) << "Output index " << i << " must be less than output number " << tuple_ptr->size()
                        << " in node " << node->DebugString();
    }
    type_ptr = (*tuple_ptr)[i];
    MS_EXCEPTION_IF_NULL(type_ptr);
  }
  if (type_ptr->isa<TensorType>()) {
    auto tensor_ptr = type_ptr->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    TypePtr elem = tensor_ptr->element();
    MS_EXCEPTION_IF_NULL(elem);
    return elem->type_id();
  }
  return type_ptr->type_id();
}

std::string GetDefaultFormat() { return kOpFormat_NCHW; }

std::string CallbackImpl::GetInputFormat(const AnfNodePtr &node, size_t i) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_with_index = AnfUtils::VisitKernel(cnode->input(i + 1), 0);
  return GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
}

std::string CallbackImpl::GetOutputFormat(const AnfNodePtr &node, size_t i) {
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!cnode->HasAttr(kOutputsFormat)) {
      MS_LOG(EXCEPTION) << "Unknown format for node " << node->fullname_with_scope();
    }
    auto vec = GetValue<std::vector<std::string>>(cnode->GetAttr(kOutputsFormat));
    if (i >= vec.size()) {
      MS_LOG(EXCEPTION) << "Index " << i << " is out of the range of outputs_format vector "
                        << cnode->GetAttr(kOutputsFormat)->ToString() << ". node is " << node->fullname_with_scope();
    }
    return vec[i];
  } else if (node->isa<Parameter>()) {
    if (params_format_.count(node) == 0) {
      MS_LOG(EXCEPTION) << "Unknown format for parameter " << node->cast<ParameterPtr>()->name();
    }
    return params_format_[node];
  } else {
    return GetDefaultFormat();
  }
}

std::string CallbackImpl::GetProcessor(const AnfNodePtr &node) { return "cpu"; }

std::string CallbackImpl::GetTargetFromContext() { return "CPU"; }

void CallbackImpl::SetGraphKernelNodeKernelInfo(const AnfNodePtr &node) {
  std::vector<std::string> graph_output_format;
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto fg = GetCNodeFuncGraph(node);
  MS_EXCEPTION_IF_NULL(fg);
  AnfNodePtrList outputs;
  if (IsPrimitiveCNode(fg->output(), prim::kPrimMakeTuple)) {
    auto fg_output = fg->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(fg_output);
    outputs.assign(fg_output->inputs().begin() + 1, fg_output->inputs().end());
  } else {
    outputs.push_back(fg->output());
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto kernel_with_index = AnfUtils::VisitKernel(outputs[i], 0);
    graph_output_format.push_back(GetOutputFormat(kernel_with_index.first, kernel_with_index.second));
  }
  cnode->AddAttr(kOutputsFormat, MakeValue(graph_output_format));

  auto inner_fg = GetCNodeFuncGraph(cnode);
  MS_EXCEPTION_IF_NULL(inner_fg);
  for (size_t i = 1; i < cnode->size(); ++i) {
    SaveParameterFormat(inner_fg->parameters()[i - 1], GetInputFormat(node, i - 1));
  }
}

void CallbackImpl::SetBasicNodeKernelInfo(const AnfNodePtr &node, const std::vector<inner::NodeBase> &outputs_info) {
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    std::vector<std::string> output_formats;
    for (size_t i = 0; i < outputs_info.size(); ++i) {
      output_formats.push_back(outputs_info[i].format);
    }
    cnode->AddAttr(kOutputsFormat, MakeValue(output_formats));
  } else if (node->isa<Parameter>()) {
    SaveParameterFormat(node, outputs_info[0].format);
  }
}

void CallbackImpl::SaveParameterFormat(const AnfNodePtr &node, const std::string &format) {
  params_format_[node] = format;
}

void CallbackImpl::SetEmptyKernelInfo(const AnfNodePtr &node) {}

void CallbackImpl::ResetKernelInfo(const AnfNodePtr &node) {}
}  // namespace mindspore::graphkernel
