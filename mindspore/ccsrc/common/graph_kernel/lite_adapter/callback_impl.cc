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

#include "common/graph_kernel/lite_adapter/callback_impl.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>
#include "base/core_ops.h"
#include "ir/dtype.h"
#include "utils/anf_utils.h"
#include "utils/utils.h"

namespace mindspore::graphkernel {
// register the callback object
GRAPH_KERNEL_CALLBACK_REGISTER(CallbackImpl);

KernelWithIndex GetPrevNodeOutput(const AnfNodePtr &anf_node, size_t i) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (IsPrimitiveCNode(anf_node, prim::kPrimTupleGetItem)) {
    return AnfUtils::VisitKernel(anf_node, 0);
  }
  auto node = anf_node->cast<CNodePtr>();
  auto input_node = node->input(i + 1);
  MS_EXCEPTION_IF_NULL(input_node);
  return AnfUtils::VisitKernel(input_node, 0);
}

ShapeVector CallbackImpl::GetInputShape(const AnfNodePtr &node, size_t i) { return GetInputInferShape(node, i); }

ShapeVector CallbackImpl::GetOutputShape(const AnfNodePtr &node, size_t i) { return GetOutputInferShape(node, i); }

ShapeVector CallbackImpl::GetInputInferShape(const AnfNodePtr &node, size_t i) {
  KernelWithIndex kernel_with_index = GetPrevNodeOutput(node, i);
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
  KernelWithIndex kernel_with_index = GetPrevNodeOutput(node, i);
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

std::string CallbackImpl::GetInputFormat(const AnfNodePtr &node, size_t i) { return kOpFormat_DEFAULT; }

std::string CallbackImpl::GetOutputFormat(const AnfNodePtr &node, size_t i) { return kOpFormat_DEFAULT; }

std::string CallbackImpl::GetProcessor(const AnfNodePtr &node) { return "cpu"; }

std::string CallbackImpl::GetTargetFromContext() { return "CPU"; }

void CallbackImpl::SetGraphKernelNodeKernelInfo(const AnfNodePtr &node) {}

void CallbackImpl::SetBasicNodeKernelInfo(const AnfNodePtr &node, const std::vector<inner::NodeBase> &outputs_info) {}

void CallbackImpl::SetEmptyKernelInfo(const AnfNodePtr &node) {}

void CallbackImpl::ResetKernelInfo(const AnfNodePtr &node) {}
}  // namespace mindspore::graphkernel
