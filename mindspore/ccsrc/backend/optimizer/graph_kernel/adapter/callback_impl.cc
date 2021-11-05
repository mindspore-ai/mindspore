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

#include "backend/optimizer/graph_kernel/adapter/callback_impl.h"

#include <algorithm>
#include <string>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore::graphkernel {
// register the callback object
GRAPH_KERNEL_CALLBACK_REGISTER(CallbackImpl);

ShapeVector CallbackImpl::GetInputShape(const AnfNodePtr &node, size_t i) {
  auto vec = AnfAlgo::GetInputDeviceShape(node, i);
  ShapeVector ret;
  std::transform(vec.begin(), vec.end(), std::back_inserter(ret), SizeToLong);
  return ret;
}

ShapeVector CallbackImpl::GetOutputShape(const AnfNodePtr &node, size_t i) {
  auto vec = AnfAlgo::GetOutputDeviceShape(node, i);
  ShapeVector ret;
  std::transform(vec.begin(), vec.end(), std::back_inserter(ret), SizeToLong);
  return ret;
}

ShapeVector CallbackImpl::GetInputInferShape(const AnfNodePtr &node, size_t i) {
  auto vec = AnfAlgo::GetPrevNodeOutputInferShape(node, i);
  ShapeVector ret;
  std::transform(vec.begin(), vec.end(), std::back_inserter(ret), SizeToLong);
  return ret;
}

ShapeVector CallbackImpl::GetOutputInferShape(const AnfNodePtr &node, size_t i) {
  auto vec = AnfAlgo::GetOutputInferShape(node, i);
  ShapeVector ret;
  std::transform(vec.begin(), vec.end(), std::back_inserter(ret), SizeToLong);
  return ret;
}

TypeId CallbackImpl::GetInputType(const AnfNodePtr &node, size_t i) { return AnfAlgo::GetInputDeviceDataType(node, i); }

TypeId CallbackImpl::GetOutputType(const AnfNodePtr &node, size_t i) {
  return AnfAlgo::GetOutputDeviceDataType(node, i);
}

TypeId CallbackImpl::GetInputInferType(const AnfNodePtr &node, size_t i) {
  return AnfAlgo::GetPrevNodeOutputInferDataType(node, i);
}

TypeId CallbackImpl::GetOutputInferType(const AnfNodePtr &node, size_t i) {
  return AnfAlgo::GetOutputInferDataType(node, i);
}

std::string CallbackImpl::GetInputFormat(const AnfNodePtr &node, size_t i) { return AnfAlgo::GetInputFormat(node, i); }

std::string CallbackImpl::GetOutputFormat(const AnfNodePtr &node, size_t i) {
  return AnfAlgo::GetOutputFormat(node, i);
}

std::string CallbackImpl::GetProcessor(const AnfNodePtr &node) { return kernel::GetProcessorStr(node); }

std::string CallbackImpl::GetProcessorFromContext() { return kernel::GetStrProcessorFromContext(); }
}  // namespace mindspore::graphkernel
