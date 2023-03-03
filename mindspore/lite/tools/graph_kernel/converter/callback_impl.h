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

#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_CALLBACK_IMPL_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_CALLBACK_IMPL_H_
#include <string>
#include <vector>
#include <utility>
#include "utils/hash_map.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"

namespace mindspore::graphkernel {
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;

// TODO(dayschan): add this function to callback class.
// Get default format for format flexible nodes.
std::string GetDefaultFormat();

class CallbackImpl : public Callback {
 public:
  ShapeVector GetInputInferShape(const AnfNodePtr &node, size_t i) override;
  ShapeVector GetOutputInferShape(const AnfNodePtr &node, size_t i) override;
  ShapeVector GetInputShape(const AnfNodePtr &node, size_t i) override;
  ShapeVector GetOutputShape(const AnfNodePtr &node, size_t i) override;
  TypeId GetInputType(const AnfNodePtr &node, size_t i) override;
  TypeId GetOutputType(const AnfNodePtr &node, size_t i) override;
  TypeId GetInputInferType(const AnfNodePtr &node, size_t i) override;
  TypeId GetOutputInferType(const AnfNodePtr &node, size_t i) override;
  std::string GetInputFormat(const AnfNodePtr &node, size_t i) override;
  std::string GetOutputFormat(const AnfNodePtr &node, size_t i) override;
  std::string GetProcessor(const AnfNodePtr &node) override;
  std::string GetTargetFromContext() override;
  void SetGraphKernelNodeKernelInfo(const AnfNodePtr &node) override;
  void SetBasicNodeKernelInfo(const AnfNodePtr &node, const std::vector<inner::NodeBase> &outputs_info) override;
  void SetEmptyKernelInfo(const AnfNodePtr &node) override;
  void ResetKernelInfo(const AnfNodePtr &node) override;

 protected:
  void SaveParameterFormat(const AnfNodePtr &node, const std::string &format);
  mindspore::HashMap<AnfNodePtr, std::string> params_format_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_CALLBACK_IMPL_H_
