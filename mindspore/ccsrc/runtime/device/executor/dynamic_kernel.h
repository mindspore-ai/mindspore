/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_EXECUTOR_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_EXECUTOR_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include "ir/anf.h"
#include "ir/tensor.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace device {
class DynamicKernel {
 public:
  DynamicKernel(void *stream, const CNodePtr &cnode_ptr)
      : stream_(stream),
        cnode_ptr_(cnode_ptr),
        is_dynamic_shape_(false),
        is_input_dynamic_shape_(false),
        is_output_dynamic_shape_(false) {}
  virtual ~DynamicKernel() = default;
  virtual void InferShape();
  virtual void UpdateArgs() = 0;
  virtual void Execute() = 0;
  virtual void PostExecute() = 0;
  bool is_dynamic_shape() const { return is_dynamic_shape_; }
  bool is_input_dynamic_shape() const { return is_input_dynamic_shape_; }
  bool is_output_dynamic_shape() const { return is_output_dynamic_shape_; }
  bool have_depends() const { return !depend_list_.empty(); }
  virtual void Initialize();
  std::string GetKernelName() { return cnode_ptr_.lock()->fullname_with_scope(); }
  int GetKernelType();

 protected:
  void RebuildDependTensor();
  void InferShapeRecursive();
  void InferShapeForNopNode(AnfNodePtr *input_node);

  void *stream_;
  const CNodeWeakPtr cnode_ptr_;
  bool is_dynamic_shape_;
  bool is_input_dynamic_shape_;
  bool is_output_dynamic_shape_;
  std::vector<uint32_t> depend_list_;
  std::map<uint32_t, tensor::TensorPtr> depend_tensor_map_;
};
using DynamicKernelPtr = std::shared_ptr<DynamicKernel>;
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_EXECUTOR_H_
