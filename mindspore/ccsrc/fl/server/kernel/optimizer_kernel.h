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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_OPTIMIZER_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_OPTIMIZER_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "fl/server/common.h"
#include "fl/server/memory_register.h"
#include "fl/server/kernel/params_info.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
using mindspore::kernel::IsSameShape;
using mindspore::kernel::USE_NESTEROV;

// OptimizerKernel is the kernel in server for weights' optimizing.
// Normally server's optimizer kernels should be inherited from CPU's optimzier kernels to reuse the implementation.
class OptimizerKernel : public CPUKernel {
 public:
  OptimizerKernel() = default;
  virtual ~OptimizerKernel() = default;

  // InitKernel and Launch methods are inherited from pure virtual function of CPUKernel so it must have implementation.
  virtual void InitKernel(const CNodePtr &kernel_node) {}
  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs) {
    return true;
  }

  // Server kernel's memory allocation method, which is different from the workflow in
  // Session(GPUSession/CPUSession/AscendSession).
  // virtual void AssignMemory(const CNodePtr &kernel_node, std::shared_ptr<MemoryRegister> memory_register) = 0;

  // Setter and getter of kernels parameters information.
  void set_params_info(const ParamsInfo &params_info) { params_info_ = params_info; }
  const std::vector<std::string> &input_names() { return params_info_.inputs_names(); }
  const std::vector<std::string> &workspace_names() { return params_info_.workspace_names(); }
  const std::vector<std::string> &output_names() { return params_info_.outputs_names(); }

  // Returns information about whether some inputs should reuse kernel node inputs memory.
  const ReuseKernelNodeInfo &reuse_kernel_node_inputs_info() { return reuse_kernel_node_inputs_info_; }

 protected:
  virtual void GenerateReuseKernelNodeInfo() = 0;

  void InitServerKernelInputOutputSize(const CNodePtr &kernel_node) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    size_t type_size = sizeof(float);
    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      std::vector<size_t> shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, input_index);
      size_t tensor_size =
        shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
      input_size_list_.emplace_back(tensor_size);
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    for (size_t output_index = 0; output_index < output_num; ++output_index) {
      std::vector<size_t> shape = AnfAlgo::GetOutputInferShape(kernel_node, output_index);
      size_t tensor_size =
        shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
      output_size_list_.emplace_back(tensor_size);
    }
  }

  // Parameters information used for kernel register, memory assignment, etc.
  ParamsInfo params_info_;

  // Information about server kernel reusing kernel node inputs memory from the front end.
  // Key refers to the server kernel's input index. Value refers to the kernel node's input index.
  ReuseKernelNodeInfo reuse_kernel_node_inputs_info_;
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_OPTIMIZER_KERNEL_H_
