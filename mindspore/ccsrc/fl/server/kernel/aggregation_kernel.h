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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_AGGREGATION_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_AGGREGATION_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "fl/server/common.h"
#include "fl/server/memory_register.h"
#include "fl/server/kernel/params_info.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
// AggregationKernel is the kernel for weight, grad or other kinds of parameters' aggregation.
// For example, dense gradients accumulation, federated average, etc.
// Normally the aggregation process in AggregationKernel is like a finite-state machine:
// Initial->Aggregating->Aggregation done->Initial.
class AggregationKernel : public CPUKernel {
 public:
  AggregationKernel() : name_(""), done_(false), done_count_(0), accum_count_(0) {}
  virtual ~AggregationKernel() = default;

  // InitKernel and Launch methods are inherited from pure virtual function of CPUKernel so it must have implementation.
  virtual void InitKernel(const CNodePtr &kernel_node) {}
  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs) {
    return true;
  }

  // Server kernel's memory allocation method, which is different from the workflow in
  // Session(GPUSession/CPUSession/AscendSession).
  // virtual void AssignMemory(const CNodePtr &kernel_node, std::shared_ptr<MemoryRegister> memory_register) = 0;

  // Set the cumulative count this aggregation kernel needs before aggregation is done.
  void set_done_count(size_t count) { done_count_ = count; }

  // So we use Reset to set the finite-state machine state to Initial after considering this round of aggregation is
  // done.
  virtual void Reset() = 0;

  virtual bool IsAggregationDone() = 0;

  // Some kernels should know the inputs/workspace/outputs addresses at initializing phase. For example, FedAvgKernel.
  virtual void SetParameterAddress(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs) {
    return;
  }

  // Reinitialize aggregation kernel after scaling operations are done.
  virtual bool ReInitForScaling() { return true; }

  virtual bool ReInitForUpdatingHyperParams(size_t) { return true; }

  // Setter and getter of kernels parameters information.
  void set_params_info(const ParamsInfo &params_info) { params_info_ = params_info; }
  const std::vector<std::string> &input_names() { return params_info_.inputs_names(); }
  const std::vector<std::string> &workspace_names() { return params_info_.workspace_names(); }
  const std::vector<std::string> &output_names() { return params_info_.outputs_names(); }

  // Returns information about whether some inputs should reuse kernel node inputs memory.
  const ReuseKernelNodeInfo &reuse_kernel_node_inputs_info() { return reuse_kernel_node_inputs_info_; }

 protected:
  virtual void GenerateReuseKernelNodeInfo() = 0;
  // Aggregation kernel's name which is set by kernel register function.
  std::string name_;

  // The aggregation is considered done after done_count_ times of accumulation.
  bool done_;

  // Cumulative count this aggregation kernel needs before aggregation is done.
  size_t done_count_;

  // Current cumulative count.
  size_t accum_count_;

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
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_AGGREGATION_KERNEL_H_
