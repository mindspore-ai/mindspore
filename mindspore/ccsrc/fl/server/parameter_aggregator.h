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

#ifndef MINDSPORE_CCSRC_FL_SERVER_PARAMETER_AGGREGATOR_H_
#define MINDSPORE_CCSRC_FL_SERVER_PARAMETER_AGGREGATOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "fl/server/common.h"
#include "fl/server/memory_register.h"
#include "fl/server/kernel/aggregation_kernel_factory.h"
#include "fl/server/kernel/optimizer_kernel_factory.h"

namespace mindspore {
namespace fl {
namespace server {
// Encapsulate the parameters for a kernel into a struct to make it convenient for ParameterAggregator to launch server
// kernels.
typedef struct {
  std::vector<AddressPtr> inputs;
  std::vector<AddressPtr> workspace;
  std::vector<AddressPtr> outputs;
} KernelParams;

// ParameterAggregator includes methods for aggregating gradients and optimizing weights(launching aggregation and
// optimizer kernels), getting weights, etc. It's not thread-safe, which means the caller must acquire lock before
// calling ParameterAggregator methods concurrently.

// Each ParameterAggregator is corresponding to one weight for now.

// ParameterAggregator is stateful because the process of aggregation and optimizing could be stateful.
// For example, the finite-state machine for the ParameterAggregator in parameter server training mode is below:
// Initial->Aggregating->Aggregation done->Optimizing->Optimizing done->Pulling->Pull done->Initial.
class ParameterAggregator {
 public:
  ParameterAggregator()
      : server_mode_(ServerMode::PARAMETER_SERVER),
        required_push_count_(0),
        required_pull_count_(0),
        current_pull_count_(0),
        aggregation_done_(false),
        optimizing_done_(false),
        pulling_done_(true),
        memory_register_(nullptr),
        requires_aggr_(true) {}
  ~ParameterAggregator() = default;

  // Initialize ParameterAggregator with a cnode. This cnode is normally a optimizer kernel for now.
  // The parameter threshold_count helps ParameterAggregator to judge the current status if it's stateful.
  bool Init(const CNodePtr &cnode, size_t threshold_count = 0);

  // Reinitialize the parameter aggregator after scaling operations are done.
  bool ReInitForScaling();

  // After hyper-parameters are updated, some parameter aggregators should be reinitialized.
  bool ReInitForUpdatingHyperParams(size_t aggr_threshold);

  // Update old data stored in ParameterAggregator with new data.
  // The data could have many meanings: weights, gradients, learning_rate, momentum, etc.
  bool UpdateData(const std::map<std::string, Address> &new_data);

  // Launch aggregators/optimizers of this ParameterAggregator in order.
  bool LaunchAggregators();
  bool LaunchOptimizers();

  // The implementation for primitive Pull in parameter server training mode.
  // Every call of this method will increase the count for pull by 1.
  AddressPtr Pull();

  // Different from the method Pull, this method simply returns the weight of this ParameterAggregator without causing
  // any change of status.
  AddressPtr GetWeight();

  // After aggregation/optimizing/pulling of one iteration is done, caller must reset the status to ensure the
  // correctness of the aggregation/optimizing/pulling for next iteration.
  void ResetAggregationStatus();
  void ResetOptimizingStatus();
  void ResetPullingStatus();

  // Returns the aggregation/optimizing/pulling status to the caller.
  bool IsAggregationDone() const;
  bool IsOptimizingDone() const;
  bool IsPullingDone() const;

  // Return whether this parameter requires aggragation.
  bool requires_aggr() const;

 private:
  // Initializing aggregation/optimizer kenerls based on the cnode. The reason of this is described in the file
  // kernel/kernel_factory.h.
  bool InitAggregationKernels(const CNodePtr &cnode);
  bool InitOptimizerKernels(const CNodePtr &cnode);

  // Assign memory for server kernel K(AggregationKernel/OptimizerKernel).
  // The memory assigned can be accessed by MemoryRegister. The memory could be weights, gradients, learning_rate,
  // momentum, etc.
  template <typename K>
  bool AssignMemory(const K server_kernel, const CNodePtr &cnode,
                    const ReuseKernelNodeInfo &reuse_kernel_node_inputs_info,
                    const std::shared_ptr<MemoryRegister> &memory_register);

  // Generate kernel parameters for aggregation/optimizer kernels. All the parameters is registered and stored in
  // memory_register.
  bool GenerateAggregationKernelParams(const std::shared_ptr<kernel::AggregationKernel> &aggr_kernel,
                                       const std::shared_ptr<MemoryRegister> &memory_register);
  bool GenerateOptimizerKernelParams(const std::shared_ptr<kernel::OptimizerKernel> &optim_kernel,
                                     const std::shared_ptr<MemoryRegister> &memory_register);

  // The selection of the aggregation algorithm depends on multiple factors. For example, server mode, user
  // configuration, etc.
  std::vector<std::string> SelectAggregationAlgorithm(const CNodePtr &cnode);

  // Judge whether the parameter needs to be aggregated.
  bool JudgeRequiresAggr(const CNodePtr &cnode);

  ServerMode server_mode_;
  size_t required_push_count_;
  size_t required_pull_count_;
  size_t current_pull_count_;

  // The status of aggregation/optimizing/pulling.
  bool aggregation_done_;
  bool optimizing_done_;
  bool pulling_done_;

  // ParameterAggregator stores all data that it needs for aggregation, optimizing, etc.
  std::shared_ptr<MemoryRegister> memory_register_;

  // Update could have multiple aggregation and optimizer server kernels.
  // Here stores multiple pairs of server kernels to parameters of their Launch function.
  std::vector<std::pair<std::shared_ptr<kernel::AggregationKernel>, KernelParams>> aggregation_kernel_parameters_;
  std::vector<std::pair<std::shared_ptr<kernel::OptimizerKernel>, KernelParams>> optimizer_kernel_parameters_;

  // Whether this parameter needs to be aggregated.
  bool requires_aggr_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_PARAMETER_AGGREGATOR_H_
