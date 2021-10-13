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

#include "fl/server/parameter_aggregator.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

namespace mindspore {
namespace fl {
namespace server {
bool ParameterAggregator::Init(const CNodePtr &cnode, size_t threshold_count) {
  MS_EXCEPTION_IF_NULL(cnode);
  memory_register_ = std::make_shared<MemoryRegister>();
  MS_EXCEPTION_IF_NULL(memory_register_);

  required_push_count_ = threshold_count;
  // The required_pull_count_ is the count for Pull, which should be the same as required_push_count_.
  // required_pull_count_ normally used in parameter server training mode.
  required_pull_count_ = threshold_count;

  MS_LOG(DEBUG) << "Start initializing kernels for " << AnfAlgo::GetCNodeName(cnode);
  if (!InitAggregationKernels(cnode)) {
    MS_LOG(EXCEPTION) << "Initializing aggregation kernels failed.";
    return false;
  }
  if (!InitOptimizerKernels(cnode)) {
    MS_LOG(EXCEPTION) << "Initializing optimizer kernels failed.";
    return false;
  }
  return true;
}

bool ParameterAggregator::ReInitForScaling() {
  auto result = std::find_if(aggregation_kernel_parameters_.begin(), aggregation_kernel_parameters_.end(),
                             [](auto aggregation_kernel) {
                               MS_ERROR_IF_NULL_W_RET_VAL(aggregation_kernel.first, true);
                               return !aggregation_kernel.first->ReInitForScaling();
                             });
  if (result != aggregation_kernel_parameters_.end()) {
    MS_LOG(ERROR) << "Reinitializing aggregation kernel after scaling failed";
    return false;
  }
  return true;
}

bool ParameterAggregator::ReInitForUpdatingHyperParams(size_t aggr_threshold) {
  required_push_count_ = aggr_threshold;
  required_pull_count_ = aggr_threshold;
  auto result = std::find_if(aggregation_kernel_parameters_.begin(), aggregation_kernel_parameters_.end(),
                             [aggr_threshold](auto aggregation_kernel) {
                               MS_ERROR_IF_NULL_W_RET_VAL(aggregation_kernel.first, true);
                               return !aggregation_kernel.first->ReInitForUpdatingHyperParams(aggr_threshold);
                             });
  if (result != aggregation_kernel_parameters_.end()) {
    MS_LOG(ERROR) << "Reinitializing aggregation kernel after scaling failed";
    return false;
  }
  return true;
}

bool ParameterAggregator::UpdateData(const std::map<std::string, Address> &new_data) {
  std::map<std::string, AddressPtr> &name_to_addr = memory_register_->addresses();
  for (const auto &data : new_data) {
    const std::string &name = data.first;
    if (name_to_addr.count(name) == 0) {
      continue;
    }

    MS_ERROR_IF_NULL_W_RET_VAL(name_to_addr[name], false);
    MS_ERROR_IF_NULL_W_RET_VAL(name_to_addr[name]->addr, false);
    MS_ERROR_IF_NULL_W_RET_VAL(data.second.addr, false);
    MS_LOG(DEBUG) << "Update data for " << name << ". Destination size: " << name_to_addr[name]->size
                  << ". Source size: " << data.second.size;
    int ret = memcpy_s(name_to_addr[name]->addr, name_to_addr[name]->size, data.second.addr, data.second.size);
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return false;
    }
  }
  return true;
}

bool ParameterAggregator::LaunchAggregators() {
  for (auto &aggregator_with_params : aggregation_kernel_parameters_) {
    KernelParams &params = aggregator_with_params.second;
    std::shared_ptr<kernel::AggregationKernel> aggr_kernel = aggregator_with_params.first;
    MS_ERROR_IF_NULL_W_RET_VAL(aggr_kernel, false);
    bool ret = aggr_kernel->Launch(params.inputs, params.workspace, params.outputs);
    if (!ret) {
      MS_LOG(ERROR) << "Launching aggregation kernel " << typeid(aggr_kernel.get()).name() << " failed.";
      return false;
    }
  }
  return true;
}

bool ParameterAggregator::LaunchOptimizers() {
  for (auto &optimizer_with_params : optimizer_kernel_parameters_) {
    KernelParams &params = optimizer_with_params.second;
    std::shared_ptr<kernel::OptimizerKernel> optimizer_kernel = optimizer_with_params.first;
    MS_ERROR_IF_NULL_W_RET_VAL(optimizer_kernel, false);
    bool ret = optimizer_kernel->Launch(params.inputs, params.workspace, params.outputs);
    if (!ret) {
      MS_LOG(ERROR) << "Launching optimizer kernel " << typeid(optimizer_kernel.get()).name() << " failed.";
      continue;
    }
  }
  // As long as all the optimizer kernels are launched, consider optimizing for this ParameterAggregator as done.
  optimizing_done_ = true;
  return true;
}

AddressPtr ParameterAggregator::Pull() {
  if (memory_register_ == nullptr) {
    MS_LOG(ERROR)
      << "The memory register of ParameterAggregator is nullptr. Please initialize ParameterAggregator first.";
    return nullptr;
  }

  current_pull_count_++;
  if (current_pull_count_ == required_pull_count_) {
    pulling_done_ = true;
  }
  MS_LOG(DEBUG) << "The " << current_pull_count_ << " time of Pull. Pulling done status: " << pulling_done_;

  std::map<std::string, AddressPtr> &name_to_addr = memory_register_->addresses();
  return name_to_addr["weight"];
}

AddressPtr ParameterAggregator::GetWeight() {
  if (memory_register_ == nullptr) {
    MS_LOG(ERROR)
      << "The memory register of ParameterAggregator is nullptr. Please initialize ParameterAggregator first.";
    return nullptr;
  }
  std::map<std::string, AddressPtr> &name_to_addr = memory_register_->addresses();
  return name_to_addr["weight"];
}

void ParameterAggregator::ResetAggregationStatus() {
  for (auto &aggregator_with_params : aggregation_kernel_parameters_) {
    std::shared_ptr<kernel::AggregationKernel> aggr_kernel = aggregator_with_params.first;
    if (aggr_kernel == nullptr) {
      MS_LOG(ERROR) << "The aggregation kernel is nullptr.";
      continue;
    }
    aggr_kernel->Reset();
  }
  return;
}

void ParameterAggregator::ResetOptimizingStatus() { optimizing_done_ = false; }

void ParameterAggregator::ResetPullingStatus() {
  pulling_done_ = false;
  current_pull_count_ = 0;
}

bool ParameterAggregator::IsAggregationDone() const {
  // Only consider aggregation done after each aggregation kernel is done.
  for (auto &aggregator_with_params : aggregation_kernel_parameters_) {
    std::shared_ptr<kernel::AggregationKernel> aggr_kernel = aggregator_with_params.first;
    MS_ERROR_IF_NULL_W_RET_VAL(aggr_kernel, false);
    if (!aggr_kernel->IsAggregationDone()) {
      return false;
    }
  }
  return true;
}

bool ParameterAggregator::IsOptimizingDone() const { return optimizing_done_; }

bool ParameterAggregator::IsPullingDone() const { return pulling_done_; }

bool ParameterAggregator::requires_aggr() const { return requires_aggr_; }

bool ParameterAggregator::InitAggregationKernels(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!JudgeRequiresAggr(cnode)) {
    MS_LOG(WARNING) << "Aggregation for weight for kernel " << AnfAlgo::GetCNodeName(cnode) << " is not required.";
  }

  std::vector<std::string> aggr_kernel_names = SelectAggregationAlgorithm(cnode);
  for (const std::string &name : aggr_kernel_names) {
    auto aggr_kernel = kernel::AggregationKernelFactory::GetInstance().Create(name, cnode);
    if (aggr_kernel == nullptr) {
      MS_LOG(EXCEPTION) << "Fail to create aggregation kernel " << name << " for " << AnfAlgo::GetCNodeName(cnode);
      return false;
    }

    // set_done_count must be called before InitKernel because InitKernel may use this count.
    aggr_kernel->set_done_count(required_push_count_);
    aggr_kernel->InitKernel(cnode);

    const ReuseKernelNodeInfo &reuse_kernel_node_inputs_info = aggr_kernel->reuse_kernel_node_inputs_info();
    if (!AssignMemory(aggr_kernel, cnode, reuse_kernel_node_inputs_info, memory_register_)) {
      MS_LOG(EXCEPTION) << "Assigning memory for kernel " << name << " failed.";
      return false;
    }

    if (!GenerateAggregationKernelParams(aggr_kernel, memory_register_)) {
      MS_LOG(EXCEPTION) << "Generating aggregation kernel parameters for " << name << " failed.";
      return false;
    }
  }
  return true;
}

bool ParameterAggregator::InitOptimizerKernels(const CNodePtr &cnode) {
  if (ps::PSContext::instance()->server_mode() == ps::kServerModeFL ||
      ps::PSContext::instance()->server_mode() == ps::kServerModeHybrid) {
    MS_LOG(DEBUG) << "Federated learning mode doesn't need optimizer kernel.";
    return true;
  }
  MS_EXCEPTION_IF_NULL(cnode);
  const std::string &name = AnfAlgo::GetCNodeName(cnode);
  auto optimizer_kernel = kernel::OptimizerKernelFactory::GetInstance().Create(name, cnode);
  if (optimizer_kernel == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to create optimizer kernel for " << name;
    return false;
  }

  optimizer_kernel->InitKernel(cnode);

  const ReuseKernelNodeInfo &reuse_kernel_node_inputs_info = optimizer_kernel->reuse_kernel_node_inputs_info();
  if (!AssignMemory(optimizer_kernel, cnode, reuse_kernel_node_inputs_info, memory_register_)) {
    MS_LOG(EXCEPTION) << "Assigning memory for kernel " << name << " failed.";
    return false;
  }

  if (!GenerateOptimizerKernelParams(optimizer_kernel, memory_register_)) {
    MS_LOG(ERROR) << "Generating optimizer kernel parameters failed.";
    return false;
  }
  return true;
}

template <typename K>
bool ParameterAggregator::AssignMemory(const K server_kernel, const CNodePtr &cnode,
                                       const ReuseKernelNodeInfo &reuse_kernel_node_inputs_info,
                                       const std::shared_ptr<MemoryRegister> &memory_register) {
  MS_EXCEPTION_IF_NULL(server_kernel);
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(memory_register);

  const std::vector<std::string> &input_names = server_kernel->input_names();
  const std::vector<size_t> &input_size_list = server_kernel->GetInputSizeList();
  if (input_names.size() != input_size_list.size()) {
    MS_LOG(EXCEPTION) << "Server kernel " << typeid(server_kernel.get()).name()
                      << " input number is not matched: input_names size is " << input_names.size()
                      << ", input_size_list size is " << input_size_list.size();
    return false;
  }

  if (reuse_kernel_node_inputs_info.size() > input_names.size()) {
    MS_LOG(EXCEPTION) << "The reuse kernel node information number is invalid: got "
                      << reuse_kernel_node_inputs_info.size() << ", but input_names size is " << input_names.size();
    return false;
  }

  for (size_t i = 0; i < input_names.size(); i++) {
    const std::string &name = input_names[i];
    if (memory_register->addresses().count(name) != 0) {
      MS_LOG(DEBUG) << "The memory for " << name << " is already assigned.";
      continue;
    }
    if (reuse_kernel_node_inputs_info.count(name) != 0) {
      // Reusing memory of the kernel node means the memory of the input is already assigned by the front end, which
      // is to say, the input node is a parameter node.
      size_t index = reuse_kernel_node_inputs_info.at(name);
      MS_LOG(INFO) << "Try to reuse memory of kernel node " << AnfAlgo::GetCNodeName(cnode) << " for parameter " << name
                   << ", kernel node index " << index;
      AddressPtr input_addr = GenerateParameterNodeAddrPtr(cnode, index);
      MS_EXCEPTION_IF_NULL(input_addr);
      memory_register->RegisterAddressPtr(name, input_addr);
    } else {
      MS_LOG(INFO) << "Assign new memory for " << name;
      auto input_addr = std::make_unique<char[]>(input_size_list[i]);
      MS_EXCEPTION_IF_NULL(input_addr);
      memory_register->RegisterArray(name, &input_addr, input_size_list[i]);
    }
  }
  return true;
}

bool ParameterAggregator::GenerateAggregationKernelParams(const std::shared_ptr<kernel::AggregationKernel> &aggr_kernel,
                                                          const std::shared_ptr<MemoryRegister> &memory_register) {
  MS_ERROR_IF_NULL_W_RET_VAL(aggr_kernel, false);
  MS_ERROR_IF_NULL_W_RET_VAL(memory_register, false);
  KernelParams aggr_params = {};

  const std::vector<std::string> &input_names = aggr_kernel->input_names();
  (void)std::transform(input_names.begin(), input_names.end(), std::back_inserter(aggr_params.inputs),
                       [&](const std::string &name) { return memory_register->addresses()[name]; });

  const std::vector<std::string> &workspace_names = aggr_kernel->workspace_names();
  (void)std::transform(workspace_names.begin(), workspace_names.end(), std::back_inserter(aggr_params.workspace),
                       [&](const std::string &name) { return memory_register->addresses()[name]; });

  const std::vector<std::string> &output_names = aggr_kernel->output_names();
  (void)std::transform(output_names.begin(), output_names.end(), std::back_inserter(aggr_params.outputs),
                       [&](const std::string &name) { return memory_register->addresses()[name]; });

  aggr_kernel->SetParameterAddress(aggr_params.inputs, aggr_params.workspace, aggr_params.outputs);
  aggregation_kernel_parameters_.push_back(std::make_pair(aggr_kernel, aggr_params));
  return true;
}

bool ParameterAggregator::GenerateOptimizerKernelParams(
  const std::shared_ptr<kernel::OptimizerKernel> &optimizer_kernel,
  const std::shared_ptr<MemoryRegister> &memory_register) {
  MS_ERROR_IF_NULL_W_RET_VAL(optimizer_kernel, false);
  MS_ERROR_IF_NULL_W_RET_VAL(memory_register, false);
  KernelParams optimizer_params = {};

  const std::vector<std::string> &input_names = optimizer_kernel->input_names();
  (void)std::transform(input_names.begin(), input_names.end(), std::back_inserter(optimizer_params.inputs),
                       [&](const std::string &name) { return memory_register->addresses()[name]; });

  const std::vector<std::string> &workspace_names = optimizer_kernel->workspace_names();
  (void)std::transform(workspace_names.begin(), workspace_names.end(), std::back_inserter(optimizer_params.workspace),
                       [&](const std::string &name) { return memory_register->addresses()[name]; });

  const std::vector<std::string> &output_names = optimizer_kernel->output_names();
  (void)std::transform(output_names.begin(), output_names.end(), std::back_inserter(optimizer_params.outputs),
                       [&](const std::string &name) { return memory_register->addresses()[name]; });

  optimizer_kernel_parameters_.push_back(std::make_pair(optimizer_kernel, optimizer_params));
  return true;
}

std::vector<std::string> ParameterAggregator::SelectAggregationAlgorithm(const CNodePtr &) {
  std::vector<std::string> aggregation_algorithm = {};
  if (ps::PSContext::instance()->server_mode() == ps::kServerModeFL ||
      ps::PSContext::instance()->server_mode() == ps::kServerModeHybrid) {
    (void)aggregation_algorithm.emplace_back("FedAvg");
  } else if (ps::PSContext::instance()->server_mode() == ps::kServerModePS) {
    (void)aggregation_algorithm.emplace_back("DenseGradAccum");
  } else {
    MS_LOG(EXCEPTION) << "Server doesn't support mode " << ps::PSContext::instance()->server_mode();
    return aggregation_algorithm;
  }

  MS_LOG(INFO) << "Aggregation algorithm selection result: " << aggregation_algorithm;
  return aggregation_algorithm;
}

bool ParameterAggregator::JudgeRequiresAggr(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  std::string cnode_name = AnfAlgo::GetCNodeName(cnode);
  if (kNameToIdxMap.count(cnode_name) == 0 || kNameToIdxMap.at(cnode_name).count("inputs") == 0 ||
      kNameToIdxMap.at(cnode_name).at("inputs").count("weight") == 0) {
    MS_LOG(EXCEPTION) << "Can't find index info of weight for kernel " << cnode_name;
    return false;
  }
  size_t cnode_weight_idx = kNameToIdxMap.at(cnode_name).at("inputs").at("weight");
  auto weight_node = AnfAlgo::VisitKernelWithReturnType(AnfAlgo::GetInputNode(cnode, cnode_weight_idx), 0).first;
  MS_EXCEPTION_IF_NULL(weight_node);

  if (!weight_node->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << weight_node->fullname_with_scope() << " is not a parameter node.";
    return false;
  }
  auto param_info = weight_node->cast<ParameterPtr>()->param_info();
  MS_EXCEPTION_IF_NULL(param_info);
  requires_aggr_ = param_info->requires_aggr();
  return requires_aggr_;
}

template bool ParameterAggregator::AssignMemory(std::shared_ptr<kernel::OptimizerKernel> server_kernel,
                                                const CNodePtr &cnode,
                                                const ReuseKernelNodeInfo &reuse_kernel_node_inputs_info,
                                                const std::shared_ptr<MemoryRegister> &memory_register);

template bool ParameterAggregator::AssignMemory(std::shared_ptr<kernel::AggregationKernel> server_kernel,
                                                const CNodePtr &cnode,
                                                const ReuseKernelNodeInfo &reuse_kernel_node_inputs_info,
                                                const std::shared_ptr<MemoryRegister> &memory_register);
}  // namespace server
}  // namespace fl
}  // namespace mindspore
