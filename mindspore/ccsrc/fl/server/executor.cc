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

#include "fl/server/executor.h"
#include <set>
#include <memory>
#include <string>
#include <vector>

namespace mindspore {
namespace fl {
namespace server {
void Executor::Initialize(const FuncGraphPtr &func_graph, size_t aggregation_count) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (aggregation_count == 0) {
    MS_LOG(EXCEPTION) << "Server aggregation count must be greater than 0";
    return;
  }
  aggregation_count_ = aggregation_count;

  // Initialize each trainable parameter's aggregator, including memory register, aggregation algorithms and optimizers.
  bool ret = InitParamAggregator(func_graph);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Initializing parameter aggregators failed.";
    return;
  }
  initialized_ = true;
  return;
}

bool Executor::ReInitForScaling() {
  auto result = std::find_if(param_aggrs_.begin(), param_aggrs_.end(),
                             [](auto param_aggr) { return !param_aggr.second->ReInitForScaling(); });
  if (result != param_aggrs_.end()) {
    MS_LOG(ERROR) << "Reinitializing aggregator of " << result->first << " for scaling failed.";
    return false;
  }
  return true;
}

bool Executor::ReInitForUpdatingHyperParams(size_t aggr_threshold) {
  aggregation_count_ = aggr_threshold;
  auto result = std::find_if(param_aggrs_.begin(), param_aggrs_.end(), [this](auto param_aggr) {
    return !param_aggr.second->ReInitForUpdatingHyperParams(aggregation_count_);
  });
  if (result != param_aggrs_.end()) {
    MS_LOG(ERROR) << "Reinitializing aggregator of " << result->first << " for scaling failed.";
    return false;
  }
  return true;
}

bool Executor::initialized() const { return initialized_; }

bool Executor::HandlePush(const std::string &param_name, const UploadData &upload_data) {
  MS_LOG(DEBUG) << "Do Push for parameter " << param_name;
  if (param_aggrs_.count(param_name) == 0) {
    MS_LOG(WARNING) << "Parameter " << param_name << " is not registered in server.";
    return false;
  }

  std::mutex &mtx = parameter_mutex_[param_name];
  std::unique_lock<std::mutex> lock(mtx);
  auto &param_aggr = param_aggrs_[param_name];
  MS_ERROR_IF_NULL_W_RET_VAL(param_aggr, false);
  // Push operation needs to wait until the pulling process is done.
  while (!param_aggr->IsPullingDone()) {
    lock.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(kThreadSleepTime));
    lock.lock();
  }

  // 1.Update data with the uploaded data of the worker.
  if (!param_aggr->UpdateData(upload_data)) {
    MS_LOG(ERROR) << "Updating data for parameter " << param_name << " failed.";
    return false;
  }
  // 2.Launch aggregation for this trainable parameter.
  if (!param_aggr->LaunchAggregators()) {
    MS_LOG(ERROR) << "Launching aggregators for parameter " << param_name << " failed.";
    return false;
  }
  if (param_aggr->IsAggregationDone()) {
    // 3.After the aggregation is done, optimize the trainable parameter.
    if (!param_aggr->LaunchOptimizers()) {
      MS_LOG(ERROR) << "Optimizing for parameter " << param_name << " failed.";
      return false;
    }
    // 4.Reset pulling and aggregation status after optimizing is done.
    param_aggr->ResetPullingStatus();
    param_aggr->ResetAggregationStatus();
  }
  return true;
}

bool Executor::HandleModelUpdate(const std::string &param_name, const UploadData &upload_data) {
  MS_LOG(DEBUG) << "Do UpdateModel for parameter " << param_name;
  if (param_aggrs_.count(param_name) == 0) {
    // The param_name could include some other parameters like momentum, but we don't think it's invalid. So here we
    // just print a warning log and return true.
    MS_LOG(WARNING) << "Parameter " << param_name << " is not registered in server.";
    return true;
  }

  std::mutex &mtx = parameter_mutex_[param_name];
  std::unique_lock<std::mutex> lock(mtx);
  auto &param_aggr = param_aggrs_[param_name];
  MS_ERROR_IF_NULL_W_RET_VAL(param_aggr, false);
  if (!param_aggr->UpdateData(upload_data)) {
    MS_LOG(ERROR) << "Updating data for parameter " << param_name << " failed.";
    return false;
  }
  // Different from Push, UpdateModel doesn't need to checkout the aggregation status.
  if (!param_aggr->LaunchAggregators()) {
    MS_LOG(ERROR) << "Launching aggregators for parameter " << param_name << " failed.";
    return false;
  }
  return true;
}

bool Executor::HandleModelUpdateAsync(const std::map<std::string, UploadData> &feature_map) {
  std::unique_lock<std::mutex> model_lock(model_mutex_);
  for (const auto &trainable_param : feature_map) {
    const std::string &param_name = trainable_param.first;
    if (param_aggrs_.count(param_name) == 0) {
      MS_LOG(WARNING) << "Parameter " << param_name << " is not registered in server.";
      continue;
    }

    std::mutex &mtx = parameter_mutex_[param_name];
    std::unique_lock<std::mutex> lock(mtx);
    auto &param_aggr = param_aggrs_[param_name];
    MS_ERROR_IF_NULL_W_RET_VAL(param_aggr, false);
    const UploadData &upload_data = trainable_param.second;
    if (!param_aggr->UpdateData(upload_data)) {
      MS_LOG(ERROR) << "Updating data for parameter " << param_name << " failed.";
      return false;
    }
    if (!param_aggr->LaunchAggregators()) {
      MS_LOG(ERROR) << "Launching aggregators for parameter " << param_name << " failed.";
      return false;
    }
  }
  return true;
}

bool Executor::HandlePushWeight(const std::map<std::string, Address> &feature_map) {
  for (const auto &trainable_param : feature_map) {
    const std::string &param_name = trainable_param.first;
    if (param_aggrs_.count(param_name) == 0) {
      MS_LOG(WARNING) << "Weight " << param_name << " is not registered in server.";
      continue;
    }

    std::mutex &mtx = parameter_mutex_[param_name];
    std::unique_lock<std::mutex> lock(mtx);
    auto &param_aggr = param_aggrs_[param_name];
    MS_ERROR_IF_NULL_W_RET_VAL(param_aggr, false);
    AddressPtr old_weight = param_aggr->GetWeight();
    const Address &new_weight = trainable_param.second;
    MS_ERROR_IF_NULL_W_RET_VAL(old_weight, false);
    MS_ERROR_IF_NULL_W_RET_VAL(old_weight->addr, false);
    MS_ERROR_IF_NULL_W_RET_VAL(new_weight.addr, false);
    int ret = memcpy_s(old_weight->addr, old_weight->size, new_weight.addr, new_weight.size);
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return false;
    }
  }
  return true;
}

AddressPtr Executor::HandlePull(const std::string &param_name) {
  MS_LOG(INFO) << "Handle blocking pull message for parameter " << param_name;
  if (param_aggrs_.count(param_name) == 0) {
    MS_LOG(WARNING) << "Parameter " << param_name << " is not registered in server.";
    return nullptr;
  }

  std::mutex &mtx = parameter_mutex_[param_name];
  std::unique_lock<std::mutex> lock(mtx);
  auto &param_aggr = param_aggrs_[param_name];
  MS_ERROR_IF_NULL_W_RET_VAL(param_aggr, nullptr);
  // Pulling must wait until the optimizing process is done.
  while (!param_aggr->IsOptimizingDone()) {
    lock.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(kThreadSleepTime));
    lock.lock();
  }
  AddressPtr addr = param_aggr->Pull();
  // If this Pull is the last one, reset pulling and optimizing status.
  if (param_aggr->IsPullingDone()) {
    param_aggr->ResetOptimizingStatus();
  }
  return addr;
}

std::map<std::string, AddressPtr> Executor::HandlePullWeight(const std::vector<std::string> &param_names) {
  std::map<std::string, AddressPtr> weights;
  for (const auto &param_name : param_names) {
    if (param_aggrs_.count(param_name) == 0) {
      MS_LOG(ERROR) << "Parameter " << param_name << " is not registered in server.";
      return weights;
    }

    std::mutex &mtx = parameter_mutex_[param_name];
    std::unique_lock<std::mutex> lock(mtx);
    const auto &param_aggr = param_aggrs_[param_name];
    MS_ERROR_IF_NULL_W_RET_VAL(param_aggr, weights);
    AddressPtr addr = param_aggr->GetWeight();
    if (addr == nullptr) {
      MS_LOG(ERROR) << "Get weight of " << param_name << " failed: the AddressPtr is nullptr.";
      continue;
    }
    weights[param_name] = addr;
  }
  return weights;
}

bool Executor::IsAllWeightAggregationDone() { return IsWeightAggrDone(param_names_); }

bool Executor::IsWeightAggrDone(const std::vector<std::string> &param_names) {
  for (const auto &name : param_names) {
    if (param_aggrs_.count(name) == 0) {
      MS_LOG(ERROR) << "Weight " << name << " is invalid in server.";
      return false;
    }

    std::mutex &mtx = parameter_mutex_[name];
    std::unique_lock<std::mutex> lock(mtx);
    auto &param_aggr = param_aggrs_[name];
    MS_ERROR_IF_NULL_W_RET_VAL(param_aggr, false);
    if (!param_aggr->requires_aggr()) {
      continue;
    }
    if (!param_aggr->IsAggregationDone()) {
      MS_LOG(DEBUG) << "Update model for " << name << " is not done yet.";
      return false;
    }
  }
  return true;
}

void Executor::ResetAggregationStatus() {
  for (const auto &param_name : param_names_) {
    std::mutex &mtx = parameter_mutex_[param_name];
    std::unique_lock<std::mutex> lock(mtx);
    auto &param_aggr = param_aggrs_[param_name];
    MS_ERROR_IF_NULL_WO_RET_VAL(param_aggr);
    param_aggr->ResetAggregationStatus();
  }
  return;
}

std::map<std::string, AddressPtr> Executor::GetModel() {
  std::map<std::string, AddressPtr> model = {};
  for (const auto &name : param_names_) {
    std::mutex &mtx = parameter_mutex_[name];
    std::unique_lock<std::mutex> lock(mtx);
    AddressPtr addr = param_aggrs_[name]->GetWeight();
    if (addr == nullptr) {
      MS_LOG(WARNING) << "Get weight of " << name << " failed.";
      continue;
    }
    model[name] = addr;
  }
  return model;
}

const std::vector<std::string> &Executor::param_names() const { return param_names_; }

bool Executor::Unmask() {
#ifdef ENABLE_ARMOUR
  auto model = GetModel();
  return cipher_unmask_.UnMask(model);
#else
  return false;
#endif
}

void Executor::set_unmasked(bool unmasked) { unmasked_ = unmasked; }

bool Executor::unmasked() const {
  std::string encrypt_type = ps::PSContext::instance()->encrypt_type();
  if (encrypt_type == ps::kPWEncryptType) {
    return unmasked_.load();
  } else {
    // If the algorithm of pairwise encrypt is not enabled, consider_ unmasked flag as true.
    return true;
  }
}

std::string Executor::GetTrainableParamName(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  std::string cnode_name = AnfAlgo::GetCNodeName(cnode);
  if (kNameToIdxMap.count(cnode_name) == 0) {
    return "";
  }
  const OptimParamNameToIndex &index_info = kNameToIdxMap.at(cnode_name);
  size_t weight_idx = index_info.at("inputs").at(kWeight);
  AnfNodePtr weight_node = AnfAlgo::VisitKernelWithReturnType(AnfAlgo::GetInputNode(cnode, weight_idx), 0).first;
  MS_EXCEPTION_IF_NULL(weight_node);
  if (!weight_node->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << weight_idx << " input of " << cnode_name << " is not a Parameter.";
    return "";
  }
  return weight_node->fullname_with_scope();
}

bool Executor::InitParamAggregator(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &cnodes = func_graph->GetOrderedCnodes();
  for (const auto &cnode : cnodes) {
    MS_EXCEPTION_IF_NULL(cnode);
    const std::string &param_name = GetTrainableParamName(cnode);
    if (param_name.empty()) {
      continue;
    }
    if (param_aggrs_.count(param_name) != 0) {
      MS_LOG(WARNING) << param_name << " already has parameter aggregator registered.";
      continue;
    }

    std::shared_ptr<ParameterAggregator> param_aggr = std::make_shared<ParameterAggregator>();
    MS_EXCEPTION_IF_NULL(param_aggr);
    param_names_.push_back(param_name);
    param_aggrs_[param_name] = param_aggr;
    parameter_mutex_[param_name];
    if (!param_aggr->Init(cnode, aggregation_count_)) {
      MS_LOG(EXCEPTION) << "Initializing parameter aggregator for " << param_name << " failed.";
      return false;
    }
    MS_LOG(DEBUG) << "Initializing parameter aggregator for param_name " << param_name << " success.";
  }
  return true;
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
