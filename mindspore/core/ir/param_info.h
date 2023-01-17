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

#ifndef MINDSPORE_CORE_IR_PARAM_INFO_H_
#define MINDSPORE_CORE_IR_PARAM_INFO_H_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "ir/dtype.h"

namespace mindspore {
class ParamInfo;
using ParamInfoPtr = std::shared_ptr<ParamInfo>;

class ParamInfo {
 public:
  ParamInfo() {}

  ParamInfo(const ParamInfo &other) = default;
  ParamInfo &operator=(const ParamInfo &other) = default;

  virtual ~ParamInfo() = default;

  const std::string &name() const { return name_; }
  void set_name(const std::string &name) { name_ = name; }

  bool requires_grad() const { return requires_grad_; }
  void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

  bool init_in_server() const { return init_in_server_; }
  void set_init_in_server(bool init_in_server) { init_in_server_ = init_in_server; }

  // Get the unique key of parameter.
  int32_t key() const { return key_; }
  // Set the unique key of parameter.
  void set_key(int32_t key) { key_ = key; }

  bool layerwise_parallel() const { return layerwise_parallel_; }
  void set_layerwise_parallel(bool layerwise_parallel) { layerwise_parallel_ = layerwise_parallel; }

  // Whether the parameter clone from other parameter.
  bool cloned() const { return cloned_; }

  // Whether the parameter is cloned.
  bool be_cloned() const { return be_cloned_; }

  // If the parameter is cloned, generate one index per clone.
  const std::vector<int32_t> &be_cloned_index() const { return be_cloned_index_; }

  // If the parameter clone from other parameter, it has a unique index.
  int32_t cloned_index() const { return cloned_index_; }

  // Make a cloned parameter and update clone info.
  ParamInfoPtr Clone() {
    static std::atomic<int32_t> parameter_cloned_index{1};
    int32_t index = parameter_cloned_index.fetch_add(1, std::memory_order_relaxed);
    auto clone = std::make_shared<ParamInfo>(*this);
    clone->be_cloned_ = false;
    clone->cloned_ = true;
    clone->be_cloned_index_ = {};
    clone->cloned_index_ = index;
    this->be_cloned_ = true;
    this->be_cloned_index_.push_back(index);
    clone->init_in_server_ = this->init_in_server_;
    clone->requires_aggr_ = this->requires_aggr_;
    clone->param_strategy_ = this->param_strategy_;
    clone->ClearParameter();
    return clone;
  }

  int32_t comm_fusion() const { return fusion_type_; }
  void set_comm_fusion(int32_t fusion_type) { fusion_type_ = fusion_type; }

  bool parallel_optimizer() const { return parallel_optimizer_; }
  void set_parallel_optimizer(bool parallel_optimizer) { parallel_optimizer_ = parallel_optimizer; }

  bool parallel_optimizer_comm_recompute() const { return parallel_optimizer_comm_recompute_; }
  void set_parallel_optimizer_comm_recompute(bool parallel_optimizer_comm_recompute) {
    parallel_optimizer_comm_recompute_ = parallel_optimizer_comm_recompute;
  }

  const std::vector<int64_t> &parameter_shape() const { return parameter_shape_; }
  void set_parameter_shape(const std::vector<int64_t> &tensor_shape) { parameter_shape_ = tensor_shape; }

  bool use_persistent_storage() const { return use_persistent_storage_; }
  void set_use_persistent_storage(bool use_persistent_storage) { use_persistent_storage_ = use_persistent_storage; }

  const std::vector<int64_t> &origin_shape() const { return origin_shape_; }
  void set_origin_shape(const std::vector<int64_t> &origin_shape) { origin_shape_ = origin_shape; }

  bool cache_enable() const { return cache_enable_; }
  void set_cache_enable(bool cache_enable) { cache_enable_ = cache_enable; }

  const std::vector<int64_t> &param_strategy() const { return param_strategy_; }
  void set_param_strategy(const std::vector<int64_t> &param_strategy) { param_strategy_ = param_strategy; }

  std::vector<int64_t> cache_shape() const { return cache_shape_; }
  void set_cache_shape(const std::vector<int64_t> &cache_shape) { cache_shape_ = cache_shape; }
  ParameterPtr parameter() const { return parameter_.lock(); }
  void set_parameter(const ParameterPtr &parameter) { parameter_ = parameter; }
  void ClearParameter() { parameter_.reset(); }

  bool requires_aggr() const { return requires_aggr_; }
  void set_requires_aggr(bool requires_aggr) { requires_aggr_ = requires_aggr; }

 private:
  std::string name_{"Parameter"};
  bool requires_grad_{true};
  bool init_in_server_{false};
  bool layerwise_parallel_{false};
  bool be_cloned_{false};
  bool cloned_{false};
  std::vector<int32_t> be_cloned_index_;
  int32_t cloned_index_{0};
  int32_t fusion_type_{1};
  bool parallel_optimizer_{true};
  bool parallel_optimizer_comm_recompute_{false};
  bool cache_enable_{false};
  std::vector<int64_t> cache_shape_;
  ParameterWeakPtr parameter_;
  bool requires_aggr_{true};
  std::vector<int64_t> parameter_shape_;

  // Record the origin shape before cut huge parameter to a small one.
  std::vector<int64_t> origin_shape_;
  // This flag indicates whether the persistent storage capability is enabled, which is generally used in very large
  // parameter scenarios.
  bool use_persistent_storage_{false};

  // Used to identify the same Parameter for Worker and Server in the embedding cache scenario.
  int32_t key_{-1};
  // Used to indicate parameter strategy, only take effect in cell shard
  std::vector<int64_t> param_strategy_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_PARAM_INFO_H_
