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

#include "ir/anf.h"
#include "mindapi/base/macros.h"

namespace mindspore {
class ParamInfo;
using ParamInfoPtr = std::shared_ptr<ParamInfo>;

class MS_CORE_API ParamInfo {
 public:
  ParamInfo() {}

  ParamInfo(const ParamInfo &other) = default;
  ParamInfo &operator=(const ParamInfo &other) = default;

  virtual ~ParamInfo() = default;

  const std::string &name() const { return name_; }
  void set_name(const std::string &name) { name_ = name; }

  bool requires_grad() const { return requires_grad_; }
  void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

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
    clone->requires_aggr_ = this->requires_aggr_;
    clone->strategy_ckpt_saved_ = this->strategy_ckpt_saved_;
    clone->param_strategy_ = this->param_strategy_;
    clone->storage_format_ = this->storage_format_;
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

  void set_strategy_ckpt_saved(bool strategy_ckpt_saved) { strategy_ckpt_saved_ = strategy_ckpt_saved; }
  bool strategy_ckpt_saved() const { return strategy_ckpt_saved_; }

  bool use_persistent_storage() const { return use_persistent_storage_; }
  void set_use_persistent_storage(bool use_persistent_storage) { use_persistent_storage_ = use_persistent_storage; }

  const std::vector<int64_t> &origin_shape() const { return origin_shape_; }
  void set_origin_shape(const std::vector<int64_t> &origin_shape) { origin_shape_ = origin_shape; }

  bool cache_enable() const { return cache_enable_; }
  void set_cache_enable(bool cache_enable) { cache_enable_ = cache_enable; }

  const std::vector<int64_t> &param_strategy() const { return param_strategy_; }
  void set_param_strategy(const std::vector<int64_t> &param_strategy) { param_strategy_ = param_strategy; }

  const std::vector<std::string> &alias_name() const { return alias_name_; }
  void set_alias_name(const std::vector<std::string> &alias_name) { alias_name_ = alias_name; }

  const std::vector<int64_t> &tensor_map() const { return tensor_map_; }
  void set_tensor_map(const std::vector<int64_t> &tensor_map) { tensor_map_ = tensor_map; }

  const std::vector<int64_t> &device_matrix() const { return device_matrix_; }
  void set_device_matrix(const std::vector<int64_t> &device_matrix) { device_matrix_ = device_matrix; }

  const bool &interleaved_parallel() const { return interleaved_parallel_; }
  void set_interleaved_parallel(const bool &interleaved_parallel) { interleaved_parallel_ = interleaved_parallel; }

  std::vector<int64_t> cache_shape() const { return cache_shape_; }
  void set_cache_shape(const std::vector<int64_t> &cache_shape) { cache_shape_ = cache_shape; }
  ParameterPtr parameter() const { return parameter_.lock(); }
  void set_parameter(const ParameterPtr &parameter) { parameter_ = parameter; }
  void ClearParameter() { parameter_.reset(); }

  bool requires_aggr() const { return requires_aggr_; }
  void set_requires_aggr(bool requires_aggr) { requires_aggr_ = requires_aggr; }

  bool is_quant_int4() const { return is_quant_int4_; }
  void set_is_quant_int4(bool is_quant_int4) { is_quant_int4_ = is_quant_int4; }

  std::vector<int64_t> quant_shape() const { return quant_shape_; }
  void set_quant_shape(const std::vector<int64_t> &quant_shape) { quant_shape_ = quant_shape; }

  bool ignore_device_addr() const { return ignore_device_addr_; }
  void set_ignore_device_addr(bool ignore) { ignore_device_addr_ = ignore; }

  std::string storage_format() const { return storage_format_; }
  void set_storage_format(const std::string &storage_format) { storage_format_ = storage_format; }

  bool is_pipeline_shared_param() const { return is_pipeline_shared_param_; }
  void set_is_pipeline_shared_param(bool is_pipeline_shared_param) {
    is_pipeline_shared_param_ = is_pipeline_shared_param;
  }

  bool is_param_init() const { return is_param_init_; }
  void set_is_param_init(bool is_param_init) { is_param_init_ = is_param_init; }

  bool is_in_pynative_shard() const { return is_in_pynative_shard_; }
  void set_is_in_pynative_shard(bool is_in_pynative_shard) { is_in_pynative_shard_ = is_in_pynative_shard; }

 private:
  std::string name_{"Parameter"};
  bool requires_grad_{true};
  bool layerwise_parallel_{false};
  bool be_cloned_{false};
  bool strategy_ckpt_saved_{false};
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
  std::string storage_format_{""};
  bool is_in_pynative_shard_{false};

  // Record the origin shape before cut huge parameter to a small one.
  std::vector<int64_t> origin_shape_;
  // This flag indicates whether the persistent storage capability is enabled, which is generally used in very large
  // parameter scenarios.
  bool use_persistent_storage_{false};

  // Used to identify the same Parameter for Worker and Server in the embedding cache scenario.
  int32_t key_{-1};
  // Used to indicate parameter strategy, only take effect in cell shard
  std::vector<int64_t> param_strategy_;
  // Used to indicate parameter layout, only take effect in cell shard
  std::vector<std::string> alias_name_;
  std::vector<int64_t> tensor_map_;
  std::vector<int64_t> device_matrix_;
  bool interleaved_parallel_{false};

  // Used to identify parameters of quant int4 type
  bool is_quant_int4_{false};
  std::vector<int64_t> quant_shape_;
  // Used to ignore unused param
  bool ignore_device_addr_{false};
  // Used to indicate shared parameter for pipeline parallel
  bool is_pipeline_shared_param_{false};
  // Used to indicate is auto_parallel mode and parameter is inited.
  bool is_param_init_{false};
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_PARAM_INFO_H_
