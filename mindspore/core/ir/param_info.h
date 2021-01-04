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

  virtual ~ParamInfo() = default;

  const std::string &name() const { return name_; }
  void set_name(const std::string &name) { name_ = name; }

  bool requires_grad() const { return requires_grad_; }
  void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

  bool init_in_server() const { return init_in_server_; }
  void set_init_in_server(bool init_in_server) { init_in_server_ = init_in_server; }

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
    return clone;
  }

  int32_t comm_fusion() const { return fusion_type_; }
  void set_comm_fusion(int32_t fusion_type) { fusion_type_ = fusion_type; }

  bool parallel_optimizer() const { return parallel_optimizer_; }
  void set_parallel_optimizer(bool parallel_optimizer) { parallel_optimizer_ = parallel_optimizer; }

  bool cache_enable() const { return cache_enable_; }
  void set_cache_enable(bool cache_enable) { cache_enable_ = cache_enable; }

  std::vector<int64_t> cache_shape() const { return cache_shape_; }
  void set_cache_shape(const std::vector<int64_t> &cache_shape) { cache_shape_ = cache_shape; }

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
  bool cache_enable_{false};
  std::vector<int64_t> cache_shape_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_PARAM_INFO_H_
