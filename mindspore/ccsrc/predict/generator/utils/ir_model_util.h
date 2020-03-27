/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PREDICT_GENERATOR_IR_IR_MODEL_UTIL_H_
#define MINDSPORE_MINDSPORE_CCSRC_PREDICT_GENERATOR_IR_IR_MODEL_UTIL_H_
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include "utils/log_adapter.h"

namespace mindspore {
namespace generator {
class IRModelUtil {
 public:
  static IRModelUtil &GetInstance();
  IRModelUtil(const IRModelUtil &) = delete;
  IRModelUtil &operator=(const IRModelUtil &) = delete;
  void Init();

  void set_version(const std::string &version) { version_ = version; }
  void set_stream_num(uint32_t stream_num) { stream_num_ = stream_num; }
  void set_event_num(uint32_t event_num) { event_num_ = event_num; }
  void set_batch_num(uint32_t batch_num) { batch_num_ = batch_num; }
  void set_memory_size(uint32_t memory_size) { memory_size_ = memory_size; }
  void set_weight_size(uint32_t weight_size) { weight_size_ = weight_size; }
  void set_var_size(uint32_t var_size) { var_size_ = var_size; }
  void set_logic_mem_base(uint32_t logic_mem_base) { logic_mem_base_ = logic_mem_base; }
  void set_logic_weight_base(uint32_t logic_weight_base) { logic_weight_base_ = logic_weight_base; }
  void set_logic_var_base(uint32_t logic_var_base) { logic_var_base_ = logic_var_base; }
  void set_priority(uint32_t priority) { priority_ = priority; }
  void set_is_enable_save_model(bool is_enable_save_model) { is_enable_save_model_ = is_enable_save_model; }
  void set_min_static_offset(uint64_t min_static_offset) { min_static_offset_ = min_static_offset; }
  void set_max_dynamic_offset(uint64_t max_dynamic_offset) { max_dynamic_offset_ = max_dynamic_offset; }
  void set_max_mem_size(uint64_t max_mem_size) { max_mem_size_ = max_mem_size; }
  void set_irmodel_mem_base(uint8_t irmodel_mem_base) { irmodel_mem_base_ = irmodel_mem_base; }

  std::string version() const { return version_; }
  uint32_t stream_num() const { return stream_num_; }
  uint32_t event_num() const { return event_num_; }
  uint32_t batch_num() const { return batch_num_; }
  uint64_t memory_size() const { return memory_size_; }
  uint64_t weight_size() const { return weight_size_; }
  uint64_t var_size() const { return var_size_; }
  uint64_t logic_mem_base() const { return logic_mem_base_; }
  uint64_t logic_weight_base() const { return logic_weight_base_; }
  uint64_t logic_var_base() const { return logic_var_base_; }
  uint32_t priority() const { return priority_; }
  bool is_enable_save_model() const { return is_enable_save_model_; }
  uint64_t min_static_offset() const { return min_static_offset_; }
  uint64_t max_dynamic_offset() const { return max_dynamic_offset_; }
  uint64_t max_mem_size() const { return max_mem_size_; }
  uint8_t irmodel_mem_base() const { return irmodel_mem_base_; }

 private:
  IRModelUtil() = default;
  ~IRModelUtil() = default;
  std::string version_;
  uint32_t stream_num_ = 0;
  uint32_t event_num_ = 0;
  uint32_t batch_num_ = 0;
  uint64_t memory_size_ = 0;
  uint64_t weight_size_ = 0;
  uint64_t var_size_ = 0;
  uint64_t logic_mem_base_ = 0;
  uint64_t logic_weight_base_ = 0;
  uint64_t logic_var_base_ = 0;
  uint32_t priority_ = 0;
  bool is_enable_save_model_ = false;
  uint64_t min_static_offset_ = 0;
  uint64_t max_dynamic_offset_ = 0;
  uint64_t max_mem_size_ = 0;
  uint8_t irmodel_mem_base_ = 0;
};
}  // namespace generator
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_PREDICT_GENERATOR_IR_IR_MODEL_UTIL_H_
