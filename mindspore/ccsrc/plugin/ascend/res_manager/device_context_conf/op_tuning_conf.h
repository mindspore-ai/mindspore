/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_DEVICE_CONTEXT_CONF_OP_TUNING_CONF_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_DEVICE_CONTEXT_CONF_OP_TUNING_CONF_H_

#include <memory>
#include <string>

#include "plugin/ascend/res_manager/visible.h"
#include "utils/ms_utils.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace mindspore {
namespace device {
namespace ascend {
class ASCEND_RES_MANAGER_EXPORT OpTuningConf {
 public:
  OpTuningConf() = default;
  ~OpTuningConf() = default;
  OpTuningConf(const OpTuningConf &) = delete;
  OpTuningConf &operator=(const OpTuningConf &) = delete;
  static std::shared_ptr<OpTuningConf> GetInstance();
  void set_jit_compile(const std::string &value) { jit_compile_ = value; }
  void set_aoe_tune_mode(const std::string &tune_mode) { aoe_tune_mode_ = tune_mode; }
  void set_aoe_job_type(const std::string &aoe_config);
  std::string jit_compile() const;
  std::string aoe_job_type() const;
  std::string aoe_tune_mode() const;
  bool EnableAoeOnline() const;
  bool EnableAoeOffline() const;
  bool IsJitCompileConfigured() const { return !jit_compile_.empty(); }
  bool IsAoeTuneModeConfigured() const { return !aoe_tune_mode_.empty(); }
  bool IsAoeJobTypeConfigured() const { return is_aoe_job_type_configured_; }
  bool IsEnableAclnnGlobalCache() const { return is_enable_aclnn_global_cache_; }
  size_t AclnnCacheQueueLength() const { return cache_queue_length_; }
  bool IsAclnnCacheConfigured() const { return is_aclnn_cache_configured_; }
  void SetAclnnGlobalCache(bool set_aclnn_global_cache) {
    is_enable_aclnn_global_cache_ = set_aclnn_global_cache;
    is_aclnn_cache_configured_ = true;
  }
  void SetAclnnCacheQueueLength(size_t cache_queue_length) {
    cache_queue_length_ = cache_queue_length;
    is_aclnn_cache_configured_ = true;
  }

 private:
  static std::shared_ptr<OpTuningConf> inst_context_;
  std::string jit_compile_{""};
  std::string aoe_tune_mode_{""};
  std::string aoe_job_type_{"2"};
  bool is_aoe_job_type_configured_{false};
  bool is_enable_aclnn_global_cache_{false};
  bool is_aclnn_cache_configured_{false};
  size_t cache_queue_length_{10000};
};

ASCEND_RES_MANAGER_EXPORT void RegOpTuningConf(py::module *m);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_DEVICE_CONTEXT_CONF_OP_TUNING_CONF_H_
