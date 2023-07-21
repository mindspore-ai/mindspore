/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_PROFILING_ASCEND_PROFILING_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_PROFILING_ASCEND_PROFILING_H_

#include <string>
#include <nlohmann/json.hpp>
#include "acl/acl_prof.h"

namespace mindspore::kernel {
namespace acl {
class Profiling {
 public:
  Profiling() = default;
  ~Profiling() = default;
  bool Init(const std::string &profiling_file, uint32_t device_id);
  bool IsProfilingOpen() { return is_profiling_open_; }
  bool StartProfiling(const aclrtStream &stream);
  bool StopProfiling(const aclrtStream &stream);

 private:
  bool is_profiling_open_{false};
  std::string output_path_;
  uint32_t device_id_;
  uint64_t profiling_mask_;
  aclprofConfig *acl_config_{nullptr};
  aclprofAicoreMetrics aic_metrics_{ACL_AICORE_PIPE_UTILIZATION};
  nlohmann::json profiling_json_;
};
}  // namespace acl
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_PROFILING_ASCEND_PROFILING_H_
