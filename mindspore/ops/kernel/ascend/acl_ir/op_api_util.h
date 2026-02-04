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

#ifndef MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_UTIL_H_
#define MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_UTIL_H_

#include <vector>
#include <string>
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "hccl/hccl.h"
#include "acl/acl_op_compiler.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "ir/anf.h"
#if (defined(ENABLE_CPU) && !defined(_WIN32) && !defined(__APPLE__))
#include "mindspore/ccsrc/include/runtime/hardware_abstract/collective/collective_manager.h"
#endif
#include "kernel/ascend/visible.h"

namespace mindspore::device::ascend {

typedef enum : int8_t {
  KEEP_DTYPE = 0,
  ALLOW_FP32_DOWN_PRECISION = 1,
  FORCE_FP16 = 2,
  FORCE_HF32 = 3,
} aclCubeMathType;

class OPS_ASCEND_API OpApiUtil {
 public:
  static aclCubeMathType GetCubeMathType(bool use_hf32 = false);
  static bool IsAllowMatmulHF32();
  static bool IsAllowConvHF32();

  static void GetValidKernelBuildInfo(const AnfNodePtr &node, std::vector<std::string> *input_formats,
                                      std::vector<std::string> *output_formats,
                                      std::vector<std::string> *input_reshape_types,
                                      std::vector<std::string> *output_reshape_types, const KernelType &kernel_type);

  static std::string GetCommName(const std::string &group);
  static inline void CheckWorldSize(const std::string &group, int64_t world_size, const std::string &op_name) {
#if (defined(ENABLE_CPU) && !defined(_WIN32) && !defined(__APPLE__))
    const auto &collective_manager = mindspore::distributed::collective::CollectiveManager::instance();
    if (collective_manager->initialized()) {
      auto expected_world_size = collective_manager->GetLocalGroupSize(group);
      if (world_size != expected_world_size) {
        MS_LOG(EXCEPTION) << op_name << ": world_size must be " << expected_world_size << ", but got " << world_size;
      }
    }
#endif
  }
  static bool NeedRebuildWorkspaceSize(const std::string &group, const std::string &inner_name);
};

class OPS_ASCEND_API AclUtil {
 public:
  static uint8_t KeepOriginDType();

  static aclError SetCompileMode(const int64_t is_dyncmic);

  static aclError SetPrecisionMode(const std::string &mode);

  static void SetOpPrecisionMode();

  // lock runtime
  static std::lock_guard<std::mutex> LockRuntime(const void *stream);
};

OPS_ASCEND_API int64_t GetCacheCapaticy();
}  // namespace  mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_UTIL_H_
