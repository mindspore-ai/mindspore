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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_COMMON_UTILS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_COMMON_UTILS_H_

#include <map>
#include <string>
#include <thread>
#include <vector>
#include "include/api/context.h"
#include "include/api/kernel.h"
#include "include/svp_acl_base.h"
#include "schema/model_generated.h"
#include "src/custom_log.h"

namespace mindspore {
namespace lite {
inline constexpr size_t kMinInputSize = 2;
inline constexpr auto kNetType = "net_type";

typedef enum Result : int { SUCCESS = 0, FAILED = 1 } Result;
typedef enum OmNetType : int { OmNetType_CNN = 0, OmNetType_ROI = 1, OmNetType_RECURRENT = 2 } OmNetType;

#define MS_CHECK_FALSE_MSG(value, errcode, msg) \
  do {                                          \
    if ((value)) {                              \
      MS_LOG(ERROR) << #msg;                    \
      return errcode;                           \
    }                                           \
  } while (0)

inline bool IsValidUnsignedNum(const std::string &num_str) {
  return !num_str.empty() && std::all_of(num_str.begin(), num_str.end(), ::isdigit);
}

bool InferDone(const std::vector<mindspore::MSTensor> &tensors);

void ExtractAttrsFromPrimitive(const mindspore::schema::Primitive *primitive,
                               std::map<std::string, std::string> *attrs);

void *ReadBinFile(const std::string &fileName, uint32_t *fileSize);

Result JudgeOmNetType(const schema::Primitive &primitive, OmNetType *net_type);

class DpicoConfigParamExtractor {
 public:
  DpicoConfigParamExtractor() = default;
  ~DpicoConfigParamExtractor() = default;
  void InitDpicoConfigParam(const kernel::Kernel &kernel);
  void UpdateDpicoConfigParam(const kernel::Kernel &kernel);
  size_t GetMaxRoiNum() { return max_roi_num_; }
  float GetNmsThreshold() { return nms_threshold_; }
  float GetScoreThreshold() { return score_threshold_; }
  float GetMinHeight() { return min_height_; }
  float GetMinWidth() { return min_width_; }
  int GetGTotalT() { return g_total_t_; }
  int GetDpicoDetectionPostProcess() { return dpico_detection_post_process_; }
  std::string GetDpicoDumpConfigFile() { return dpico_dump_config_file_; }

 private:
  size_t max_roi_num_{400};
  float nms_threshold_{0.9f};
  float score_threshold_{0.08f};
  float min_height_{1.0f};
  float min_width_{1.0f};
  int g_total_t_{0};
  int dpico_detection_post_process_{0};
  bool has_init_{false};
  std::string dpico_dump_config_file_{""};
};

class DpicoContextManager {
 public:
  DpicoContextManager() = default;
  ~DpicoContextManager() = default;
  Result InitContext(std::string dpico_dump_config_file);
  void DestroyContext();
  svp_acl_rt_context GetSvpContext() { return svp_context_; }

 private:
  svp_acl_rt_context svp_context_{nullptr};
};

class DpicoAicpuThreadManager {
 public:
  DpicoAicpuThreadManager() = default;
  ~DpicoAicpuThreadManager() = default;
  void CreateAicpuThread(uint32_t model_id);
  void DestroyAicpuThread();

 private:
  uint32_t all_aicpu_task_num_{0};
  bool is_aicpu_thread_activity_{false};
  std::thread aicpu_thread_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_COMMON_UTILS_H
