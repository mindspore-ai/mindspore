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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_CXX_API_CONVERTERS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_CXX_API_CONVERTERS_H_

#include <vector>
#include <string>
#include <memory>
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/api/cfg.h"
#include "include/train/train_cfg.h"
#include "src/litert/inner_context.h"
#include "src/litert/c_api/context_c.h"
#include "src/common/log_adapter.h"

namespace mindspore {
class ContextUtils {
 public:
  static std::shared_ptr<lite::InnerContext> Convert(Context *context);
  static std::shared_ptr<lite::InnerContext> Convert(const ContextC *context_c);

 private:
  static void SetContextAttr(int32_t thread_num, int32_t inter_op_parallel_num, bool enable_parallel,
                             const std::vector<int32_t> &affinity_core_list, int delegate_mode,
                             const std::shared_ptr<Delegate> &delegate, lite::InnerContext *inner_context,
                             bool float_mode = false);
  static Status AddCpuDevice(const std::shared_ptr<Allocator> &allocator, int affinity_mode, bool enable_fp16,
                             const std::string &provider, const std::string &provider_device,
                             lite::InnerContext *inner_context);
  static Status AddGpuDevice(bool enable_fp16, uint32_t device_id, int rank_id, int group_size, bool enable_gl_texture,
                             void *gl_context, void *gl_display, const std::string &provider,
                             const std::string &provider_device, const std::shared_ptr<Allocator> &allocator,
                             lite::InnerContext *inner_context);
  static Status AddNpuDevice(bool enable_fp16, int frequency, lite::InnerContext *inner_context);
  static Status AddAscendDevice(lite::InnerContext *inner_context, DeviceInfoContext *device);
  static Status AddCustomDevice(lite::InnerContext *inner_context, const std::shared_ptr<DeviceInfoContext> &device);
  static bool IsAffinityModeValid(int affinity_mode) {
    return affinity_mode >= lite::NO_BIND && affinity_mode <= lite::MID_CPU;
  }
  static void ResetContextDefaultParam(Context *context);
};

inline lite::QuantizationType A2L_ConvertQT(mindspore::QuantizationType qt) {
  if (qt == kNoQuant) {
    return lite::QT_NONE;
  }
  if (qt == kWeightQuant) {
    return lite::QT_WEIGHT;
  }
  if (qt == kFullQuant || qt == kUnknownQuantType) {
    MS_LOG(WARNING) << "QuantizationType " << qt << " does not support, set the quantizationType to default.";
  }
  return lite::QT_DEFAULT;
}

Status A2L_ConvertConfig(const TrainCfg *a_train_cfg, lite::TrainCfg *l_train_cfg);
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_CXX_API_CONVERTERS_H_
