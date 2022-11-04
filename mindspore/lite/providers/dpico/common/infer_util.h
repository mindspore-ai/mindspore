/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_PROVIDERS_DPICO_COMMON_INFER_UTIL_H_
#define MINDSPORE_LITE_PROVIDERS_DPICO_COMMON_INFER_UTIL_H_

#include <string>
#include <vector>
#include <thread>
#include "include/api/types.h"
#include "schema/model_generated.h"

namespace mindspore {
namespace lite {
int CheckCustomInputOutput(const std::vector<mindspore::MSTensor> *inputs,
                           const std::vector<mindspore::MSTensor> *outputs, const schema::Primitive *primitive);
int CheckCustomParam(const schema::Custom *param, const std::string &param_name);
class DpicoAicpuThreadManager {
 public:
  DpicoAicpuThreadManager() = default;
  ~DpicoAicpuThreadManager() = default;
  int CreateAicpuThread(uint32_t model_id);
  int DestroyAicpuThread();
  bool g_threadExitFlag_{true};

 private:
  uint32_t all_aicpu_task_num_{0};
  bool is_aicpu_thread_activity_{false};
  std::thread aicpu_thread_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_PROVIDERS_DPICO_COMMON_INFER_UTIL_H_
