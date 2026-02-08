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

#ifndef MINDSPORE_CCSRC_DEBUG_TFT_ADAPTER_TFT_WAIT_SEM_H_
#define MINDSPORE_CCSRC_DEBUG_TFT_ADAPTER_TFT_WAIT_SEM_H_
#include <unordered_set>
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <semaphore.h>
#endif
#include <utility>
#include "tools/visible.h"

namespace mindspore {
namespace tools {
class TOOLS_EXPORT TFTWaitSem {
 public:
  static TFTWaitSem &GetInstance();
  ~TFTWaitSem();
  TFTWaitSem(const TFTWaitSem &) = delete;
  TFTWaitSem &operator=(const TFTWaitSem &) = delete;
  void Wait();
  void Post();
  void Clear();
  void StartRecordThreads();
  void FinishRecordThreads();
  bool HasThreadsExited();
  static void Enable();
  static bool IsEnable();

 private:
  TFTWaitSem();
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
  sem_t waitSem_;
  std::unordered_set<pid_t> tft_thread_ids_;
#endif
  static bool isEnable_;
  void RecordThreads(bool is_start);
};
}  // namespace tools
}  // namespace mindspore
#endif
