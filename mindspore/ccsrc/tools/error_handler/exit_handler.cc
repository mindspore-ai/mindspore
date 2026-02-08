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

#include "tools/error_handler/exit_handler.h"
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <unistd.h>
#include <dirent.h>
#endif
#include <signal.h>
#include <string>
#include <sstream>
#include <mutex>
#include "utils/ms_context.h"
#include "include/utils/utils.h"
#include "include/utils/callback.h"

namespace mindspore {
namespace tools {
namespace {
constexpr size_t kMaxPathNameLen = 256;
constexpr size_t kMaxProcFileLen = 128;
constexpr size_t kTftWaitSeconds = 2;
}  // namespace
bool TFTWaitSem::isEnable_ = false;
TFTWaitSem &TFTWaitSem::GetInstance() {
  static TFTWaitSem instance;
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
  static std::once_flag initFlag = {};
  std::call_once(initFlag, [&]() { sem_init(&(instance.waitSem_), 0, 0); });
#endif
  return instance;
}

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
void TFTWaitSem::Wait() {
  while (sem_trywait(&waitSem_) != 0) {
    if (HasThreadsExited()) {
      MS_LOG(WARNING) << "Found mindio heartbeat or processor exited, no need to wait semaphore.";
      // mindio registered SIGTERM, here restore it to default action
      signal(SIGTERM, SIG_DFL);
      break;
    }
    MS_LOG(INFO) << "Semaphore for tft is not released, check it later.";
    sleep(kTftWaitSeconds);
  }
}

void TFTWaitSem::Post() { sem_post(&waitSem_); }

void TFTWaitSem::Clear() { sem_destroy(&waitSem_); }

void TFTWaitSem::RecordThreads(bool is_start) {
  char path[kMaxPathNameLen];
  if (snprintf_s(path, sizeof(path), kMaxProcFileLen, "/proc/%d/task", getpid()) < 0) {
    MS_LOG(ERROR) << "snprintf_s encountered an error when constructing file path.";
    tft_thread_ids_.clear();
    return;
  }

  DIR *proc_dir = opendir(path);
  if (proc_dir == nullptr) {
    MS_LOG(ERROR) << "Open path " << path << "failed.";
    tft_thread_ids_.clear();
    return;
  }

  if (is_start) {
    tft_thread_ids_.clear();
  }

  // /proc available, iterate through tasks...
  struct dirent *entry;
  while ((entry = readdir(proc_dir)) != NULL) {
    if (entry->d_name[0] == '.') continue;
    auto thread_id = std::stoul(entry->d_name);
    if (is_start) {
      (void)tft_thread_ids_.insert(thread_id);
    } else {
      // clear original tids, and insert tft newly created tids
      if (tft_thread_ids_.erase(thread_id) == 0) {
        (void)tft_thread_ids_.insert(thread_id);
      }
    }
  }
  (void)closedir(proc_dir);

  if (!is_start) {
    std::ostringstream oss;
    for (const auto &tid : tft_thread_ids_) {
      oss << " " << tid;
    }
    MS_LOG(INFO) << "TFT create " << tft_thread_ids_.size() << " new threads, thread ids are" << oss.str();
  }
}

bool TFTWaitSem::HasThreadsExited() {
  char path[kMaxPathNameLen];
  for (const auto &tid : tft_thread_ids_) {
    if (snprintf_s(path, sizeof(path), kMaxProcFileLen, "/proc/%d/task/%d", getpid(), tid) < 0) {
      MS_LOG(ERROR) << "snprintf_s encountered an error when constructing file path.";
      continue;
    }
    if (access(path, F_OK) == -1) {
      return true;
    }
  }
  return false;
}
#else
void TFTWaitSem::Wait() {}
void TFTWaitSem::Post() {}
void TFTWaitSem::Clear() {}
void TFTWaitSem::RecordThreads(bool is_start) {}
bool TFTWaitSem::HasThreadsExited() { return false; }
#endif
TFTWaitSem::TFTWaitSem() {}
TFTWaitSem::~TFTWaitSem() {}
void TFTWaitSem::Enable() { isEnable_ = true; }
bool TFTWaitSem::IsEnable() {
  auto msContext = MsContext::GetInstance();
  if (!IsGraphPipelineCompiled() || msContext->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice) {
    return false;
  }
  return isEnable_;
}
void TFTWaitSem::StartRecordThreads() { RecordThreads(true); }

void TFTWaitSem::FinishRecordThreads() { RecordThreads(false); }

void TftWaitSemaphore() {
  if (TFTWaitSem::IsEnable()) {
    MS_LOG(INFO) << "Start waiting for TFT.";
    TFTWaitSem::GetInstance().Wait();
    MS_LOG(INFO) << "End waiting for TFT.";
  }
}

void TftDestroyWaitSemaphore() { TFTWaitSem::GetInstance().Clear(); }

REGISTER_COMMON_CALLBACK(TftWaitSemaphore);
REGISTER_COMMON_CALLBACK(TftDestroyWaitSemaphore);
}  // namespace tools
}  // namespace mindspore
