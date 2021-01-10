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
#include "minddata/dataset/util/sig_handler.h"
#include <signal.h>
#include <sys/types.h>
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <ucontext.h>
#endif
#include <unistd.h>
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
// Register the custom signal handlers
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
void RegisterHandlers() {
  struct sigaction new_int_action;
  struct sigaction new_term_action;

  // For the interrupt handler, we do not use SA_RESETHAND so this handler remains in play
  // permanently, do not use the OS default handler for it.
  new_int_action.sa_sigaction = [](int num, siginfo_t *info, void *ctx) { IntHandler(num, info, ctx); };
  (void)sigemptyset(&new_int_action.sa_mask);
  new_int_action.sa_flags = SA_RESTART | SA_SIGINFO;

  new_term_action.sa_sigaction = [](int num, siginfo_t *info, void *ctx) { IntHandler(num, info, ctx); };
  (void)sigemptyset(&new_term_action.sa_mask);
  new_term_action.sa_flags = SA_RESTART | SA_SIGINFO;

  (void)sigaction(SIGINT, &new_int_action, nullptr);
  (void)sigaction(SIGTERM, &new_term_action, nullptr);
}

extern void IntHandler(int sig_num,          // The signal that was raised
                       siginfo_t *sig_info,  // The siginfo structure.
                       void *context) {      // context info
  // Wake up the watchdog which is designed as async-signal-safe.
  TaskManager::WakeUpWatchDog();
}
#endif
}  // namespace dataset
}  // namespace mindspore
