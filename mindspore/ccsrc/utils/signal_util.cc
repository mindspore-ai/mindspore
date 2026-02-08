/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "include/utils/signal_util.h"
#include <csignal>
#include "utils/log_adapter.h"

namespace mindspore {
SignalGuard::SignalGuard(IntHandlerFunc func) { RegisterHandlers(func); }

SignalGuard::~SignalGuard() {
  if (old_handler != nullptr) {
    int_action.sa_sigaction = old_handler;
    (void)sigemptyset(&int_action.sa_mask);
    int_action.sa_flags = SA_RESTART | SA_SIGINFO;
    (void)sigaction(SIGINT, &int_action, nullptr);
    old_handler = nullptr;
  }
}

void SignalGuard::RegisterHandlers(IntHandlerFunc IntHandler) {
  struct sigaction old_int_action;
  (void)sigaction(SIGINT, nullptr, &old_int_action);
  if (old_int_action.sa_sigaction != nullptr) {
    MS_LOG(DEBUG) << "The signal has been registered";
    old_handler = old_int_action.sa_sigaction;
  }
  int_action.sa_sigaction = IntHandler;
  (void)sigemptyset(&int_action.sa_mask);
  int_action.sa_flags = SA_RESTART | SA_SIGINFO;
  (void)sigaction(SIGINT, &int_action, nullptr);
}

bool RegisterGlobalSignalHandler(IntHandlerFunc handler) {
  struct sigaction int_action;
  struct sigaction old_int_action;

  MS_LOG(INFO) << "MSCONTEXT_REGISTER_INIT_FUNC register int handler";
  if (sigaction(SIGINT, nullptr, &old_int_action) == -1) {
    MS_LOG(ERROR) << "Failed to retrieve current SIGINT handler";
    return false;
  }
  if (old_int_action.sa_sigaction != nullptr) {
    MS_LOG(INFO) << "The signal has been registered";
  }

  int_action.sa_sigaction = handler;
  (void)sigemptyset(&int_action.sa_mask);
  int_action.sa_flags = SA_RESTART | SA_SIGINFO;

  if (sigaction(SIGINT, &int_action, nullptr) == -1) {
    MS_LOG(ERROR) << "Failed to register SIGINT handler";
    return false;
  }
  return true;
}

void DefaultIntHandler(int, siginfo_t *, void *) {
  int this_pid = getpid();
  (void)kill(this_pid, SIGTERM);
}
}  // namespace mindspore
