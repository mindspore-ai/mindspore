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

#include "runtime/device/ascend/signal_util.h"
#include <signal.h>
#include "utils/log_adapter.h"
#include "backend/session/kernel_build_client.h"

namespace mindspore {
SignalGuard::SignalGuard() { RegisterHandlers(); }

SignalGuard::~SignalGuard() {
  if (old_handler != nullptr) {
    int_action.sa_sigaction = old_handler;
    (void)sigemptyset(&int_action.sa_mask);
    int_action.sa_flags = SA_RESTART | SA_SIGINFO;
    (void)sigaction(SIGINT, &int_action, nullptr);
  }
}

void SignalGuard::RegisterHandlers() {
  struct sigaction old_int_action;
  (void)sigaction(SIGINT, nullptr, &old_int_action);
  if (old_int_action.sa_sigaction != nullptr) {
    MS_LOG(INFO) << "The signal has been registered";
    old_handler = old_int_action.sa_sigaction;
  }
  int_action.sa_sigaction = &IntHandler;
  (void)sigemptyset(&int_action.sa_mask);
  int_action.sa_flags = SA_RESTART | SA_SIGINFO;
  (void)sigaction(SIGINT, &int_action, nullptr);
}

void SignalGuard::IntHandler(int, siginfo_t *, void *) {
  kernel::AscendKernelBuildClient::Instance().Close();
  int this_pid = getpid();
  MS_LOG(WARNING) << "Process " << this_pid << " receive KeyboardInterrupt signal.";
  (void)kill(this_pid, SIGTERM);
}
}  // namespace mindspore
