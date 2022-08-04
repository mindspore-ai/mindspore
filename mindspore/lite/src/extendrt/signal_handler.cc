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
#include "src/extendrt/signal_handler.h"
#include <execinfo.h>
#include <signal.h>
#include <map>
#include <string>
#include <vector>
#include "src/common/log_adapter.h"

namespace mindspore {
namespace {
static std::map<int, std::string> kSigs = {
  {SIGSEGV, "SIGSEGV"}, {SIGABRT, "SIGABRT"}, {SIGFPE, "SIGFPE"}, {SIGBUS, "SIGBUS"}, {SIGILL, "SIGILL"}};

std::string GetSigStr(int sig) { return kSigs[sig]; }
}  // namespace

std::string GetBackTrace(int sig) {
  constexpr int kFrameSize = 128;
  void *frames_buf[kFrameSize];
  auto frames_num = backtrace(frames_buf, kFrameSize);
  char **frames = backtrace_symbols(frames_buf, frames_num);
  if (frames == nullptr) {
    MS_LOG(ERROR) << "backtrace_symbols failed!";
    std::cerr << "backtrace_symbols failed!";
    return "";
  }

  std::stringstream frames_result;
  frames_result << "Capture signal [" << GetSigStr(sig) << "]" << std::endl;
  for (int i = 0; i < frames_num; ++i) {
    frames_result << frames[i] << std::endl;
  }

  free(frames);
  return frames_result.str();
}

void SignalHandler(int sig) {
  std::string stack_info = GetBackTrace(sig);
  MS_LOG(ERROR) << stack_info;
  std::cerr << stack_info << std::endl;
  struct sigaction sa_default;
  sigemptyset(&sa_default.sa_mask);
  sa_default.sa_handler = SIG_DFL;
  sigaction(SIGABRT, &sa_default, NULL);
  abort();
}

void CaptureSignal() {
  static bool capture_already = false;
  if (capture_already) {
    return;
  }
  std::vector<int> sigs = {SIGSEGV, SIGABRT, SIGFPE, SIGBUS, SIGILL};
  for (auto sig : sigs) {
    struct sigaction sa;
    sa.sa_handler = SignalHandler;
    auto ret = sigaction(sig, &sa, NULL);
    if (ret != 0) {
      MS_LOG(ERROR) << "sigaction [" << sig << "] failed!";
    }
  }
  capture_already = true;
}
}  // namespace mindspore
