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

#if defined(__linux__) && !defined(DEBUG)
#include <csignal>
#endif
#include "tools/converter/converter.h"

#if defined(__linux__) && !defined(DEBUG)
void SignalHandler(int sig) {
  printf("encounter an unknown error, please verify the input model file or build the debug version\n");
  exit(0);
}
#endif

namespace mindspore {
extern "C" {
extern void common_log_init();
}
}  // namespace mindspore

int main(int argc, const char **argv) {
#if defined(__linux__) && !defined(DEBUG)
  signal(SIGSEGV, SignalHandler);
  signal(SIGABRT, SignalHandler);
  signal(SIGFPE, SignalHandler);
  signal(SIGBUS, SignalHandler);
#endif
  int ret = 0;
  try {
    mindspore::common_log_init();
    ret = mindspore::lite::RunConverter(argc, argv);
  } catch (const std::exception &e) {
    std::cout << e.what();
  }
  return ret;
}
