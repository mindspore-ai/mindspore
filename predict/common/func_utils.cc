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

#include "common/func_utils.h"

namespace mindspore {
namespace predict {
#if MS_USE_ARM
_Unwind_Reason_Code PrintTraceArm(_Unwind_Context *ctx, void *d) {
  MS_ASSERT(ctx != nullptr);
  MS_ASSERT(d != nullptr);
  Dl_info info;
  int *depth = static_cast<int *>(d);
  auto ipAddr = static_cast<int64_t>(_Unwind_GetIP(ctx));
  if (dladdr(reinterpret_cast<void *>(ipAddr), &info)) {
    const char *symbol = "";
    const char *dlfile = "";
    if (info.dli_sname) {
      symbol = info.dli_sname;
    }
    if (info.dli_fname) {
      dlfile = info.dli_fname;
    }
    MS_PRINT_ERROR("#%d: (%08lx) %s %s ", *depth, ipAddr, dlfile, symbol);
  }

  (*depth)++;
  return _URC_NO_REASON;
}
#endif

void CoreDumpTraceFunc(int iSignum) {
  MS_PRINT_ERROR("----- start get backtrace info -----");
#if MS_USE_ARM
  int depth = 0;
  _Unwind_Backtrace(&PrintTraceArm, &depth);
#else
  const auto maxDeep = 32;
  const auto maxStringLen = 100;
  void *apBuffer[maxStringLen];
  char **ppStrings;

  auto iStackDepth = backtrace(apBuffer, maxDeep);
  if (0 > iStackDepth) {
    KillProcess("Get backtrace depth failed");
    return;
  }
  MS_PRINT_ERROR("Current stack depth is %d", iStackDepth);
  ppStrings = backtrace_symbols(apBuffer, iStackDepth);
  if (nullptr == ppStrings) {
    KillProcess("Get backtrace_symbols failed");
    return;
  }

  for (int iLoop = 0; iLoop < iStackDepth; iLoop++) {
    MS_PRINT_ERROR("%s \n", ppStrings[iLoop]);
  }
#endif
  MS_PRINT_ERROR("----- finish get backtrace info -----");
  KillProcess("Exit after core dump");
  return;  // try exit 1
}
}  // namespace predict
}  // namespace mindspore
