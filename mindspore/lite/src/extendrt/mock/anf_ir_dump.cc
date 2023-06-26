/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
#define PrintDeprecatedLog                                                          \
  static bool already_printed = false;                                              \
  if (already_printed) {                                                            \
    return;                                                                         \
  }                                                                                 \
  already_printed = true;                                                           \
  MS_LOG(WARNING) << "The functionality of dumping function graph IR is disabled, " \
                  << "please recompile source to enable it. See help of building script.";

void DumpIR(const std::string &, const FuncGraphPtr &, bool, LocDumpMode, const std::string &) { PrintDeprecatedLog }

void DumpIR(std::ostringstream &, const FuncGraphPtr &, bool, LocDumpMode) { PrintDeprecatedLog }

void DumpIRForRDR(const std::string &, const FuncGraphPtr &, bool, LocDumpMode) { PrintDeprecatedLog }

#undef PrintDeprecatedLog
}  // namespace mindspore
