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

#include "tools/converter/micro/coder/opcoders/parallel.h"
#include <string>
#include "coder/log.h"

namespace mindspore::lite::micro {
const char *gThreadNum = nullptr;
void FreeThread() {
  if (gThreadNum != nullptr) {
    free(const_cast<char *>(gThreadNum));
    gThreadNum = nullptr;
  }
}

void InitThread(int model_index) {
  FreeThread();
  std::string weight_name = "m" + std::to_string(model_index) + "_thread_num";
  gThreadNum = static_cast<const char *>(malloc((weight_name.size() + 1) * sizeof(char)));
  int ret = snprintf(const_cast<char *>(gThreadNum), weight_name.size() + 1, "%s\n", weight_name.c_str());
  if (ret == RET_ERROR) {
    MS_LOG(ERROR) << "snprintf failed";
    return;
  }
}
}  // namespace mindspore::lite::micro
