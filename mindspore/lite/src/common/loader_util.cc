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

#include "src/common/loader_util.h"
#include <string.h>
#include <climits>
#include "include/errorcode.h"
#include "src/common/log_util.h"

#ifndef _WIN32

namespace mindspore {
namespace lite {
int SoLoader::Open(const char *so_path, int mode) {
  if ((strlen(so_path)) >= PATH_MAX) {
    MS_LOG(ERROR) << "path is too long";
    return RET_ERROR;
  }
  char resolved_path[PATH_MAX];
  auto resolve_res = realpath(so_path, resolved_path);
  if (resolve_res == nullptr) {
    MS_LOG(ERROR) << "path not exist";
    return RET_ERROR;
  }
  handler_ = dlopen(so_path, mode);
  if (handler_ == nullptr) {
    MS_LOG(ERROR) << "open path failed";
    return RET_ERROR;
  }
  return RET_OK;
}

void *SoLoader::GetFunc(const char *func_name) { return dlsym(handler_, func_name); }

int SoLoader::Close() {
  auto close_res = dlclose(handler_);
  if (close_res != 0) {
    MS_LOG(ERROR) << "can not close handler";
    return RET_ERROR;
  }
  return RET_OK;
}

}  // namespace lite
}  // namespace mindspore

#endif
