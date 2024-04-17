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
#include <string>
#include "transform/symbol/acl_base_symbol.h"
#include "transform/symbol/acl_compiler_symbol.h"
#include "transform/symbol/acl_mdl_symbol.h"
#include "transform/symbol/acl_op_symbol.h"
#include "transform/symbol/acl_prof_symbol.h"
#include "transform/symbol/acl_rt_allocator_symbol.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/acl_symbol.h"
#include "transform/symbol/acl_tdt_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore {
namespace transform {

static bool load_ascend_api = false;

void *GetLibHandler(const std::string &lib_path) {
  auto handler = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (handler == nullptr) {
    MS_LOG(INFO) << "Dlopen " << lib_path << " failed!" << dlerror();
  }
  return handler;
}

std::string GetAscendPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(aclrtMalloc), &info) == 0) {
    MS_LOG(INFO) << "Get dladdr failed, skip.";
    return "";
  }
  auto path_tmp = std::string(info.dli_fname);
  const std::string kLib64 = "lib64";
  auto pos = path_tmp.find(kLib64);
  if (pos == std::string::npos) {
    MS_EXCEPTION(ValueError) << "Get ascend path failed, please check the run package.";
  }
  return path_tmp.substr(0, pos);
}

void LoadAscendApiSymbols() {
  if (load_ascend_api) {
    MS_LOG(INFO) << "Ascend api is already loaded.";
    return;
  }
  std::string ascend_path = GetAscendPath();
  LoadAclBaseApiSymbol(ascend_path);
  LoadAclOpCompilerApiSymbol(ascend_path);
  LoadAclMdlApiSymbol(ascend_path);
  LoadAclOpApiSymbol(ascend_path);
  LoadProfApiSymbol(ascend_path);
  LoadAclAllocatorApiSymbol(ascend_path);
  LoadAclRtApiSymbol(ascend_path);
  LoadAclApiSymbol(ascend_path);
  LoadAcltdtApiSymbol(ascend_path);
  load_ascend_api = true;
  MS_LOG(INFO) << "Load ascend api success!";
}

}  // namespace transform
}  // namespace mindspore
