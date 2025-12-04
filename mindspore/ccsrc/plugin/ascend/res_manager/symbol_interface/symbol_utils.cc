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
#include "symbol_utils.h"
#include <string>
#include "acl_base_symbol.h"
#include "acl_compiler_symbol.h"
#include "acl_mdl_symbol.h"
#include "acl_op_symbol.h"
#include "acl_prof_symbol.h"
#include "acl_rt_allocator_symbol.h"
#include "acl_rt_symbol.h"
#include "acl_symbol.h"
#include "acl_tdt_symbol.h"

namespace mindspore::device::ascend {

namespace {
bool load_ascend_api = false;
bool load_simulation_api = false;
}  // namespace

void *GetLibHandler(const std::string &lib_path, bool if_global) {
  void *handler = nullptr;
  if (if_global) {
    handler = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  } else {
    handler = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  }
  if (handler == nullptr) {
    MS_LOG(INFO) << "Dlopen " << lib_path << " failed!" << dlerror();
  }
  return handler;
}

std::string GetAscendPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(aclrtMalloc), &info) == 0) {
    MS_LOG(ERROR) << "Get dladdr failed.";
    return "";
  }
  auto path_tmp = std::string(info.dli_fname);
  const std::string kCann = "cann";
  auto pos = path_tmp.rfind(kCann);
  if (pos == std::string::npos) {
    MS_EXCEPTION(ValueError)
      << "Get ascend path failed, please check whether CANN packages are installed correctly, \n"
         "and environment variables are set by source ${LOCAL_ASCEND}/cann/set_env.sh.";
  }
  return path_tmp.substr(0, pos) + kCann + "/";
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

void LoadSimulationApiSymbols() {
  if (load_simulation_api) {
    MS_LOG(INFO) << "Simulation api is already loaded.";
    return;
  }

  LoadSimulationAclBaseApi();
  LoadSimulationRtApi();
  LoadSimulationTdtApi();
  LoadSimulationAclOpCompilerApi();
  LoadSimulationAclMdlApi();
  LoadSimulationProfApi();
  LoadSimulationAclAllocatorApi();
  load_simulation_api = true;
  MS_LOG(INFO) << "Load simulation api success!";
}
}  // namespace mindspore::device::ascend
