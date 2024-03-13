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

#include "transform/acl_ir/op_api_exec.h"

namespace mindspore::transform {
namespace {
using InitHugeMemThreadLocalCast = int (*)(void *, bool);
using UnInitHugeMemThreadLocalCast = void (*)(void *, bool);
using ReleaseHugeMemCast = void (*)(void *, bool);
}  // namespace

static std::mutex init_mutex;
static bool aclnn_init = false;
HashMap<void *, std::string> opapi_lib_handle;

OpApiDefaultResource &OpApiDefaultResource::GetInstance() {
  static OpApiDefaultResource instance;
  return instance;
}

InitHugeMemThreadLocal OpApiDefaultResource::init_mem_func() {
  if (init_mem_func_ != nullptr) {
    return init_mem_func_;
  }
  auto init_mem_func = GetOpApiFunc("InitHugeMemThreadLocal");
  if (init_mem_func == nullptr) {
    MS_LOG(EXCEPTION) << "InitHugeMemThreadLocal not in " << GetOpApiLibName() << ", please check!";
  }
  init_mem_func_ = reinterpret_cast<InitHugeMemThreadLocalCast>(init_mem_func);
  return init_mem_func_;
}

UnInitHugeMemThreadLocal OpApiDefaultResource::uninit_mem_func() {
  if (uninit_mem_func_ != nullptr) {
    return uninit_mem_func_;
  }
  auto uninit_mem_func = GetOpApiFunc("UnInitHugeMemThreadLocal");
  if (uninit_mem_func == nullptr) {
    MS_LOG(EXCEPTION) << "UnInitHugeMemThreadLocal not in " << GetOpApiLibName() << ", please check!";
  }
  uninit_mem_func_ = reinterpret_cast<UnInitHugeMemThreadLocalCast>(uninit_mem_func);
  return uninit_mem_func_;
}

ReleaseHugeMem OpApiDefaultResource::release_mem_func() {
  if (release_mem_func_ != nullptr) {
    return release_mem_func_;
  }
  auto release_mem_func = GetOpApiFunc("ReleaseHugeMem");
  if (release_mem_func == nullptr) {
    MS_LOG(EXCEPTION) << "ReleaseHugeMem not in " << GetOpApiLibName() << ", please check!";
  }
  release_mem_func_ = reinterpret_cast<ReleaseHugeMemCast>(release_mem_func);
  return release_mem_func_;
}

ShapeVector UpdateOutputShape(const aclTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  static const auto op_api_func = GetOpApiFunc("aclGetViewShape");
  if (op_api_func == nullptr) {
    MS_LOG(EXCEPTION) << "aclGetViewShape not in " << GetOpApiLibName() << ", please check!";
  }
  using aclGetViewShapeFunc = int (*)(const aclTensor *tensor, int64_t **view_dims, uint64_t *view_dims_num);
  auto aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(op_api_func);
  int64_t *view_dims = nullptr;
  uint64_t view_dim_num = 0;
  auto ret = aclGetViewShape(tensor, &view_dims, &view_dim_num);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "aclGetViewShape failed!";
  }
  ShapeVector output_shape(view_dims, view_dims + view_dim_num);
  delete view_dims;
  view_dims = nullptr;
  return output_shape;
}

void LoadOpApiLib() {
  auto cust_paths = common::GetEnv("ASCEND_CUSTOM_OPP_PATH");
  std::vector<std::string> cust_path_vec;
  if (!cust_paths.empty()) {
    std::regex re{":"};
    std::vector<std::string> split_path_vec(std::sregex_token_iterator(cust_paths.begin(), cust_paths.end(), re, -1),
                                            std::sregex_token_iterator());
    for (const auto &cust_path : split_path_vec) {
      if (cust_path.empty()) {
        continue;
      }
      auto lib_path = cust_path + GetCustOpApiLibName();
      auto ret = access(lib_path.c_str(), F_OK);
      if (ret == 0) {
        cust_path_vec.push_back(lib_path);
      }
    }
  }

  for (const auto &cust_lib_path : cust_path_vec) {
    auto cust_handler = GetOpApiLibHandler(cust_lib_path);
    if (cust_handler != nullptr) {
      MS_LOG(DEBUG) << "Load cust open api lib " << cust_lib_path << " success";
      opapi_lib_handle[cust_handler] = cust_lib_path;
    }
  }

  auto ascend_path = mindspore::transform::GetAscendPath();
  const std::vector<std::string> depend_libs = {"libdummy_tls.so", "libnnopbase.so"};
  for (const auto &dep_lib : depend_libs) {
    (void)GetOpApiLibHandler(ascend_path + "lib64/" + dep_lib);
  }

  auto lib_path = ascend_path + GetOpApiLibName();
  auto handle = GetOpApiLibHandler(lib_path);
  if (handle != nullptr) {
    MS_LOG(DEBUG) << "Load open api lib " << lib_path << " success";
    opapi_lib_handle[handle] = lib_path;
  }
  MS_LOG(DEBUG) << "Load all open api lib success";
}

void AclnnInit() {
  std::lock_guard<std::mutex> lock(init_mutex);
  if (aclnn_init) {
    return;
  }
  static const auto aclnn_init_func = GetOpApiFunc("aclnnInit");
  if (aclnn_init_func == nullptr) {
    MS_LOG(EXCEPTION) << "aclnnInit not in " << GetOpApiLibName() << ", please check!";
  }
  using aclnnInitFunc = int (*)(const char *);
  auto aclnnInit = reinterpret_cast<aclnnInitFunc>(aclnn_init_func);

  auto ret = aclnnInit(nullptr);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "aclnnInit failed!";
  }
  aclnn_init = true;
  MS_LOG(DEBUG) << "aclnn init success!";
}

void AclnnFinalize() {
  if (!aclnn_init) {
    return;
  }
  static const auto aclnn_finalize_func = GetOpApiFunc("aclnnFinalize");
  if (aclnn_finalize_func == nullptr) {
    MS_LOG(EXCEPTION) << "aclnnFinalize not in " << GetOpApiLibName() << ", please check!";
  }
  using aclnnFinalizeFunc = int (*)();
  auto aclnnFinalize = reinterpret_cast<aclnnFinalizeFunc>(aclnn_finalize_func);

  auto ret = aclnnFinalize();
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "aclnnFinalize failed!";
  }
  aclnn_init = false;
  MS_LOG(DEBUG) << "aclnn finalize success!";
}
}  // namespace mindspore::transform
