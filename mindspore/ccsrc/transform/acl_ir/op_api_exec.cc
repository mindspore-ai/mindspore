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

void RunOpApi(const std::string &aclnn_api, const aclrtStream acl_stream, void *workspace_addr, uint64_t workspace_size,
              aclOpExecutor *executor, const ReleaseCallBack &release_func) {
  static const auto op_api_func = GetOpApiFunc(aclnn_api.c_str());
  if (op_api_func == nullptr) {
    MS_LOG(EXCEPTION) << aclnn_api << " not in " << GetOpApiLibName() << ", please check!";
  }
  using RunApiFunc = int (*)(void *, uint64_t, aclOpExecutor *, const aclrtStream);
  auto run_api_func = reinterpret_cast<RunApiFunc>(op_api_func);
  auto ret = run_api_func(workspace_addr, workspace_size, executor, acl_stream);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "call " << aclnn_api << " failed, detail:" << aclGetRecentErrMsg();
  }
  if (release_func != nullptr) {
    release_func();
  }
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
}  // namespace mindspore::transform
