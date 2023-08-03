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

#ifndef MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_EXEC_H_
#define MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_EXEC_H_

#include <dlfcn.h>
#include <vector>
#include <functional>
#include "acl/acl_base.h"
#include "transform/acl_ir/op_api_convert.h"
#include "transform/acl_ir/acl_allocator.h"

namespace mindspore::transform {
using InitHugeMemThreadLocal = std::function<int(void *, bool)>;
using UnInitHugeMemThreadLocal = std::function<void(void *, bool)>;
using ReleaseHugeMem = std::function<void(void *, bool)>;

class OpApiDefaultResource {
 public:
  static OpApiDefaultResource &GetInstance();

  InitHugeMemThreadLocal init_mem_func();
  UnInitHugeMemThreadLocal uninit_mem_func();
  ReleaseHugeMem release_mem_func();

  void *AllocWorkspace(size_t size) { return allocator.AllocFunc(&allocator, size); }

  void FreeWorkspace(void *block) { allocator.FreeFunc(&allocator, block); }

 private:
  OpApiDefaultResource() { allocator.Initialize(); }
  ~OpApiDefaultResource() { allocator.Finalize(); }

  InitHugeMemThreadLocal init_mem_func_{nullptr};
  UnInitHugeMemThreadLocal uninit_mem_func_{nullptr};
  ReleaseHugeMem release_mem_func_{nullptr};

  AclAllocator allocator;
};

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return call(f, t, std::make_index_sequence<size>{});
}

// Async execute
// TODO(OpApi): sync exec and move to acl_util run.
#define EXEC_NPU_CMD(aclnn_api, acl_stream, ...)                                                             \
  do {                                                                                                       \
    static const auto get_workspace_size_func_ptr = GetOpApiFunc(#aclnn_api "GetWorkspaceSize");             \
    if (get_workspace_size_func_ptr == nullptr) {                                                            \
      MS_LOG(EXCEPTION) << #aclnn_api "GetWorkspaceSize"                                                     \
                        << " not in " << GetOpApiLibName() << ", please check!";                             \
    }                                                                                                        \
    static const auto op_api_func = GetOpApiFunc(#aclnn_api);                                                \
    if (op_api_func == nullptr) {                                                                            \
      MS_LOG(EXCEPTION) << #aclnn_api << " not in " << GetOpApiLibName() << ", please check!";               \
    }                                                                                                        \
    uint64_t workspace_size = 0;                                                                             \
    uint64_t *workspace_size_addr = &workspace_size;                                                         \
    aclOpExecutor *executor = nullptr;                                                                       \
    aclOpExecutor **executor_addr = &executor;                                                               \
    auto init_mem_func = OpApiDefaultResource::GetInstance().init_mem_func();                                \
    auto uninit_mem_func = OpApiDefaultResource::GetInstance().uninit_mem_func();                            \
    if (init_mem_func) {                                                                                     \
      init_mem_func(nullptr, false);                                                                         \
    }                                                                                                        \
    auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                   \
    static auto get_workspace_size_func = ConvertToOpApiFunc(converted_params, get_workspace_size_func_ptr); \
    auto workspace_status = call(get_workspace_size_func, converted_params);                                 \
    if (workspace_status != 0) {                                                                             \
      MS_LOG(EXCEPTION) << #aclnn_api << " not in " << GetOpApiLibName() << ", please check!";               \
    }                                                                                                        \
    void *workspace_addr = nullptr;                                                                          \
    if (workspace_size != 0) {                                                                               \
      workspace_addr = OpApiDefaultResource::GetInstance().AllocWorkspace(workspace_size);                   \
    }                                                                                                        \
    auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]() -> int {      \
      using RunApiFunc = int (*)(void *, uint64_t, aclOpExecutor *, const aclrtStream);                      \
      auto run_api_func = reinterpret_cast<RunApiFunc>(op_api_func);                                         \
      auto api_ret = run_api_func(workspace_addr, workspace_size, executor, acl_stream);                     \
      if (api_ret != 0) {                                                                                    \
        MS_LOG(EXCEPTION) << "call " #aclnn_api " failed, detail:" << aclGetRecentErrMsg();                  \
      }                                                                                                      \
      ReleaseConvertTypes(converted_params);                                                                 \
      auto release_mem_func = OpApiDefaultResource::GetInstance().release_mem_func();                        \
      if (release_mem_func) {                                                                                \
        release_mem_func(nullptr, false);                                                                    \
      }                                                                                                      \
      return api_ret;                                                                                        \
    };                                                                                                       \
    auto call_ret = acl_call();                                                                              \
    if (call_ret != 0) {                                                                                     \
      MS_LOG(EXCEPTION) << #aclnn_api << " exec failed!" << aclGetRecentErrMsg();                            \
    }                                                                                                        \
    if (uninit_mem_func) {                                                                                   \
      uninit_mem_func(nullptr, false);                                                                       \
    }                                                                                                        \
  } while (false)
}  // namespace mindspore::transform

#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_EXEC_H_
