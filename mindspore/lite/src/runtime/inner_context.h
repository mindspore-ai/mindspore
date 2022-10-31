/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_INNER_CONTEXT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_INNER_CONTEXT_H_
#include <set>
#include <string>
#include <unordered_map>
#include "include/context.h"
#ifdef BFC_MEMORY
#include "src/runtime/dynamic_mem_allocator.h"
#else
#include "src/runtime/inner_allocator.h"
#endif
#include "thread/threadpool.h"
#include "nnacl/op_base.h"
#ifdef ENABLE_ARM
#include "src/runtime/cpu_info.h"
#endif

namespace mindspore::lite {
#ifdef ENABLE_MINDRT
constexpr int kDefaultParallelNum = 2;
#endif
struct InnerContext : public Context {
 public:
  InnerContext() { InitDeviceFp16(); }

  explicit InnerContext(const Context *context);

  int Init();

  bool IsCpuFloat16Enabled() const;

  bool IsGpuFloat16Enabled() const;

  bool IsGLTextureEnabled() const;

  bool IsDeviceTypeEnabled(DeviceType type) const;

  bool IsProviderEnabled() const;

  std::set<std::string> GetProviders() const;

  DeviceInfo GetDeviceInfo(DeviceType type) const;

  int IsValid();

  ThreadPool *thread_pool() const;

  virtual ~InnerContext();

  bool device_and_pkg_support_fp16() const;

  std::set<void *> GetLinkInfo(void *pre) const;

  std::unordered_map<void *, std::set<void *>> GetAllLinkInfo() const;

  void SetLinkInfo(void *pre, void *suc);

  void SetAllLinkInfo(const std::unordered_map<void *, std::set<void *>> &all_link_info);

  void ReplaceLinkInfoReceiverWithNewOne(void *new_receiver, void *old_receiver);

  void ReplaceLinkInfoSenderWithNewOne(void *new_sender, void *old_sender);

 private:
  bool IsAllDeviceTypeValid() const;

  bool IsCpuBindModeInvalid() const;

  void SetContextDevice(const Context *context);

  void InitDeviceFp16();

  int CreateThreadPool();

  void InitExperimentalExecEnv();

  bool device_and_pkg_support_fp16_ = false;

#ifdef BFC_MEMORY
  int node_id_ = -1;
#endif

  BindMode bind_mode_{Power_NoBind};
  size_t actor_thread_num_{0};
  ThreadPool *thread_pool_{nullptr};

  // key is the precursor tensor's pointer, value is the group of successors' pointer.
  std::unordered_map<void *, std::set<void *>> link_info_{};
};

int ParallelLaunch(const Context *context, const Func &func, Content content, int task_num);
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_RUNTIME_INNER_CONTEXT_H_
