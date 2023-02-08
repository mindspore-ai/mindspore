/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include <vector>
#include <unordered_map>
#ifdef BFC_MEMORY
#include "src/extendrt/dynamic_mem_allocator.h"
#else
#include "src/litert/inner_allocator.h"
#endif
#include "thread/threadpool.h"
#include "nnacl/op_base.h"
#ifdef ENABLE_ARM
#include "src/litert/cpu_info.h"
#endif
#include "include/lite_types.h"
#include "src/litert/infer_manager.h"

namespace mindspore::lite {
typedef struct CpuDeviceInfo {
  bool enable_float16_ = false; /**< prior enable float16 inference */
  CpuBindMode cpu_bind_mode_ = MID_CPU;
} CpuDeviceInfo;

typedef struct GpuDeviceInfo {
  bool enable_float16_ = false; /**< prior enable float16 inference */
  uint32_t gpu_device_id_ = 0;
  int rank_id_ = 0;
  int group_size_ = 0;
  bool enable_gl_texture_ = false; /**<enable sharing OpenGL texture with OpenCL */
  void *gl_context_ = nullptr;
  void *gl_display_ = nullptr;
} GpuDeviceInfo;

typedef struct NpuDeviceInfo {
  bool enable_float16_ = false; /**< prior enable float16 inference */
  int frequency_ = 3; /**< npu frequency inference, low 1, medium 2, high 3, extreme 4, other values will be set to 3 */
} NpuDeviceInfo;

typedef struct AscendDeviceInfo {
  uint32_t device_id_ = 0;
  std::string batch_size_;
  std::string image_size_;
} AscendDeviceInfo;

struct DeviceInfo {
  CpuDeviceInfo cpu_device_info_;
  GpuDeviceInfo gpu_device_info_;
  NpuDeviceInfo npu_device_info_;
  AscendDeviceInfo ascend_device_info_;
};

struct DeviceContext {
  DeviceType device_type_ = DT_CPU;
  DeviceInfo device_info_;
  std::string provider_{};
  std::string provider_device_{};
  AllocatorPtr allocator_ = nullptr;
};

typedef struct InstructionsContext {
  // Instructions should be checked in the beginning.
  bool support_fp16 = false;
  bool support_sdot = false;
  bool support_sse = false;
  bool support_avx512 = false;
} InstructionsContext;

struct InnerContext {
 public:
  InnerContext();
  virtual ~InnerContext();
  int Init();
  bool IsCpuFloat16Enabled() const;
  bool IsGpuFloat16Enabled() const;
  bool IsNpuFloat16Enabled() const;
  bool IsGLTextureEnabled() const;
  bool IsDeviceTypeEnabled(DeviceType type) const;
  bool IsProviderEnabled() const;
  int GetDelegateMode() const;
  std::set<std::string> GetProviders() const;
  DeviceInfo GetDeviceInfo(DeviceType type) const;
  std::set<void *> GetLinkInfo(void *pre) const;
  std::unordered_map<void *, std::set<void *>> GetAllLinkInfo() const;
  void SetLinkInfo(void *pre, void *suc);
  void SetAllLinkInfo(const std::unordered_map<void *, std::set<void *>> &all_link_info);
  void ReplaceLinkInfoReceiverWithNewOne(void *new_receiver, void *old_receiver);
  void ReplaceLinkInfoSenderWithNewOne(void *new_sender, void *old_sender);
  inline void SetBindRunnerId(std::string runner_id) { runner_id_ = runner_id; }
  inline void set_infer_checker(const InferChecker checker) { infer_checker_ = checker; }
  inline const InferChecker get_infer_checker() const { return infer_checker_; }

  std::string vendor_name_;
  InstructionsContext instructions_ctx_;
  int thread_num_ = 2; /**< thread number config for thread pool */
  int inter_op_parallel_num_ = 1;
  bool enable_parallel_ = false;
  std::vector<int> affinity_core_list_; /**< explicitly specify the core to be bound. priority use affinity core list */
  AllocatorPtr allocator = nullptr;
  std::vector<DeviceContext> device_list_ = {{DT_CPU, {{false, MID_CPU}}}};
  int delegate_mode_ = 0;
  DelegatePtr delegate = nullptr;
  bool float_mode = false; /**< convert full quant model to float model */

  bool device_and_pkg_support_fp16_ = false;
  ThreadPool *thread_pool_ = nullptr;
  InferChecker infer_checker_{InferCheckerOutput};
  // key is the precursor tensor's pointer, value is the group of successors' pointer.
  std::unordered_map<void *, std::set<void *>> link_info_{};

 private:
  int IsValid();
  bool IsAllDeviceTypeValid() const;
  bool IsCpuBindModeInvalid() const;
  int CreateThreadPool();
  void InitExperimentalExecEnv();

  std::string runner_id_;
  BindMode bind_mode_{Power_NoBind};
  size_t actor_thread_num_{0};
};

int ParallelLaunch(const InnerContext *context, const Func &func, Content content, int task_num);
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_RUNTIME_INNER_CONTEXT_H_
