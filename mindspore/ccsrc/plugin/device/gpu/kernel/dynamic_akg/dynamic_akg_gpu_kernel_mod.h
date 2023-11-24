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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_GPU_DYNAMIC_AKG_GPU_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_GPU_DYNAMIC_AKG_GPU_KERNEL_MOD_H_
#include <cuda.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <memory>
#include <utility>
#include "plugin/device/gpu/kernel/dynamic_akg/dynamic_utils.h"
#include "kernel/kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_mod.h"
#include "plugin/device/gpu/kernel/akg/akg_gpu_kernel_mod.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
constexpr auto kMappingUpdated = "updated";
constexpr auto kBlockIdxX = "blockIdx.x";
constexpr auto kBlockIdxY = "blockIdx.y";
constexpr auto kBlockIdxZ = "blockIdx.z";
constexpr auto kThreadIdxX = "threadIdx.x";
constexpr auto kThreadIdxY = "threadIdx.y";
constexpr auto kThreadIdxZ = "threadIdx.z";

class DynamicAkgGpuKernelManager {
 public:
  DynamicAkgGpuKernelManager();
  virtual ~DynamicAkgGpuKernelManager() {
    for (auto iter = infotable_.begin(); iter != infotable_.end(); ++iter) {
      CUresult ret = cuModuleUnload(iter->second->module_);
      if (ret != CUDA_SUCCESS && ret != CUDA_ERROR_DEINITIALIZED) {
        const char *msg = nullptr;
        cuGetErrorName(ret, &msg);
        MS_LOG(ERROR) << "Unload GPU Module failed. cuModuleUnload error message: " << msg;
      }
    }
  }
  CUresult GetCUResult(const char *kernel_content, bool force_reload, std::vector<uint32_t> *thread_info,
                       CUfunction *func, const std::string kernel_name);
  CUresult GetFunction(const KernelPackPtr &kernel_pack, bool force_reload, std::vector<uint32_t> *thread_info,
                       CUfunction *func, const std::string kernel_name);

 private:
  std::unordered_map<std::string, GpuKernelMetaPtr> infotable_;
};
using DynamicAkgGpuKernelManagerPtr = std::shared_ptr<DynamicAkgGpuKernelManager>;

class DynamicAkgGpuKernelMod : public GpuKernelMod {
 public:
  explicit DynamicAkgGpuKernelMod(const KernelPackPtr &kernel_pack);
  virtual ~DynamicAkgGpuKernelMod() {}

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return true;
  };

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  void Initialize();
  void CheckJsonParsed();
  void InitJsonShapeInformation();
  bool CheckJsonValueFormat(const std::string key) {
    auto value = parsed_js_[key];
    return (value.is_array() && value.size() == 2 && value[0].is_string() && value[1].is_number());
  }
  void InitJsonMappingInformation();
  void InitBeforeMapping();
  void UpdateDynamicShapeTilingInfo();
  void UpdateDynamicShapeMappingInfo();
  void InitAkgKernelImpls();
  void UpdateStaticShapeMappingInfo();
  void UpdateShapeList(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  void SetKernelDynamicStatus(bool is_dynamic) { is_dynamic_ = is_dynamic; }

  enum KernelModType GetKernelModType() const override { return KernelModType::DynamicAkgCpuKernelMod; }

  static DynamicAkgGpuKernelManagerPtr kernel_manager_;
  std::vector<KernelAttr> GetOpSupport() override { return {}; }
  std::string kernel_name_;

 private:
  KernelPackPtr kernel_pack_;
  std::vector<uint32_t> thread_info_;
  CUfunction kernel_addr_{nullptr};
  bool is_dynamic_{false};
  std::vector<std::vector<int64_t>> shape_list_;
  nlohmann::json parsed_js_;
  std::vector<int> arg_size_vec_;

  AkgKernelImplInfoPtr kernel_impl_;
  std::unordered_map<std::string, AkgKernelImplInfoPtr> kernel_map_;
  AkgKernelImplInfoPtr SelectKernelImpl();
};

class DynamicAkgGpuKernelModDebug : public DynamicAkgGpuKernelMod {
 public:
  explicit DynamicAkgGpuKernelModDebug(const KernelPackPtr &kernel_pack) : DynamicAkgGpuKernelMod(kernel_pack) {}
  virtual ~DynamicAkgGpuKernelModDebug() {}
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    auto ptr = reinterpret_cast<CUstream>(stream_ptr);
    CUresult before_launch = cuStreamSynchronize(ptr);
    const char *msg = nullptr;
    if (before_launch != CUDA_SUCCESS) {
      cuGetErrorName(before_launch, &msg);
      MS_LOG(ERROR) << "before_launch sycn failed, Kernel name is : " << kernel_name_ << ", Error message: " << msg;
    }
    auto result = DynamicAkgGpuKernelMod::Launch(inputs, workspace, outputs, stream_ptr);
    CUresult after_launch = cuStreamSynchronize(ptr);
    if (after_launch != CUDA_SUCCESS) {
      cuGetErrorName(after_launch, &msg);
      MS_LOG(ERROR) << "after_launch sycn failed, Kernel name is : " << kernel_name_ << ", Error message: " << msg;
    }
    return result;
  }
};
using DynamicAkgGpuKernelModPtr = std::shared_ptr<DynamicAkgGpuKernelMod>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_GPU_DYNAMIC_AKG_GPU_KERNEL_MOD_H_
