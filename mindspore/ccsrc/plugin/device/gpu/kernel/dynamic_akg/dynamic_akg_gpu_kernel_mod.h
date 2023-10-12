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
#include "kernel/kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_mod.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
struct GpuKernelMeta {
  CUfunction func_addr_;
  CUmodule module_;
  std::vector<uint32_t> thread_info_;
  GpuKernelMeta(CUfunction funcAddr, CUmodule module, const std::vector<uint32_t> &thread_info)
      : func_addr_(funcAddr), module_(module), thread_info_(thread_info) {}
};
using GpuKernelMetaPtr = std::shared_ptr<GpuKernelMeta>;

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

struct PairHash {
  template <typename T>
  size_t operator()(const std::pair<T, T> &p) const {
    auto h1 = std::hash<T>{}(p.first);
    auto h2 = std::hash<T>{}(p.second);
    return h1 ^ h2;
  }
};

struct MappingInfo {
  uint32_t total_alloc_grid{1};
  uint32_t total_alloc_block{1};
  uint32_t curr_grid[3]{1, 1, 1};
  uint32_t curr_block[3]{1, 1, 1};
  uint32_t max_grid[3]{2147483647, 65535, 65535};
  uint32_t max_block[3]{1024, 1024, 64};
  uint32_t max_total_block{1024};
  std::vector<size_t> solve_order_id;
  uint32_t GetMapLimit(size_t id);
  void UpdateCurrMapSize(size_t id, uint32_t map_size);
  std::string ToString() {
    std::string res;
    res += "total_alloc_grid: [";
    for (auto g : curr_grid) {
      res += std::to_string(g) + ", ";
    }
    res += "] = " + std::to_string(total_alloc_grid) + "; ";
    res += "total_alloc_block: [";
    for (auto b : curr_block) {
      res += std::to_string(b) + ", ";
    }
    res += "] = " + std::to_string(total_alloc_block) + "\n";
    return res;
  }
};

struct RuntimeVar {
 public:
  // Init from json
  int64_t prime;                   // prime is like a unique id for this var to speedup lower in pipeline
  int argIndex{-1};                // index in the func argument
  std::string mapping{"Default"};  // used for GPU mapping, can be chosen from [Grid, Block, Seq]
  std::string mapDim{""};          // used for GPU mapping, can be chosen from [x, y, z]
  std::string expr{""};            // used to solve dynamic tiling

  // Init in resize
  int64_t upper_bound{-1};
  int outer_map_id{-1};
  int curr_map_id{-1};
  int64_t runtime_size{-1};

  std::string ArgIndexKey() { return "argIndex"; }
  std::string ExprKey() { return "expr"; }
  std::string MapDimKey() { return "mapDim"; }
  std::string MappingKey() { return "mapping"; }
  std::string PrimeKey() { return "prime"; }
  std::string ToString() {
    std::string res = "[RuntimeVar " + std::to_string(prime) + "]";
    res += "  -> " + mapping + "." + mapDim + " at " + std::to_string(argIndex) + " input\n";
    res += "  -> expr: " + expr + "\n";
    res += "  -> upper bound " + std::to_string(upper_bound) + "; curr_map_id " + std::to_string(curr_map_id) +
           "; outer_map_id " + std::to_string(outer_map_id) + "\n";
    res += "  -> runtime_size " + std::to_string(runtime_size) + "\n";
    return res;
  }
};
using RuntimeVarPtr = std::shared_ptr<RuntimeVar>;
using RuntimeVarsMap = std::map<uint32_t, RuntimeVarPtr>;

class DynamicAkgGpuKernelMod : public GpuKernelMod {
 public:
  explicit DynamicAkgGpuKernelMod(const KernelPackPtr &kernel_pack);
  virtual ~DynamicAkgGpuKernelMod() {}

  bool Init(const BaseOperatorPtr & /* base_operator */, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    return true;
  };

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  void CheckJsonParsed();
  void InitJsonShapeInformation();
  void InitJsonMappingInformation();
  void InitBeforeMapping();
  void UpdateDynamicShapeTilingInfo();
  void UpdateDynamicShapeMappingInfo();
  void UpdateStaticShapeMappingInfo();
  void UpdateShapeList(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs);
  void GetDeviceArgSizeVec(const size_t inputs_num);
  void SetKernelDynamicStatus(bool is_dynamic) { is_dynamic_ = is_dynamic; }

  enum KernelModType GetKernelModType() const override { return KernelModType::DynamicAkgCpuKernelMod; }

  static DynamicAkgGpuKernelManagerPtr kernel_manager_;
  std::vector<KernelAttr> GetOpSupport() override { return {}; }
  std::string kernel_name_;

 private:
  KernelPackPtr kernel_pack_;
  std::vector<uint32_t> thread_info_;
  std::vector<uint32_t> init_mapping_;
  CUfunction kernel_addr_{nullptr};
  bool is_dynamic_{false};
  std::vector<std::vector<int64_t>> shape_list_;
  std::vector<std::vector<int64_t>> device_shape_list_;
  nlohmann::json parsed_js_;
  std::vector<int> arg_size_vec_;
  std::unordered_map<std::string, std::pair<size_t, size_t>> host_loc_map_;
  std::unordered_map<size_t, std::pair<size_t, size_t>> unknown_map_loc_;
  bool json_shape_updated_{false};
  std::unordered_map<std::pair<size_t, size_t>, std::pair<size_t, size_t>, PairHash> device_host_shape_loc_;

  // Used to solve dynamic tiling size during resize
  void UpdateRuntimeVarUpperBound();
  void SolveDynamicTiling(size_t curr_id);
  size_t max_shape_rank_ = 0;
  std::unordered_set<size_t> solved_map_loc_;
  RuntimeVarsMap runtime_vars_;
  std::vector<std::pair<uint32_t, RuntimeVarPtr>> sorted_runtime_vars_;
  std::unordered_map<uint32_t, std::pair<size_t, size_t>> local_upper_bound_;
  MappingInfo init_mapping_info_;
  MappingInfo curr_mapping_info_;
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
