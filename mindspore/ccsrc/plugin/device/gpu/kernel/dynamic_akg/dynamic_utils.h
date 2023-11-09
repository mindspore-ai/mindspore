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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_GPU_DYNAMIC_AKG_GPU_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_GPU_DYNAMIC_AKG_GPU_UTILS_H_
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
struct PairHash {
  template <typename T>
  size_t operator()(const std::pair<T, T> &p) const {
    auto h1 = std::hash<T>{}(p.first);
    auto h2 = std::hash<T>{}(p.second);
    return h1 ^ h2;
  }
};
enum GpuMemScope {
  // global
  MEM_SCOPE_GM = 0,
  // gpu
  MEM_SCOPE_SHARED,
  MEM_SCOPE_LOCAL,
  // end
  MEM_SCOPE_BULK,
};
class GpuInfo {
 public:
  GpuInfo(const GpuInfo &) = delete;
  GpuInfo &operator=(const GpuInfo &) = delete;
  ~GpuInfo() {}
  static GpuInfo &GetInstance(const std::string &device_type) {
    static GpuInfo hardware_info(device_type);
    return hardware_info;
  }

  int64_t GetMemoryLimitInScope(int scope_idx) {
    if (scope_idx > MEM_SCOPE_BULK) {
      MS_EXCEPTION(RuntimeError) << "scope_idx should be less than " << MEM_SCOPE_BULK << ", but got " << scope_idx
                                 << "\n";
      return 0;
    }
    return gpuMemLimit[scope_idx];
  }

  int GetWarpSizes() { return warpSize; }
  int GetNumSm() { return numSm; }
  std::pair<int, int> GetActiveBlocksPerSm() { return activeBlocksPerSm; }
  std::pair<int, int> GetThreadCoef() { return threadCoef; }
  int GetMinElemForIoBound() { return minElemForIoBound; }
  int GetMaxElemForIoBound() { return maxElemForIoBound; }
  int GetTotalAvailableBlocks() { return totalAvailableBlocks; }
  std::vector<int64_t> GetMaxGrids() { return {maxGridX, maxGridYZ, maxGridYZ}; }
  std::vector<int64_t> GetMaxBlocks() { return {maxBlockXY, maxBlockXY, maxBlockZ}; }

 private:
  explicit GpuInfo(const std::string &device_type) {
    InitGpuMemoryLimit(device_type);
    InitGpuComputeCapability(device_type);
  }
  int64_t gpuMemLimit[MEM_SCOPE_BULK]{0};
  int numSm{80};
  int warpSize{32};
  int minElemForIoBound{2};
  int maxElemForIoBound{32};
  int totalAvailableBlocks{1024};
  std::pair<int, int> threadCoef{8, 16};
  std::pair<int, int> activeBlocksPerSm{5, 6};
  int64_t maxGridX = 2147483647;
  int64_t maxGridYZ = 65535;
  int64_t maxBlockXY = 1024;
  int64_t maxBlockZ = 64;

  void InitGpuMemoryLimit(const std::string &device_type);
  void InitGpuComputeCapability(const std::string &device_type);
};

struct MappingInfo {
  uint32_t total_alloc_grid{1};
  uint32_t total_alloc_block{1};
  uint32_t curr_grid[3]{1, 1, 1};
  uint32_t curr_block[3]{1, 1, 1};
  int64_t proposal_grid{1};
  int64_t proposal_block{1};
  std::vector<size_t> solve_order_id;
  uint32_t GetMapLimit(size_t id, const std::string &device_target);
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
  int mark{999};                   // used to solve dynamic tiling for specific algorithms, default is unknown{999}

  // Init in resize
  int64_t upper_bound{-1};
  int outer_map_id{-1};
  int curr_map_id{-1};
  int64_t runtime_size{-1};

  std::string ArgIndexKey() { return "argIndex"; }
  std::string ExprKey() { return "expr"; }
  std::string MarkKey() { return "mark"; }
  std::string MapDimKey() { return "mapDim"; }
  std::string MappingKey() { return "mapping"; }
  std::string PrimeKey() { return "prime"; }
  std::string ToString() {
    std::string res = "[RuntimeVar " + std::to_string(prime) + "]";
    res += "  -> " + mapping + "." + mapDim + " at " + std::to_string(argIndex) + " input\n";
    res += "  -> expr: " + expr + "\n";
    res += "  -> mark: " + std::to_string(mark) + "\n";
    res += "  -> upper bound " + std::to_string(upper_bound) + "; curr_map_id " + std::to_string(curr_map_id) +
           "; outer_map_id " + std::to_string(outer_map_id) + "\n";
    res += "  -> runtime_size " + std::to_string(runtime_size) + "\n";
    return res;
  }

  static std::unordered_map<std::string, int> mark_table_;
};
using RuntimeVarPtr = std::shared_ptr<RuntimeVar>;
using RuntimeVarsMap = std::map<int, RuntimeVarPtr>;

enum AkgKernelImplType {
  DEFAULT = 0,
  STATIC_TILE,
  DYNYAMIC_TILE,
};

using LocVector = std::vector<std::pair<size_t, size_t>>;
class AkgKernelImplInfo {
 public:
  AkgKernelImplInfo(const std::string &kernel_name, nlohmann::json json);
  virtual ~AkgKernelImplInfo() = default;

  virtual void Init() {}
  virtual void Resize() {}

  // update each time
  std::vector<uint32_t> thread_info_;
  std::unordered_map<size_t, LocVector> unknown_map_loc_;
  std::unordered_set<size_t> solved_map_loc_;
  std::vector<int> arg_size_vec_;
  MappingInfo curr_mapping_info_;
  std::vector<std::vector<int64_t>> shape_list_;
  std::vector<std::vector<int64_t>> device_shape_list_;
  int64_t problem_size_ = 1;

  // no change
  std::string kernel_name_;
  nlohmann::json parsed_js_;
  AkgKernelImplType kernel_type_{AkgKernelImplType::DEFAULT};
  std::vector<uint32_t> init_mapping_;
  MappingInfo init_mapping_info_;
  RuntimeVarsMap runtime_vars_;
  std::vector<std::pair<uint32_t, RuntimeVarPtr>> sorted_runtime_vars_;
  std::unordered_map<uint32_t, std::pair<size_t, size_t>> local_upper_bound_;
  size_t max_shape_rank_ = 0;
  std::string device_target_;
  std::unordered_map<std::pair<size_t, size_t>, LocVector, PairHash> device_host_shape_loc_;
  std::unordered_map<std::string, std::pair<size_t, size_t>> host_loc_map_;
  std::unordered_map<int64_t, std::string> product_var_;   // map: prime -> symbol like `s0`
  std::unordered_map<std::string, int> axis_length_left_;  // update map: symbol axis -> total length of one axis
  std::unordered_map<int, std::vector<int>> related_values_;
  std::unordered_map<size_t, std::string> unknown_map_symbol_;
  std::unordered_map<uint32_t, std::string> local_upper_bound_symbol_;
  std::vector<std::string> map_arg_list_;

  int static_reduce_length_{1};
  std::vector<int> runtime_threads_order_;
  std::vector<std::pair<int, int>> template_tiling_order_;  // pair includes prime & mark
  std::unordered_map<int, int> prime_to_mapping_idx_;
  std::unordered_map<int, int> prime_to_mapping_dividend_;
  bool enable_atomic_{false};
  int dyn_algorithm_{0};
  static std::unordered_map<std::string, int> algo_to_int_;

  void preprocessDynamicReduceTiling();
  LocVector GetHostLocationVec(std::string symbol_expr, const size_t pure_num_flag);
  void InitJsonShapeInformation();
  void InitJsonMappingInformation();
  bool CheckJsonValueFormat(const std::string key) {
    auto value = parsed_js_[key];
    return (value.is_array() && value.size() == 2 && value[0].is_string() && value[1].is_number());
  }
  void GetDeviceArgSizeVec();
  void InitBeforeMapping();
  int64_t GetFoldedShape(const LocVector &host_loc_vec);
  void UpdateDynamicShapeMappingInfo();
};
using AkgKernelImplInfoPtr = std::shared_ptr<AkgKernelImplInfo>;

class DynamicTileImpl : public AkgKernelImplInfo {
 public:
  DynamicTileImpl(const std::string &kernel_name, nlohmann::json json) : AkgKernelImplInfo(kernel_name, json) {
    this->kernel_type_ = AkgKernelImplType::DYNYAMIC_TILE;
  }
  virtual ~DynamicTileImpl() = default;
  void Init() override {
    this->InitBeforeMapping();
    this->UpdateRuntimeVarUpperBound();
  }
  void Resize() override {
    this->InitBeforeMapping();
    this->GetDeviceArgSizeVec();
    this->UpdateDynamicShapeTilingInfo();
    this->UpdateDynamicShapeMappingInfo();
  }

 private:
  void UpdateDynamicShapeTilingInfo();
  void UpdateRuntimeVarUpperBound();
  void SolveDynamicReduction();
  void SolveDynamicTiling(size_t curr_id);
  int64_t TileSizeOpt(const RuntimeVarPtr &var, int64_t dyn_tile_size);
  void UpdateMapping(int curr_id, int64_t map_size, int64_t prime);
};

class StaticTileImpl : public AkgKernelImplInfo {
 public:
  StaticTileImpl(const std::string &kernel_name, nlohmann::json json) : AkgKernelImplInfo(kernel_name, json) {
    this->kernel_name_ = kernel_name;
    this->parsed_js_ = json;
    this->kernel_type_ = AkgKernelImplType::STATIC_TILE;
  }
  virtual ~StaticTileImpl() = default;
  void Init() override {
    this->InitBeforeMapping();
    this->UpdateDynamicShapeMappingInfo();
  }
  void Resize() override {
    this->InitBeforeMapping();
    this->GetDeviceArgSizeVec();
    this->UpdateDynamicShapeMappingInfo();
  }
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_GPU_DYNAMIC_AKG_GPU_UTILS_H_
