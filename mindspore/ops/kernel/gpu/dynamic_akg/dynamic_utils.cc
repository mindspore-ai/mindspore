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

#include "kernel/gpu/dynamic_akg/dynamic_utils.h"
#include <fstream>
#include <algorithm>
#include <map>
#include "nlohmann/json.hpp"
#include "utils/ms_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "include/utils/common.h"
#include "kernel/gpu/gpu_common.h"

namespace mindspore {
namespace kernel {
using std::fstream;
using std::string;
using std::unordered_map;
using std::vector;
namespace {
constexpr auto kV100Device = "v100";
constexpr auto kA100Device = "a100";
constexpr auto kSharedMem = "shared_mem";
constexpr auto kRegMem = "reg_mem";
constexpr auto kLocalPrefix = "Seq";
constexpr auto kHostShapes = "hostShapes";
constexpr auto kDeviceShapes = "deviceShapes";
constexpr auto kRuntimeVars = "runtimeVars";
constexpr auto kTargetInfo = "targetInfo";
constexpr auto kSupportInfo = "SupportInfo";
constexpr auto kReduceSizeStatic = "ReduceSizeStatic";
constexpr auto kDynAlgorithm = "DynAlgorithm";
constexpr auto kEnableAtomic = "EnableAtomic";
constexpr auto kMultiply = "*";

constexpr auto kBlockIdxX = "blockIdx.x";
constexpr auto kBlockIdxY = "blockIdx.y";
constexpr auto kBlockIdxZ = "blockIdx.z";
constexpr auto kThreadIdxX = "threadIdx.x";
constexpr auto kThreadIdxY = "threadIdx.y";
constexpr auto kThreadIdxZ = "threadIdx.z";
constexpr int64_t kRemove = -100000;
constexpr int64_t kKeep = -99999;
const int AKG_KERNEL_MOD_BX_IDX = 0;
const int AKG_KERNEL_MOD_BY_IDX = 1;
const int AKG_KERNEL_MOD_BZ_IDX = 2;
const int AKG_KERNEL_MOD_TX_IDX = 3;
const int AKG_KERNEL_MOD_TY_IDX = 4;
const int AKG_KERNEL_MOD_TZ_IDX = 5;
const int WARP_SIZE = 32;
const int WARP_ALLOC_GRAN = 4;
const int ELEM_BEST_GRID_SIZE = 512;
const int MAX_THREAD_NUM = 1024;
const int k48KB = 49152;
const int k64KB = 65536;
const int kNum80 = 80;
const int kNum108 = 108;
const int kNum512 = 512;
const int kNum1024 = 1024;
const int kNum2 = 2;
const int kNum16 = 16;
const int kNum32 = 32;
const int kNum64 = 64;
const int kNum256 = 256;

const int AKG_DYN_ALGO_REDUCE_X = 10;
const int AKG_DYN_ALGO_REDUCE_Y = 11;
const int AKG_DYN_ALGO_REDUCE_SMALL = 12;
const int AKG_DYN_MARK_THREAD_LOWER_BOUND = 10;
const int AKG_DYN_MARK_THREAD_UPPER_BOUND = 20;
const int AKG_DYN_MARK_SEQ_LOWER_BOUND = 20;
const int AKG_DYN_MARK_SEQ_UPPER_BOUND = 30;
const int AKG_DYN_MARK_ONE = 30;
const int AKG_DYN_MARK_PRODUCT = 40;
const int AKG_DYN_MARK_UNKNOWN = 999;
}  // namespace
struct RuntimeVarCompare {
  bool operator()(const std::pair<uint32_t, RuntimeVarPtr> &a, const std::pair<uint32_t, RuntimeVarPtr> &b) const {
    return a.second->argIndex < b.second->argIndex;
  }
};

std::unordered_map<std::string, int> AkgKernelImplInfo::algo_to_int_ = {{"reduce-x", AKG_DYN_ALGO_REDUCE_X},
                                                                        {"reduce-y", AKG_DYN_ALGO_REDUCE_Y},
                                                                        {"reduce-small", AKG_DYN_ALGO_REDUCE_SMALL}};

std::unordered_map<std::string, int> RuntimeVar::mark_table_ = {{"unknown", 999},
                                                                {"reduce-thread-last", 10},
                                                                {"reduce-thread", 11},
                                                                {"parallel-thread-last", 12},
                                                                {"parallel-thread", 13},
                                                                {"reduce-y-seq", 20},
                                                                {"reduce-x-seq", 21},
                                                                {"parallel-seq", 22},
                                                                {"1", 30},
                                                                {"product", 40}};

inline void GetProperReduceXConfig(int redSize, const bool &useAtomicFlag, int *block_num, int *thread_num,
                                   int *seq_num) {
  const int acc_num = 4;
  *block_num = useAtomicFlag ? (redSize - 1) / MAX_THREAD_NUM + 1 : 1;
  redSize = useAtomicFlag ? (redSize - 1) / (*block_num) + 1 : redSize;
  (*thread_num) = redSize < kNum32 ? redSize : kNum32;
  while ((*thread_num) * kNum2 * acc_num <= redSize && (*thread_num) <= MAX_THREAD_NUM) {
    (*thread_num) *= kNum2;
  }
  (*seq_num) = (redSize - 1) / ((*thread_num) * (*block_num)) + 1;
}

inline void GetProperReduceYConfig(const int &redSize, const bool &useAtomicFlag, int *block_num, int *seq_num) {
  *block_num = 1;
  *seq_num = redSize;
  if (useAtomicFlag) {
    if (redSize < kNum256) {
      *seq_num = kNum16;
    } else if (redSize < kNum1024) {
      *seq_num = kNum32;
    } else {
      *seq_num = kNum64;
    }
    *block_num = (redSize - 1) / (*seq_num) + 1;
  }
}

inline void GetReduceTileSize(const int &mark, const int &upper_bound, const int &proper_thread, const int &proper_seq,
                              int *tile_size) {
  if (AKG_DYN_MARK_THREAD_LOWER_BOUND <= mark && mark < AKG_DYN_MARK_THREAD_UPPER_BOUND) {
    *tile_size = std::min(upper_bound, proper_thread);
  } else if (AKG_DYN_MARK_SEQ_LOWER_BOUND <= mark && mark < AKG_DYN_MARK_SEQ_UPPER_BOUND) {
    *tile_size = std::min(upper_bound, proper_seq);
  } else if (mark == AKG_DYN_MARK_ONE) {
    *tile_size = 1;
  } else {
    *tile_size = -1;
  }
}

static std::pair<int, int> GetProposalParallelSize(int problemSize, const std::string &device_target) {
  GpuInfo &gpu_info = GpuInfo::GetInstance(device_target);
  int proposedGrid = 1;
  int proposedBlock = 1;
  auto numSm = gpu_info.GetNumSm();
  auto threadCoef = gpu_info.GetThreadCoef();
  auto warpSizes = gpu_info.GetWarpSizes();
  auto activeBlocksPerSm = gpu_info.GetActiveBlocksPerSm();
  auto totalBlocks = gpu_info.GetTotalAvailableBlocks();
  if (problemSize <= warpSizes) {
    proposedBlock = warpSizes;
  } else if (problemSize <= warpSizes * numSm) {
    proposedBlock = warpSizes;
    proposedGrid = numSm;
  } else if (problemSize <= warpSizes * threadCoef.first * numSm * activeBlocksPerSm.first) {
    proposedBlock = warpSizes * threadCoef.first;
    proposedGrid = numSm * activeBlocksPerSm.first;
  } else if (problemSize <= warpSizes * threadCoef.second * numSm * activeBlocksPerSm.second) {
    proposedBlock = warpSizes * threadCoef.second;
    proposedGrid = numSm * activeBlocksPerSm.second;
  } else if (problemSize <= warpSizes * threadCoef.second * numSm * activeBlocksPerSm.second * numSm) {
    proposedBlock = totalBlocks;
    proposedGrid = numSm * activeBlocksPerSm.second;
  } else {
    // extremely large shape
    proposedBlock = totalBlocks;
    proposedGrid = numSm * activeBlocksPerSm.second * kNum2;
  }
  return std::make_pair(proposedGrid, proposedBlock);
}

uint32_t MappingInfo::GetMapLimit(size_t id, const std::string &device_target) {
  if (id < AKG_KERNEL_MOD_BX_IDX || id > AKG_KERNEL_MOD_TZ_IDX) {
    MS_EXCEPTION(RuntimeError) << "Map id should be in range [" << AKG_KERNEL_MOD_BX_IDX << ", "
                               << AKG_KERNEL_MOD_TZ_IDX << "], but got " << id;
  }
  GpuInfo &gpu_info = GpuInfo::GetInstance(device_target);
  if (id >= AKG_KERNEL_MOD_TX_IDX) {
    auto max_block = gpu_info.GetMaxBlocks();
    return std::min<uint32_t>(max_block[id - AKG_KERNEL_MOD_TX_IDX],
                              gpu_info.GetTotalAvailableBlocks() / total_alloc_block);
  } else {
    auto max_grid = gpu_info.GetMaxGrids();
    return max_grid[id];
  }
}

void MappingInfo::UpdateCurrMapSize(size_t id, uint32_t map_size) {
  if (id < AKG_KERNEL_MOD_BX_IDX || id > AKG_KERNEL_MOD_TZ_IDX) {
    MS_EXCEPTION(RuntimeError) << "Map id should be in range [" << AKG_KERNEL_MOD_BX_IDX << ", "
                               << AKG_KERNEL_MOD_TZ_IDX << "], but got " << id;
  }
  if (id >= AKG_KERNEL_MOD_TX_IDX) {
    curr_block[id - AKG_KERNEL_MOD_TX_IDX] = map_size;
    total_alloc_block *= map_size;
  } else {
    curr_grid[id] = map_size;
    total_alloc_grid *= map_size;
  }
}

void GpuInfo::InitGpuMemoryLimit(const std::string &device_type) {
  auto CollectLimit = [this, &device_type](const std::string &scope, GpuMemScope mem) {
    if (device_type == kV100Device) {
      if (scope == kSharedMem) {
        gpuMemLimit[mem] = k48KB;
      } else if (scope == kRegMem) {
        gpuMemLimit[mem] = k64KB;
      }
    } else if (device_type == kA100Device) {
      if (scope == kSharedMem) {
        gpuMemLimit[mem] = k64KB;
      } else if (scope == kRegMem) {
        gpuMemLimit[mem] = k64KB;
      }
    }
  };
  CollectLimit(kSharedMem, MEM_SCOPE_SHARED);
  CollectLimit(kRegMem, MEM_SCOPE_LOCAL);
  gpuMemLimit[MEM_SCOPE_GM] = 0;
}

void GpuInfo::InitGpuComputeCapability(const std::string &device_type) {
  if (device_type == kV100Device) {
    numSm = kNum80;
    totalAvailableBlocks = kNum512;
  } else if (device_type == kA100Device) {
    numSm = kNum108;
    totalAvailableBlocks = kNum1024;
  }
}

AkgKernelImplInfo::AkgKernelImplInfo(const std::string &kernel_name, nlohmann::json json) {
  kernel_name_ = kernel_name;
  parsed_js_ = json;
  device_target_ = parsed_js_[kTargetInfo];
  if (device_target_.empty()) {
    device_target_ = kV100Device;
  }
}

LocVector AkgKernelImplInfo::GetHostLocationVec(std::string symbol_expr, const size_t pure_num_flag) {
  std::string delimiter = kMultiply;
  std::vector<std::string> symbol_vec;
  if (symbol_expr.find(delimiter) != std::string::npos) {
    // multiplication expr after folding like 's0*1024*s1'
    size_t pos_start = 0;
    size_t pos_end;
    size_t delim_len = 1;
    std::string symbol;
    // split each symbol or number and store in a list
    while ((pos_end = symbol_expr.find(delimiter, pos_start)) != std::string::npos) {
      symbol = symbol_expr.substr(pos_start, pos_end - pos_start);
      pos_start = pos_end + delim_len;
      (void)symbol_vec.emplace_back(symbol);
    }
    (void)symbol_vec.emplace_back(symbol_expr.substr(pos_start));
  } else {
    (void)symbol_vec.emplace_back(symbol_expr);
  }

  LocVector host_loc_vec;
  for (auto symbol : symbol_vec) {
    if (std::all_of(symbol.begin(), symbol.end(), [](char c) { return std::isdigit(c); })) {
      // for number '32', save a pair of <M, number>, where M = num of inputs + outputs
      // M must be greater than any host_loc index, so it can be a flag for pure numbers
      auto number_pair = std::make_pair(pure_num_flag, IntToSize(std::stoi(symbol)));
      (void)host_loc_vec.emplace_back(number_pair);
    } else if (host_loc_map_.find(symbol) != host_loc_map_.end()) {
      // for symbol 's0', save its location in host shape as a pair of <i, j>
      (void)host_loc_vec.emplace_back(host_loc_map_[symbol]);
    } else {
      MS_EXCEPTION(RuntimeError) << "For " << kernel_name_ << ", symbol '" << symbol
                                 << "' of device shape is not in host shape.";
    }
  }
  return host_loc_vec;
}

void AkgKernelImplInfo::InitJsonShapeInformation() {
  // Initialize device shape list using -1 for unknown dims
  // Record map <device_loc, symbol>
  unordered_map<std::pair<size_t, size_t>, string, PairHash> device_loc_map;
  for (size_t i = 0; i < parsed_js_[kDeviceShapes].size(); i++) {
    vector<int64_t> device_tensor_shape;
    auto device_tensor_rank = parsed_js_[kDeviceShapes][i].size();
    max_shape_rank_ = std::max<size_t>(max_shape_rank_, device_tensor_rank);

    for (size_t j = 0; j < device_tensor_rank; j++) {
      string device_shape_str = parsed_js_[kDeviceShapes][i][j];
      if (std::all_of(device_shape_str.begin(), device_shape_str.end(), [](char c) { return std::isdigit(c); })) {
        (void)device_tensor_shape.emplace_back(std::stoi(device_shape_str));
      } else {
        (void)device_tensor_shape.emplace_back(-1);
        auto device_loc = std::make_pair(i, j);
        device_loc_map[device_loc] = device_shape_str;
      }
    }

    (void)device_shape_list_.emplace_back(device_tensor_shape);
  }
  // Record map <symbol, host_loc>
  for (int i = static_cast<int>(parsed_js_[kHostShapes].size()) - 1; i >= 0; i--) {
    for (size_t j = 0; j < parsed_js_[kHostShapes][i].size(); j++) {
      string shape_str = parsed_js_[kHostShapes][i][j];
      if ((!std::all_of(shape_str.begin(), shape_str.end(), [](char c) { return std::isdigit(c); })) &&
          (host_loc_map_.find(shape_str) == host_loc_map_.end())) {
        host_loc_map_[shape_str] = std::make_pair(i, j);
      }
    }
  }
  // Get map <device_loc, host_loc>
  for (auto item : device_loc_map) {
    device_host_shape_loc_[item.first] = GetHostLocationVec(item.second, parsed_js_[kHostShapes].size());
  }
  MS_LOG(INFO) << "Done InitJsonShapeInformation for " << kernel_name_;
}

void AkgKernelImplInfo::InitJsonMappingInformation() {
  // Create map <prime, var> where var is the dynamic tile size and prime is its unique id
  runtime_vars_.clear();
  sorted_runtime_vars_.clear();
  for (size_t i = 0; i < parsed_js_[kRuntimeVars].size(); i++) {
    RuntimeVarPtr v = std::make_shared<RuntimeVar>();
    v->argIndex = parsed_js_[kRuntimeVars][i][v->ArgIndexKey()];
    v->expr = parsed_js_[kRuntimeVars][i][v->ExprKey()];
    v->mark = RuntimeVar::mark_table_[parsed_js_[kRuntimeVars][i].value(v->MarkKey(), "unknown")];
    v->mapDim = parsed_js_[kRuntimeVars][i][v->MapDimKey()];
    v->mapping = parsed_js_[kRuntimeVars][i][v->MappingKey()];
    v->prime = parsed_js_[kRuntimeVars][i][v->PrimeKey()];
    runtime_vars_[v->prime] = v;
    sorted_runtime_vars_.emplace_back(std::make_pair(v->prime, v));
  }

  // Sort the map according to arg index of var
  std::sort(sorted_runtime_vars_.begin(), sorted_runtime_vars_.end(), RuntimeVarCompare());

  // Init mapping info with the staic mapping size and dynamic mapping size will be calculate during resize
  init_mapping_info_ = MappingInfo();
  init_mapping_info_.solve_order_id = {AKG_KERNEL_MOD_TX_IDX, AKG_KERNEL_MOD_TY_IDX, AKG_KERNEL_MOD_TZ_IDX,
                                       AKG_KERNEL_MOD_BX_IDX, AKG_KERNEL_MOD_BY_IDX, AKG_KERNEL_MOD_BZ_IDX};

  // Initialize mapping info as a vector. Only store dividend number for unknown mapping args
  // Record map <unknown map ard id, host_loc>
  init_mapping_.clear();
  map_arg_list_ = {kBlockIdxX, kBlockIdxY, kBlockIdxZ, kThreadIdxX, kThreadIdxY, kThreadIdxZ};
  for (size_t i = 0; i < map_arg_list_.size(); i++) {
    auto map_arg = map_arg_list_[i];
    if (parsed_js_[map_arg].is_number()) {
      uint32_t map_size = parsed_js_[map_arg];
      (void)init_mapping_.emplace_back(map_size);
      auto it = runtime_vars_.find(map_size);
      if (it == runtime_vars_.end()) {
        // update static mapping
        init_mapping_info_.UpdateCurrMapSize(i, map_size);
      } else {
        it->second->curr_map_id = i;
      }
    } else if (CheckJsonValueFormat(map_arg)) {
      string divisor_symbol = parsed_js_[map_arg][0];
      auto dividend = parsed_js_[map_arg][1];
      (void)init_mapping_.emplace_back(static_cast<uint32_t>(dividend));
      unknown_map_loc_[i] = GetHostLocationVec(divisor_symbol, parsed_js_[kHostShapes].size());
      unknown_map_symbol_[i] = divisor_symbol;
      product_var_[dividend] = divisor_symbol;
    } else {
      MS_EXCEPTION(RuntimeError) << "Mapping info format error.";
      return;
    }
  }

  if (runtime_vars_.empty()) {
    return;
  }
  for (auto it = parsed_js_.begin(); it != parsed_js_.end(); ++it) {
    std::string key = it.key();
    if (key.find(kLocalPrefix) != std::string::npos && CheckJsonValueFormat(key)) {
      string divisor_symbol = parsed_js_[key][0];
      auto prime = static_cast<int>(parsed_js_[key][1]);
      local_upper_bound_[prime] = host_loc_map_[divisor_symbol];
      local_upper_bound_symbol_[prime] = divisor_symbol;
      product_var_[prime] = divisor_symbol;
    }
  }

  // update relationship: product = prime0 * prime1
  for (const auto &kv : runtime_vars_) {
    int prime0 = kv.first;
    if (prime0 <= 1) continue;
    if (runtime_vars_.find(prime0) != runtime_vars_.end()) {
      for (const auto &kv2 : runtime_vars_) {
        int product = kv2.first;
        if (product > 0 && product != prime0 && product % prime0 == 0) {
          product_var_[prime0] = product_var_[product];
          product_var_[product / prime0] = product_var_[product];
          std::vector<int> vars({prime0, product / prime0});
          related_values_[product] = vars;
        }
      }
    }
  }
  MS_LOG(INFO) << "Done InitJsonMappingInformation for " << kernel_name_;
}

void AkgKernelImplInfo::preprocessDynamicReduceTiling() {
  // keep static reduce length
  static_reduce_length_ = parsed_js_[kSupportInfo][kReduceSizeStatic];

  // collect thread-level idx
  runtime_threads_order_.clear();
  for (const auto &kv : runtime_vars_) {
    auto var = kv.second;
    // whether the runtime var has "thread" mark
    if (var->prime <= 0 ||
        (var->mark < AKG_DYN_MARK_THREAD_LOWER_BOUND || var->mark >= AKG_DYN_MARK_THREAD_UPPER_BOUND))
      continue;
    runtime_threads_order_.push_back(kv.first);
  }

  // sort tiling orders
  const std::vector<int> orders = {10, 11, 12, 13, 20, 21, 22, 30, 40};  // mark order
  template_tiling_order_.clear();
  for (auto order : orders) {
    for (auto kv : runtime_vars_) {
      auto var = kv.second;
      if (var->prime <= 0 || var->mark != order) continue;
      template_tiling_order_.push_back(std::make_pair(kv.first, var->mark));
    }
  }

  // build a map from prime number to mapping idx
  prime_to_mapping_idx_.clear();
  prime_to_mapping_dividend_.clear();
  for (const auto &p : template_tiling_order_) {
    int prime = p.first;
    prime_to_mapping_idx_[prime] = -1;
    prime_to_mapping_dividend_[prime] = -1;
    for (size_t i = 0; i < map_arg_list_.size(); i++) {
      auto map_arg = map_arg_list_[i];
      if (parsed_js_[map_arg].is_number() && parsed_js_[map_arg] == prime) {
        prime_to_mapping_idx_[prime] = i;
        break;
      } else if (CheckJsonValueFormat(map_arg) && parsed_js_[map_arg][1] == prime) {
        prime_to_mapping_dividend_[prime] = i;
        break;
      }
    }
  }

  enable_atomic_ = parsed_js_[kSupportInfo][kEnableAtomic];
  dyn_algorithm_ = AkgKernelImplInfo::algo_to_int_[parsed_js_[kSupportInfo][kDynAlgorithm]];
}

void AkgKernelImplInfo::InitBeforeMapping() {
  thread_info_ = init_mapping_;
  if (!runtime_vars_.empty()) {
    solved_map_loc_.clear();
    curr_mapping_info_ = init_mapping_info_;
    auto [g, b] = GetProposalParallelSize(problem_size_, device_target_);
    curr_mapping_info_.proposal_grid = g;
    curr_mapping_info_.proposal_block = b;
  }
}

void DynamicTileImpl::UpdateDynamicShapeTilingInfo() {
  if (runtime_vars_.empty()) {
    MS_LOG(DEBUG) << "Static Tile " << kernel_name_;
    return;
  }
  UpdateRuntimeVarUpperBound();
  if (parsed_js_[kSupportInfo]["OperatorType"] == "Reduce") {
    SolveDynamicReduction();
  } else {
    for (auto curr_id : curr_mapping_info_.solve_order_id) {
      SolveDynamicTiling(curr_id);
    }
  }

  for (auto it : sorted_runtime_vars_) {
    arg_size_vec_.push_back(it.second->runtime_size);
  }
  MS_LOG(INFO) << "Done UpdateDynamicShapTilingInfo for " << kernel_name_;
}

int64_t AkgKernelImplInfo::GetFoldedShape(const LocVector &host_loc_vec) {
  auto folded_shape = 1;
  for (auto host_loc : host_loc_vec) {
    auto curr_shape = 1;
    if (host_loc.first == shape_list_.size()) {
      // pure number pair <M, number>
      curr_shape = SizeToInt(host_loc.second);
    } else if (shape_list_[host_loc.first].size() != 0) {
      curr_shape = shape_list_[host_loc.first][host_loc.second];
    }
    if (folded_shape > INT64_MAX / curr_shape) {
      MS_EXCEPTION(RuntimeError) << "For " << kernel_name_ << ", the product of shapes, " << folded_shape << " and "
                                 << curr_shape << ", exceeds INT64_MAX.";
    }
    folded_shape *= curr_shape;
  }
  return folded_shape;
}

void AkgKernelImplInfo::UpdateDynamicShapeMappingInfo() {
  for (auto item : unknown_map_loc_) {
    auto thread_info_id = item.first;
    if (solved_map_loc_.count(thread_info_id)) {
      continue;
    }
    if (thread_info_id >= thread_info_.size()) {
      MS_EXCEPTION(RuntimeError) << "Unknown thread arg index should not exceed thread_info length, "
                                 << "which is 6 (Grid.X/Y/Z, Block.X/Y/Z).";
    }
    auto host_loc_vec = item.second;
    auto dim_size = GetFoldedShape(host_loc_vec);
    auto tile_size = thread_info_[thread_info_id];
    auto dim_size_float = static_cast<float>(dim_size);
    auto tile_size_float = static_cast<float>(tile_size);
    thread_info_[thread_info_id] = static_cast<uint32_t>(std::ceil(dim_size_float / tile_size_float));
  }

  MS_LOG(DEBUG) << "For " << kernel_name_ << ",  thread_info = " << thread_info_;
  MS_LOG(INFO) << "Done UpdateDynamicShapeMappingInfo for " << kernel_name_;
}

void AkgKernelImplInfo::GetDeviceArgSizeVec() {
  arg_size_vec_.clear();
  std::vector<std::vector<int64_t>> device_shape(device_shape_list_);
  // Update each unknown value in device_shape_list to get real device shape
  for (auto item : device_host_shape_loc_) {
    auto device_loc = item.first;
    auto host_loc_vec = item.second;
    device_shape[device_loc.first][device_loc.second] = GetFoldedShape(host_loc_vec);
  }
  problem_size_ = 1;
  for (size_t i = 0; i < device_shape.size(); i++) {
    MS_LOG(DEBUG) << "For " << kernel_name_ << ", input[" << i << "]: host_shape = " << shape_list_[i]
                  << ", device_shape = " << device_shape[i];
    arg_size_vec_.push_back(kRemove);  // useless in memref
    arg_size_vec_.push_back(kKeep);    // data ptr
    arg_size_vec_.push_back(0);        // offset
    auto device_tensor_shape = device_shape[i];
    int64_t tensor_size = 1;
    for (auto item : device_tensor_shape) {
      if (item <= 0) {
        MS_EXCEPTION(RuntimeError) << "Shape still have negative value for kernel: " << kernel_name_
                                   << "with host shape[" << i << "] = " << shape_list_[i] << ", device_tensor_shape["
                                   << i << "] = " << device_tensor_shape;
      }
      arg_size_vec_.push_back(item);
      tensor_size *= item;
    }
    problem_size_ = std::max<int64_t>(problem_size_, tensor_size);
    vector<int64_t> strides(device_tensor_shape.size(), 1);
    for (int j = SizeToInt(device_tensor_shape.size()) - 2; j >= 0; j--) {
      strides[j] = strides[j + 1] * device_tensor_shape[j + 1];
    }
    for (auto item : strides) {
      arg_size_vec_.push_back(item);
    }
  }
  MS_LOG(DEBUG) << "For " << kernel_name_ << ", arg_size_vec = " << arg_size_vec_;
  MS_LOG(INFO) << "Done GetDeviceArgSizeVec for " << kernel_name_;
}

void DynamicTileImpl::UpdateRuntimeVarUpperBound() {
  // Comes from `Block/Thread: [upper_bound, prime]`
  for (const auto &item : unknown_map_loc_) {
    auto thread_info_id = item.first;
    auto host_loc_vec = item.second;
    auto tile_size = thread_info_[thread_info_id];
    auto it = runtime_vars_.find(tile_size);
    if (it != runtime_vars_.end()) {
      auto dim_size = GetFoldedShape(host_loc_vec);
      it->second->upper_bound = dim_size;
      it->second->outer_map_id = thread_info_id;
      auto symbol = unknown_map_symbol_[thread_info_id];
      axis_length_left_[symbol] = dim_size;
    }
  }

  // Comes from `Seq: [upper_bound, prime]`
  for (const auto &it : local_upper_bound_) {
    auto prime = it.first;
    if (runtime_vars_.find(prime) != runtime_vars_.end()) {
      auto host_loc = it.second;
      auto dim_size = shape_list_[host_loc.first][host_loc.second];
      runtime_vars_[prime]->upper_bound = dim_size;
      auto symbol = local_upper_bound_symbol_[prime];
      axis_length_left_[symbol] = dim_size;
    }
  }
}

void DynamicTileImpl::UpdateMapping(int curr_id, int64_t map_size, int64_t prime) {
  // skip when mark is seq.x
  if (curr_id != -1) {
    thread_info_[curr_id] = map_size;
    solved_map_loc_.insert(curr_id);
    curr_mapping_info_.UpdateCurrMapSize(curr_id, map_size);
  }
  if (runtime_vars_.find(prime) == runtime_vars_.end()) {
    return;
  }
  runtime_vars_[prime]->runtime_size = map_size;
  int64_t neg_prime = -prime;
  if (runtime_vars_.find(neg_prime) != runtime_vars_.end()) {
    runtime_vars_[neg_prime]->runtime_size = -map_size;
  }
}

int64_t DynamicTileImpl::TileSizeOpt(const RuntimeVarPtr &var, int64_t dyn_tile_size) {
  // Currently we only optimize tile size for elementwise ops based on problem size.
  bool map_outer_grid = var->curr_map_id >= AKG_KERNEL_MOD_BX_IDX && var->curr_map_id <= AKG_KERNEL_MOD_BZ_IDX;
  if (map_outer_grid) {
    auto rest_grid = std::max<int64_t>(1, curr_mapping_info_.proposal_grid / curr_mapping_info_.total_alloc_grid);
    dyn_tile_size = std::min<int64_t>(dyn_tile_size, rest_grid);
  } else {
    auto rest_block = std::max<int64_t>(1, curr_mapping_info_.proposal_block / curr_mapping_info_.total_alloc_block);
    if (var->curr_map_id == AKG_KERNEL_MOD_TY_IDX && var->upper_bound % rest_block != 0) {
      rest_block = 1;
    }
    dyn_tile_size = std::min<int64_t>(dyn_tile_size, rest_block);
  }
  return dyn_tile_size;
}

void DynamicTileImpl::SolveDynamicReduction() {
  int proper_block = -1, proper_thread = -1, proper_seq = -1;
  int total_red_size =
    std::accumulate(runtime_threads_order_.begin(), runtime_threads_order_.end(), 1,
                    [&](int total, int prime) { return total * axis_length_left_[product_var_[prime]]; });

  if (dyn_algorithm_ == AKG_DYN_ALGO_REDUCE_X) {
    GetProperReduceXConfig(total_red_size, enable_atomic_, &proper_block, &proper_thread, &proper_seq);
  } else if (dyn_algorithm_ == AKG_DYN_ALGO_REDUCE_Y) {
    GetProperReduceYConfig(total_red_size, enable_atomic_, &proper_block, &proper_seq);
  }

  if (dyn_algorithm_ != AKG_DYN_ALGO_REDUCE_X) {
    proper_thread = kNum32;
  }

  proper_thread = (proper_thread - 1) / curr_mapping_info_.total_alloc_block + 1;  // remove used
  for (const auto &p : template_tiling_order_) {
    int prime = p.first;
    int mark = p.second;
    int upper_bound = -1;
    auto symbol = product_var_[prime];
    auto current_length = axis_length_left_[symbol];
    if (AKG_DYN_MARK_THREAD_LOWER_BOUND <= mark && mark < AKG_DYN_MARK_THREAD_UPPER_BOUND) {
      upper_bound = std::min<int>(current_length, (MAX_THREAD_NUM / curr_mapping_info_.total_alloc_block));
    } else {
      upper_bound = current_length;
    }
    int tile_size = -1;
    if (mark == AKG_DYN_MARK_PRODUCT) {
      tile_size =
        runtime_vars_[related_values_[prime][0]]->runtime_size * runtime_vars_[related_values_[prime][1]]->runtime_size;
    } else {
      GetReduceTileSize(mark, upper_bound, proper_thread, proper_seq, &tile_size);
      if (AKG_DYN_MARK_SEQ_LOWER_BOUND <= mark && mark < AKG_DYN_MARK_SEQ_UPPER_BOUND) {
        proper_seq = proper_seq / tile_size;
      }
    }
    // update mapping
    if (prime_to_mapping_idx_[prime] != -1) {
      // scenario 1: BlockIdx.x = prime
      auto curr_idx = prime_to_mapping_idx_[prime];
      thread_info_[curr_idx] = tile_size;
      solved_map_loc_.insert(curr_idx);
    } else if (prime_to_mapping_dividend_[prime] != -1) {
      // scenario 2: BlockIdx.x = symbol / prime
      // NOTE: since we know thread_info_'s format here, we only use tile_size
      // to represent both divier and dividend. update var name later.
      auto curr_idx = prime_to_mapping_dividend_[prime];
      thread_info_[curr_idx] = tile_size;
    }
    if (runtime_vars_.find(prime) != runtime_vars_.end()) {
      runtime_vars_[prime]->runtime_size = tile_size;
      int64_t neg_prime = -prime;
      if (runtime_vars_.find(neg_prime) != runtime_vars_.end()) {
        runtime_vars_[neg_prime]->runtime_size = -tile_size;
      }
    }
    axis_length_left_[symbol] = (current_length - 1) / tile_size + 1;
  }
}

void DynamicTileImpl::SolveDynamicTiling(size_t curr_id) {
  auto prime = thread_info_[curr_id];
  auto it = runtime_vars_.find(prime);
  if (it == runtime_vars_.end()) {
    return;
  }
  auto var = it->second;
  if (var->curr_map_id != static_cast<int>(curr_id)) {
    if (var->outer_map_id != static_cast<int>(curr_id)) {
      MS_EXCEPTION(RuntimeError) << "Unknown var: " << var->ToString() << "; Cannot map currId: " << curr_id;
    } else {
      // In this branch, dividend is stored in thread_info_ and dividend equals to prime number
      // means that the mapping of `curr_id` equals to the upper bound ceildiv the `runtime_size`.
      auto dividend = static_cast<uint32_t>((var->upper_bound - 1) / var->runtime_size) + 1;
      UpdateMapping(curr_id, dividend, 0);
    }
    return;
  }
  auto map_limit = curr_mapping_info_.GetMapLimit(curr_id, device_target_);
  auto upper_bound = std::min<uint32_t>(map_limit, var->upper_bound);
  if (upper_bound < 1) {
    MS_EXCEPTION(RuntimeError) << " Invalid upper_bound of runtime var : " << var->ToString();
  }

  // Init dynamic tile size to the upper bound
  int64_t dyn_tile_size = upper_bound;

  // Add dynamic tiling strategy here
  auto warp_num = std::max<int64_t>(1, var->upper_bound / WARP_SIZE);
  bool map_outer_block = var->outer_map_id >= AKG_KERNEL_MOD_BX_IDX && var->outer_map_id <= AKG_KERNEL_MOD_BZ_IDX;
  if (var->curr_map_id == AKG_KERNEL_MOD_TX_IDX && var->upper_bound % WARP_SIZE != 0) {
    dyn_tile_size = WARP_SIZE * warp_num;
  } else if (var->curr_map_id == AKG_KERNEL_MOD_TY_IDX && var->upper_bound % WARP_SIZE != 0) {
    if (map_outer_block && curr_mapping_info_.total_alloc_grid * WARP_SIZE < ELEM_BEST_GRID_SIZE) {
      dyn_tile_size = 1;
    } else {
      dyn_tile_size = (WARP_SIZE / WARP_ALLOC_GRAN) * warp_num;
    }
  }
  dyn_tile_size = std::min<int64_t>(dyn_tile_size, upper_bound);
  dyn_tile_size = TileSizeOpt(var, dyn_tile_size);
  UpdateMapping(curr_id, dyn_tile_size, prime);
}
}  // namespace kernel
}  // namespace mindspore
