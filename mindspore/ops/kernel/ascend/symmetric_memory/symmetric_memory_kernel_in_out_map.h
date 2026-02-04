/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_SYMMETRIC_MEMORY_KERNEL_IN_OUT_MAP_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_SYMMETRIC_MEMORY_KERNEL_IN_OUT_MAP_H_

#include <set>
#include <map>
#include <vector>
#include <unordered_map>
#include <string>
#include "mindapi/base/type_id.h"
#include "include/symmetric_memory.h"

namespace mindspore {
namespace kernel {
#define INPUT_NUM_1 1
#define INPUT_NUM_2 2
#define INPUT_NUM_3 3
#define INPUT_NUM_4 4
#define INPUT_NUM_5 5
#define INPUT_NUM_6 6
#define INPUT_NUM_7 7
#define INPUT_NUM_8 8
#define INPUT_NUM_9 9
#define INPUT_NUM_10 10
#define INPUT_NUM_11 11
#define INPUT_NUM_26 26
#define OUTPUT_NUM_1 1
#define OUTPUT_NUM_2 2
#define OUTPUT_NUM_3 3
#define OUTPUT_NUM_4 4
#define OUTPUT_NUM_5 5
#define OUTPUT_NUM_6 6
#define INDEX_0 0
#define INDEX_1 1
#define INDEX_2 2
#define INDEX_3 3
#define INDEX_4 4
#define INDEX_5 5
#define INDEX_6 6
#define INDEX_7 7
#define INDEX_8 8
#define INDEX_9 9
#define INDEX_10 10
#define INDEX_11 11
#define INDEX_12 12
#define INDEX_13 13
#define INDEX_14 14
#define INDEX_15 15
#define INDEX_16 16
#define INDEX_17 17
#define INDEX_18 18
#define INDEX_19 19
#define INDEX_20 20
#define INDEX_21 21
#define INDEX_22 22
#define INDEX_23 23
#define INDEX_24 24
#define INDEX_25 25
enum SymmetricMemoryKernelMapDtype : int {
  SYMMETRIC_MEMORY_KERNEL_MAP_INPUT = 0,
  SYMMETRIC_MEMORY_KERNEL_MAP_OUTPUT = 1
};
class SymmetricMemoryKernelModInOutMap {
 public:
  SymmetricMemoryKernelModInOutMap() = default;
  ~SymmetricMemoryKernelModInOutMap() = default;

  static SymmetricMemoryKernelModInOutMap *GetInstance();
  void AppendKernelMap(const std::string &op_name, SymmetricMemoryKernelMapDtype map_dtype, std::vector<int> map);
  void AppendMutableList(const std::string &op_name, SymmetricMemoryKernelMapDtype map_dtype);
  std::vector<int> GetKernelInMap(const std::string &op_name, bool *is_mutable);
  std::vector<int> GetKernelOutMap(const std::string &op_name, bool *is_mutable);

  std::vector<symmetricmemory::DataType> MapSymmetricMemoryInputDtypes(const std::string &op_name,
                                                                       const std::vector<TypeId> &ms_dtypes);
  std::vector<symmetricmemory::DataType> MapSymmetricMemoryOutputDtypes(const std::string &op_name,
                                                                        const std::vector<TypeId> &ms_dtypes);

 private:
  std::map<std::string, std::vector<int>> input_idx_;  /* ms idx */
  std::map<std::string, std::vector<int>> output_idx_; /* ms idx */
  std::set<std::string> mutable_input_list_;
  std::set<std::string> mutable_output_list_;
};

class SymmetricMemoryKernelModInOutRegistrar {
 public:
  SymmetricMemoryKernelModInOutRegistrar(const std::string op_name, const int map_type, int total_count, ...);
  ~SymmetricMemoryKernelModInOutRegistrar() = default;
};
#define SYMMETRIC_MEMORY_KERNEL_IN_OUT_MUTABLE_LENGTH 999
#define REG_MS_TO_SYMMETRIC_MEMORY_IN_TENSOR_IDX_MAP(op, map_cnt, ...) \
  static SymmetricMemoryKernelModInOutRegistrar g_symmetric_memory_map_in_##op##map_cnt(#op, 0, map_cnt, ##__VA_ARGS__);
#define REG_MS_TO_SYMMETRIC_MEMORY_OUT_TENSOR_IDX_MAP(op, map_cnt, ...)                                   \
  static SymmetricMemoryKernelModInOutRegistrar g_symmetric_memory_map_out_##op##map_cnt(#op, 1, map_cnt, \
                                                                                         ##__VA_ARGS__);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_SYMMETRIC_MEMORY_KERNEL_IN_OUT_MAP_H_
