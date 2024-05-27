/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_INTERNAL_KERNEL_IN_OUT_MAP_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_INTERNAL_KERNEL_IN_OUT_MAP_H_

#include <set>
#include <map>
#include <vector>
#include <unordered_map>
#include <string>
#include "mindapi/base/type_id.h"

namespace mindspore {
namespace kernel {
class InternalKernelModInOutMap {
 public:
  InternalKernelModInOutMap() = default;
  ~InternalKernelModInOutMap() = default;

  static InternalKernelModInOutMap *GetInstance();
  void SetKernelMap(const std::string op_name, int map_dtype, std::vector<int> map);
  void SetMutableList(const std::string op_name, int map_dtype);
  std::vector<int> GetKernelInMap(std::string op_name, bool *is_mutable);
  std::vector<int> GetKernelOutMap(std::string op_name, bool *is_mutable);
  std::vector<int64_t> MapInternelInputDtypes(std::string op_name, const std::vector<TypeId> &ms_dtypes);
  std::vector<int64_t> MapInternelOutputDtypes(std::string op_name, const std::vector<TypeId> &ms_dtypes);

 private:
  std::map<std::string, std::vector<int>> input_idx_;  /* ms idx */
  std::map<std::string, std::vector<int>> output_idx_; /* ms idx */
  std::vector<std::string> mutable_input_list_;
  std::vector<std::string> mutable_output_list_;
};

class InternalKernelModInOutRegistrar {
 public:
  InternalKernelModInOutRegistrar(const std::string op_name, const int map_type, int total_count, ...);
  ~InternalKernelModInOutRegistrar() = default;
};
#define INTERNEL_KERNEL_IN_OUT_MUTABLE_LENGTH 999
#define REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(op, map_cnt, ...) \
  static InternalKernelModInOutRegistrar g_internal_map_in_##op##map_cnt(#op, 0, map_cnt, ##__VA_ARGS__);
#define REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(op, map_cnt, ...) \
  static InternalKernelModInOutRegistrar g_internal_map_out_##op##map_cnt(#op, 1, map_cnt, ##__VA_ARGS__);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_INTERNAL_KERNEL_IN_OUT_MAP_H_
