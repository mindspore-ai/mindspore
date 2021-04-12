/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PS_PS_CACHE_ASCEND_ASCEND_PS_CACHE_H_
#define MINDSPORE_CCSRC_PS_PS_CACHE_ASCEND_ASCEND_PS_CACHE_H_

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "ps/ps_cache/ps_cache_basic.h"
#include "backend/kernel_compiler/aicpu/aicpu_kernel_mod.h"
#include "ir/dtype.h"

namespace mindspore {
namespace ps {
namespace ascend {
struct KernelNodeInfo {
  KernelNodeInfo(const std::string &op_name, std::vector<std::vector<size_t>> input_data_shape,
                 std::vector<TypeId> input_data_type, std::vector<std::vector<size_t>> output_data_shape,
                 std::vector<TypeId> output_data_type)
      : op_name_(op_name) {
    input_data_shape_.swap(input_data_shape);
    input_data_type_.swap(input_data_type);
    output_data_shape_.swap(output_data_shape);
    output_data_type_.swap(output_data_type);
  }
  std::string op_name_;
  std::vector<std::vector<size_t>> input_data_shape_;
  std::vector<TypeId> input_data_type_;
  std::vector<std::vector<size_t>> output_data_shape_;
  std::vector<TypeId> output_data_type_;
};

class AscendPsCache : public PsCacheBasic {
 public:
  AscendPsCache() = default;
  ~AscendPsCache() override = default;
  bool InitDevice(uint32_t device_id, const void *context) override;
  void *MallocMemory(size_t size) override;
  bool MallocConstantMemory(size_t cache_vocab_size) override;
  bool RecordEvent() override;
  bool SynchronizeEvent() override;
  bool SynchronizeStream() override;
  bool CopyHostMemToDevice(void *dst, const void *src, size_t size) override;
  bool CopyDeviceMemToHost(void *dst, const void *src, size_t size) override;
  bool HashSwapOut(void *hash_table_addr, void *swap_out_value_addr, void *swap_out_index_addr, size_t cache_vocab_size,
                   size_t embedding_size, size_t swap_out_size) override;
  bool HashSwapIn(void *hash_table_addr, void *swap_in_value_addr, void *swap_in_index_addr, size_t cache_vocab_size,
                  size_t embedding_size, size_t swap_in_size) override;

 private:
  int *offset_addr_{nullptr};
  int *cache_vocab_size_addr_{nullptr};
  std::unique_ptr<rtEvent_t> event_;
};
}  // namespace ascend
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_PS_CACHE_ASCEND_ASCEND_PS_CACHE_H_
