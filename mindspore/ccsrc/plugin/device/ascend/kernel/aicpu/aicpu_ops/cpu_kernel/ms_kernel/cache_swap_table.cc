/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "cache_swap_table.h"
#include <securec.h>
#include <map>
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "utils/sparse_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *const kCacheSwapTable = "CacheSwapTable";
}

namespace aicpu {
template <typename T>
uint32_t CacheSwapTableTask(std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs, int64_t batch_size,
                            int64_t output_size, int64_t one_line_col, int type_size) {
  if (inputs.size() == 0 || outputs.size() == 0) {
    KERNEL_LOG_ERROR("CacheSwapTable input or output is empty.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  char *cache_table = reinterpret_cast<char *>(inputs[0]->GetData());
  T *swap_cache_idx = reinterpret_cast<T *>(inputs[1]->GetData());
  uint64_t swap_cache_idx_size = inputs[1]->GetDataSize();
  char *miss_value = reinterpret_cast<char *>(inputs[2]->GetData());

  char *old_value = reinterpret_cast<char *>(outputs[0]->GetData());

  errno_t ret = memset_s(old_value, static_cast<size_t>(output_size * type_size), 0x00,
                         static_cast<size_t>(output_size * type_size));
  if (ret != EOK) {
    KERNEL_LOG_ERROR("Memset failed, result[%d]", ret);
    return KERNEL_STATUS_INNER_ERROR;
  }

  uint64_t single_copy_size = static_cast<uint64_t>(type_size * one_line_col);

  if (swap_cache_idx_size < static_cast<uint64_t>(batch_size)) {
    KERNEL_LOG_ERROR(
      "The value of swap_cache_idx_size:[%llu] must be less than "
      "batch_size:[%lld]",
      swap_cache_idx_size, batch_size);
    return KERNEL_STATUS_INNER_ERROR;
  }

  uint64_t old_value_size = outputs[0]->GetDataSize();
  uint64_t cache_table_size = inputs[0]->GetDataSize();
  for (int64_t i = 0; i < batch_size; ++i) {
    if (swap_cache_idx[i] < 0) {
      continue;
    }
    ret = memcpy_s(old_value + i * single_copy_size, old_value_size, cache_table + swap_cache_idx[i] * single_copy_size,
                   single_copy_size);
    old_value_size -= single_copy_size;
    if (ret != EOK) {
      KERNEL_LOG_ERROR("CacheSwapTable memcpy failed, result [%d].", ret);
      return KERNEL_STATUS_INNER_ERROR;
    }
    ret = memcpy_s(cache_table + swap_cache_idx[i] * single_copy_size, cache_table_size,
                   miss_value + i * single_copy_size, single_copy_size);
    cache_table_size -= single_copy_size;
    if (ret != EOK) {
      KERNEL_LOG_ERROR("CacheSwapTable memcpy failed, result [%d].", ret);
      return KERNEL_STATUS_INNER_ERROR;
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t CacheSwapTableMsCpuKernel::DoCompute() {
  std::map<int, std::function<uint32_t(std::vector<Tensor *> &, std::vector<Tensor *> &, int64_t &, int64_t &,
                                       int64_t &, int &)>>
    calls;
  calls[DT_INT32] = CacheSwapTableTask<int32_t>;
  calls[DT_INT64] = CacheSwapTableTask<int64_t>;

  if (calls.find(indices_type_) == calls.end()) {
    KERNEL_LOG_ERROR(
      "CacheSwapTableMsCpuKernel op doesn't support indices tensor types: "
      "[%s]",
      DTypeStr(indices_type_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  int type_size = GetSizeByDataType(param_type_);
  return calls[indices_type_](inputs_, outputs_, batch_size_, output_size_, one_line_col_, type_size);
}

uint32_t CacheSwapTableMsCpuKernel::GetInputAndCheck(const CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("GetInputAndCheck start!");
  // get input Tensors
  const uint32_t kNumInput = 3;
  for (uint32_t i = 0; i < kNumInput; ++i) {
    Tensor *tensor = ctx.Input(i);
    KERNEL_CHECK_NULLPTR(tensor, KERNEL_STATUS_PARAM_INVALID, "Get input tensor[%d] failed", i)
    inputs_.push_back(tensor);
  }
  // get output Tensors
  const uint32_t kNumOutput = 1;
  for (uint32_t i = 0; i < kNumOutput; ++i) {
    Tensor *tensor = ctx.Output(i);
    KERNEL_CHECK_NULLPTR(tensor, KERNEL_STATUS_PARAM_INVALID, "Get output tensor[%d] failed", i)
    outputs_.push_back(tensor);
  }
  // get param type
  param_type_ = static_cast<DataType>(inputs_[0]->GetDataType());
  indices_type_ = static_cast<DataType>(inputs_[1]->GetDataType());
  KERNEL_LOG_INFO("GetInputAndCheck success!");

  std::shared_ptr<TensorShape> cache_table_shape = ctx.Input(0)->GetTensorShape();
  std::shared_ptr<TensorShape> indices_shape = ctx.Input(1)->GetTensorShape();

  for (int32_t i = 1; i < cache_table_shape->GetDims(); ++i) {
    KERNEL_CHECK_ASSIGN_64S_MULTI(one_line_col_, cache_table_shape->GetDimSize(i), one_line_col_,
                                  KERNEL_STATUS_PARAM_INVALID);
  }
  for (int32_t i = 0; i < indices_shape->GetDims(); ++i) {
    KERNEL_CHECK_ASSIGN_64S_MULTI(batch_size_, indices_shape->GetDimSize(i), batch_size_, KERNEL_STATUS_PARAM_INVALID);
  }
  output_size_ = batch_size_ * one_line_col_;
  return KERNEL_STATUS_OK;
}

uint32_t CacheSwapTableMsCpuKernel::Compute(const CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  res = DoCompute();
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Compute failed");
    return res;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kCacheSwapTable, CacheSwapTableMsCpuKernel);
}  // namespace aicpu
