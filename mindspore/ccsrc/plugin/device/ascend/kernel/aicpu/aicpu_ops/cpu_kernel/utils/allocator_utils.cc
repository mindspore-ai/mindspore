/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "cpu_kernel/utils/allocator_utils.h"
#include <unordered_set>
#include <vector>
#include "securec/include/securec.h"

#include "cce/fwk_adpt_struct.h"
#include "inc/kernel_log.h"
#include "context/common/status.h"

namespace {
std::unordered_set<uint64_t> g_allocated_ptr;
}

namespace aicpu {
uint32_t CpuKernelAllocatorUtils::ParamCheck(CpuKernelContext &ctx, const std::vector<int64_t> &dims,
                                             const void *data_ptr, Tensor *&outputResultTensor) {
  if (dims.empty()) {
    CUST_KERNEL_LOG_ERROR(ctx, "UpdateOutputDataTensor dims size == 0.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  CUST_KERNEL_CHECK_NULLPTR(ctx, outputResultTensor, KERNEL_STATUS_PARAM_INVALID, "outputResultTensor nullptr");
  CUST_KERNEL_CHECK_NULLPTR(ctx, data_ptr, KERNEL_STATUS_PARAM_INVALID, "data_ptr nullptr");
  return KERNEL_STATUS_OK;
}

uint32_t CpuKernelAllocatorUtils::UpdateOutputDataTensor(CpuKernelContext &ctx, const std::vector<int64_t> &dims,
                                                         DataType type, const void *data_ptr, int64_t input_data_size,
                                                         Tensor *&outputResultTensor) {
  uint32_t check_ret = ParamCheck(ctx, dims, &data_ptr, outputResultTensor);
  if (check_ret != KERNEL_STATUS_OK) {
    return check_ret;
  }
  CUST_KERNEL_LOG_INFO(ctx, "UpdateOutputDataTensor::START!!");

  int64_t data_size = GetInputDataSize(ctx, dims, type);
  if (data_size < 0) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (data_size > input_data_size) {
    CUST_KERNEL_LOG_ERROR(ctx, "data_size[%ld] mast less than input_data_size[%ld]!", data_size, input_data_size);
    return KERNEL_STATUS_INNER_ERROR;
  }

  int64_t shape_buff_size = 0;
  CUST_KERNEL_CHECK_ASSIGN_64S_MULTI(ctx, int64_t(dims.size()), int64_t(sizeof(int64_t)), shape_buff_size,
                                     KERNEL_STATUS_PARAM_INVALID);

  void *output_shape_ptr = malloc(shape_buff_size);
  CUST_KERNEL_CHECK_NULLPTR(ctx, output_shape_ptr, KERNEL_STATUS_PARAM_INVALID, "malloc error, size[%ld]!",
                            shape_buff_size);

  int32_t ret = memcpy_s(output_shape_ptr, shape_buff_size, dims.data(), shape_buff_size);
  if (ret != EOK) {
    free(output_shape_ptr);
    CUST_KERNEL_LOG_ERROR(ctx, "memcpy error, size[%ld], ret[%d]!", shape_buff_size, ret);
    return KERNEL_STATUS_INNER_ERROR;
  }

  aicpu::FWKAdapter::ResultSummary *result_summary =
    reinterpret_cast<aicpu::FWKAdapter::ResultSummary *>(outputResultTensor->GetData());
  result_summary->raw_data_size = data_size;
  result_summary->shape_data_size = shape_buff_size;

  if (data_size == 0) {
    result_summary->raw_data_ptr = reinterpret_cast<uint64_t>(nullptr);
    result_summary->shape_data_ptr = reinterpret_cast<uint64_t>(output_shape_ptr);
    (void)g_allocated_ptr.insert(result_summary->shape_data_ptr);
    CUST_KERNEL_LOG_INFO(ctx, "UpdateOutputDataTensor:: empty tensor END!!");
    return KERNEL_STATUS_OK;
  }

  void *output_data_ptr = malloc(data_size);
  if (output_data_ptr == nullptr) {
    CUST_KERNEL_LOG_ERROR(ctx, "malloc error, size[%ld]!", data_size);
    free(output_shape_ptr);
    return KERNEL_STATUS_INNER_ERROR;
  }

  ret = memcpy_s(output_data_ptr, data_size, data_ptr, data_size);
  if (ret != EOK) {
    free(output_data_ptr);
    free(output_shape_ptr);
    CUST_KERNEL_LOG_ERROR(ctx, "memcpy_s error, size[%ld], ret[%d]!", data_size, ret);
    return KERNEL_STATUS_INNER_ERROR;
  }

  result_summary->raw_data_ptr = reinterpret_cast<uint64_t>(output_data_ptr);
  result_summary->shape_data_ptr = reinterpret_cast<uint64_t>(output_shape_ptr);
  CUST_KERNEL_LOG_INFO(ctx, "raw_data_ptr [%p]", output_data_ptr);
  CUST_KERNEL_LOG_INFO(ctx, "shape_data_ptr [%p]", output_shape_ptr);

  (void)g_allocated_ptr.insert(result_summary->raw_data_ptr);
  (void)g_allocated_ptr.insert(result_summary->shape_data_ptr);
  CUST_KERNEL_LOG_INFO(ctx, "UpdateOutputDataTensor :: END!!");

  return KERNEL_STATUS_OK;
}

int64_t CpuKernelAllocatorUtils::GetInputDataSize(CpuKernelContext &ctx, const std::vector<int64_t> &dims,
                                                  DataType type) {
  int64_t num_elements = 1;
  for (size_t i = 0; i < dims.size(); i++) {
    CUST_KERNEL_CHECK_ASSIGN_64S_MULTI(ctx, num_elements, dims[i], num_elements, KERNEL_STATUS_PARAM_INVALID);
  }

  int64_t data_size = 0;
  int element_size = GetSizeByDataType(type);
  CUST_KERNEL_CHECK_ASSIGN_64S_MULTI(ctx, num_elements, int64_t(element_size), data_size, KERNEL_STATUS_PARAM_INVALID);

  if (data_size < 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "UpdateOutputDataTensor data_size[%ld].", data_size);
  }

  return data_size;
}

uint32_t CpuKernelAllocatorUtils::CheckOutputDataPtr(CpuKernelContext &ctx, const uint64_t data_ptr) {
  auto find_data_ptr = g_allocated_ptr.find(data_ptr);
  if ((find_data_ptr == g_allocated_ptr.end())) {
    CUST_KERNEL_LOG_ERROR(ctx, "CheckOutputDataPtr invalid [%lu].", data_ptr);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t CpuKernelAllocatorUtils::DeleteOutputDataPtr(CpuKernelContext &ctx, const uint64_t data_ptr) {
  CUST_KERNEL_LOG_INFO(ctx, "DeleteOutputDataPtr [%lu]", data_ptr);
  auto find_data_ptr = g_allocated_ptr.find(data_ptr);
  if (find_data_ptr != g_allocated_ptr.end()) {
    free(reinterpret_cast<void *>(data_ptr));
    g_allocated_ptr.erase(find_data_ptr);
  } else {
    CUST_KERNEL_LOG_WARN(ctx, "DeleteOutputDataPtr invalid [%lu].", data_ptr);
  }

  return KERNEL_STATUS_OK;
}

uint32_t CpuKernelAllocatorUtils::AllocateOutputTensorDataMemory(CpuKernelContext &ctx,
                                                                 const std::vector<uint64_t> &shape, DataType type,
                                                                 Tensor *&outputResultTensor) {
  CUST_KERNEL_CHECK_NULLPTR(ctx, outputResultTensor, KERNEL_STATUS_PARAM_INVALID, "outputResultTensor nullptr");
  CUST_KERNEL_LOG_INFO(ctx, "AllocateOutputTensorDataMemory::START!!");
  if (shape.empty()) {
    CUST_KERNEL_LOG_ERROR(ctx, "AllocateOutputTensorDataMemory shape size == 0.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t num_elements = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    CUST_KERNEL_CHECK_ASSIGN_64S_MULTI(ctx, num_elements, static_cast<int64_t>(shape[i]), num_elements,
                                       KERNEL_STATUS_PARAM_INVALID);
  }

  uint64_t data_size = 0;
  int32_t element_size = GetSizeByDataType(type);
  CUST_KERNEL_CHECK_ASSIGN_64S_MULTI(ctx, num_elements, element_size, data_size, KERNEL_STATUS_PARAM_INVALID);
  uint64_t shape_buffer_size = 0;
  CUST_KERNEL_CHECK_ASSIGN_64S_MULTI(ctx, shape.size(), sizeof(int64_t), shape_buffer_size,
                                     KERNEL_STATUS_PARAM_INVALID);

  void *output_shape_ptr = malloc(shape_buffer_size);
  CUST_KERNEL_CHECK_NULLPTR(ctx, output_shape_ptr, KERNEL_STATUS_PARAM_INVALID, "malloc error, size[%llu]!",
                            shape_buffer_size);
  int32_t ret = memcpy_s(output_shape_ptr, shape_buffer_size, shape.data(), shape_buffer_size);
  if (ret != EOK) {
    free(output_shape_ptr);
    CUST_KERNEL_LOG_ERROR(ctx, "memcpy error, size[%llu], ret[%d]!", shape_buffer_size, ret);
    return KERNEL_STATUS_INNER_ERROR;
  }
  aicpu::FWKAdapter::ResultSummary *result_summary =
    reinterpret_cast<aicpu::FWKAdapter::ResultSummary *>(outputResultTensor->GetData());
  if (data_size == 0) {
    result_summary->raw_data_ptr = reinterpret_cast<uint64_t>(nullptr);
    result_summary->raw_data_size = 0;
    result_summary->shape_data_ptr = reinterpret_cast<uint64_t>(output_shape_ptr);
    result_summary->shape_data_size = shape_buffer_size;
    (void)g_allocated_ptr.insert(result_summary->shape_data_ptr);
    CUST_KERNEL_LOG_INFO(ctx, "AllocateOutputTensorDataMemory:: empty tensor END!!");
    return KERNEL_STATUS_OK;
  }
  void *output_data_ptr = malloc(data_size);
  if (output_data_ptr == nullptr) {
    CUST_KERNEL_LOG_ERROR(ctx, "malloc error, size[%lu]!", data_size);
    free(output_shape_ptr);
    return KERNEL_STATUS_INNER_ERROR;
  }

  result_summary->raw_data_size = data_size;
  result_summary->raw_data_ptr = reinterpret_cast<uint64_t>(output_data_ptr);
  result_summary->shape_data_size = shape_buffer_size;
  result_summary->shape_data_ptr = reinterpret_cast<uint64_t>(output_shape_ptr);

  CUST_KERNEL_LOG_INFO(ctx, "raw_data_ptr [%llu]", output_data_ptr);
  CUST_KERNEL_LOG_INFO(ctx, "shape_data_ptr [%llu]", output_shape_ptr);

  (void)g_allocated_ptr.insert(result_summary->raw_data_ptr);
  (void)g_allocated_ptr.insert(result_summary->shape_data_ptr);
  CUST_KERNEL_LOG_INFO(ctx, "AllocateOutputTensorDataMemory :: END!!");

  return KERNEL_STATUS_OK;
}

}  // namespace aicpu
