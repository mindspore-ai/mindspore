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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/index_fill_tensor_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "kernel/ascend/acl_ir/acl_helper.h"

namespace mindspore {
namespace kernel {
namespace index_fill_tensor {
std::vector<int64_t> IndexFillTensorAscend::GetIndexArray(const std::vector<KernelTensor *> &inputs) {
  std::vector<int64_t> convertedIndex = {};
  auto index_shape = inputs[kIndex2]->GetShapeVector();
  if (std::any_of(index_shape.begin(), index_shape.end(), [](const auto &dim) { return dim == 0; })) {
    return convertedIndex;
  }

  auto index_dtype_id = inputs[kIndex2]->dtype_id();
  switch (index_dtype_id) {
    case kNumberTypeInt64: {
      auto index = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    case kNumberTypeInt32: {
      auto index = device::ascend::ConvertKernelTensor<std::vector<int32_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    case kNumberTypeInt16: {
      auto index = device::ascend::ConvertKernelTensor<std::vector<int16_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    case kNumberTypeInt8: {
      auto index = device::ascend::ConvertKernelTensor<std::vector<int8_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    case kNumberTypeUInt8: {
      auto index = device::ascend::ConvertKernelTensor<std::vector<uint8_t>>(inputs[kIndex2]);
      convertedIndex.assign(index.begin(), index.end());
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "IndexFillTensor only support int32, int64, int16, int8, and uint8, but got "
                        << TypeIdToString(index_dtype_id);
  }

  return convertedIndex;
}

void IndexFillTensorAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  dim_ = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  value_ = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex3]);
  auto convertedIndex = GetIndexArray(inputs);
  GetWorkspaceForResize(inputs[kIndex0], dim_, convertedIndex, value_, outputs[kIndex0]);
}

bool IndexFillTensorAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto convertedIndex = GetIndexArray(inputs);
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  if (convertedIndex.empty() &&
      std::any_of(input_shape.begin(), input_shape.end(), [](const auto &dim) { return dim == 0; })) {
    MS_LOG(DEBUG) << "aclnnIndexFillTensor throws 'CopyNpuToNpuOp' error when 'self' and 'index' are empty tensors, "
                  << "because we allocated NPU memory for empty 'self' tensor. So we skip launching in this case.";
    return true;
  }

  RunOp(stream_ptr, workspace, inputs[kIndex0], dim_, convertedIndex, value_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(IndexFillTensor, IndexFillTensorAscend);
}  // namespace index_fill_tensor
}  // namespace kernel
}  // namespace mindspore
