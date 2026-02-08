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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/quant_batch_matmul_aclnn_kernel.h"
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "include/backend/common/ms_device_shape_transfer.h"

namespace mindspore {
namespace kernel {
namespace quant_batch_matmul {
void QuantMatmulV4Ascend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  transpose_x1_ = inputs[kIndex6]->GetValueWithCheck<bool>();
  transpose_x2_ = inputs[kIndex7]->GetValueWithCheck<bool>();
  const auto w_tensor = std::make_shared<KernelTensor>(*inputs[kIndex1]);
  auto format = w_tensor->format();
  if (format == FRACTAL_NZ) {
    if (w_tensor->tensor_storage_info() != nullptr) {
      MS_LOG(EXCEPTION) << "For QuantMatmulV4Ascend NZ is not support when storage_info is not nullptr";
    }

    auto nd_shape = w_tensor->GetShapeVector();
    auto nz_shape =
      trans::DeviceShapeTransfer().GetDeviceShapeByFormat(nd_shape, w_tensor->GetStringFormat(), w_tensor->dtype_id());

    auto strides = nd_shape;
    if (!strides.empty()) {
      strides.erase(strides.begin());
    }
    strides.push_back(1);
    for (int i = static_cast<int>(strides.size()) - 2; i >= 0; i--) {
      strides[i] = strides[i] * strides[i + 1];
    }
    auto storage_info = std::make_shared<TensorStorageInfo>(nd_shape, strides, nz_shape, strides, true);
    w_tensor->set_tensor_storage_info(storage_info);
  }
  GetWorkspaceForResize(inputs[kIndex0], w_tensor.get(), inputs[kIndex2], inputs[kIndex3], inputs[kIndex5],
                        inputs[kIndex4], transpose_x1_, transpose_x2_, outputs[kIndex0]);
}

bool QuantMatmulV4Ascend::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex5],
        inputs[kIndex4], transpose_x1_, transpose_x2_, outputs[kIndex0]);
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(QuantBatchMatmul, QuantMatmulV4Ascend);
}  // namespace quant_batch_matmul
}  // namespace kernel
}  // namespace mindspore
