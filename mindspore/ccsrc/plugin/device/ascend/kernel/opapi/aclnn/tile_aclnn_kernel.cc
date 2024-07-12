/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/opapi/aclnn/tile_aclnn_kernel.h"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
void TileAscend::GetAdaptedMultiples(KernelTensor *x_tensor, KernelTensor *multiples_tensor) {
  auto x_shape = x_tensor->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(x_shape))) {
    MS_LOG(EXCEPTION) << "For 'Tile', the tensor's shape should not be dynamic rank in launch stage!";
  }
  auto x_dim = LongToSize(x_shape.size());
  auto multiples_vector = multiples_tensor->GetValueWithCheck<std::vector<int64_t>>();
  // Expand dims with 1 in head when its length is less than x rank.
  if (x_dim > multiples_vector.size()) {
    multiples_vector.reserve(x_dim);
    auto expand_len = x_dim - multiples_vector.size();
    (void)multiples_vector.insert(multiples_vector.begin(), expand_len, 1);
  }

  multiples_ = std::move(multiples_vector);
}

void TileAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  GetAdaptedMultiples(inputs[kIndex0], inputs[kIndex1]);
  GetWorkspaceForResize(inputs[kIndex0], multiples_, outputs[kIndex0]);
}

bool TileAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], multiples_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Tile, TileAscend);
}  // namespace kernel
}  // namespace mindspore
