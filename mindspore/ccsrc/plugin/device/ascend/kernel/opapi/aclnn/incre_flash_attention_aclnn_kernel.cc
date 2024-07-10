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
#include "plugin/device/ascend/kernel/opapi/aclnn/incre_flash_attention_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "transform/graph_ir/op_adapter_base.h"

namespace mindspore {
namespace kernel {

std::vector<int64_t> ConvertActualSeqLengthsToVector(KernelTensor *const actual_seq_length_ptr) {
  MS_EXCEPTION_IF_NULL(actual_seq_length_ptr);
  std::vector<int64_t> actual_seq_lengths_vector;
  if (actual_seq_length_ptr->type_id() != kMetaTypeNone) {
    TypeId actual_seq_lengths_dtype_id = actual_seq_length_ptr->dtype_id();
    if (actual_seq_lengths_dtype_id == kNumberTypeInt64) {
      actual_seq_lengths_vector = actual_seq_length_ptr->GetValueWithCheck<std::vector<int64_t>>();
    } else if (actual_seq_lengths_dtype_id == kNumberTypeInt32) {
      std::vector<int32_t> actual_seq_lengths_vector_int32 =
        actual_seq_length_ptr->GetValueWithCheck<std::vector<int32_t>>();
      actual_seq_lengths_vector.assign(actual_seq_lengths_vector_int32.begin(), actual_seq_lengths_vector_int32.end());
    } else {
      MS_LOG(EXCEPTION) << "actual_seq_lengths data type must be Int32 or Int64, but got "
                        << TypeIdToString(actual_seq_lengths_dtype_id);
    }
  }
  return actual_seq_lengths_vector;
}

void IncreFlashAttentionAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  std::vector<KernelTensor *> key_vector{inputs[kIndex1]};
  MS_EXCEPTION_IF_NULL(inputs[kIndex2]);
  std::vector<KernelTensor *> value_vector{inputs[kIndex2]};
  MS_EXCEPTION_IF_NULL(inputs[kIndex15]);
  auto num_heads = transform::ConvertKernelTensor<int64_t>(inputs[kIndex15]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex16]);
  auto input_layout = transform::ConvertKernelTensor<int64_t>(inputs[kIndex16]);
  auto input_layout_str = transform::FASInputLayoutMode::ConvertEnumToString(input_layout);
  MS_EXCEPTION_IF_NULL(inputs[kIndex17]);
  auto scale_value = transform::ConvertKernelTensor<float>(inputs[kIndex17]);
  auto scale_value_d = static_cast<double>(scale_value);
  MS_EXCEPTION_IF_NULL(inputs[kIndex18]);
  auto num_key_value_heads = transform::ConvertKernelTensor<int64_t>(inputs[kIndex18]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex19]);
  auto block_size = transform::ConvertKernelTensor<int64_t>(inputs[kIndex19]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex20]);
  auto inner_precise = transform::ConvertKernelTensor<int64_t>(inputs[kIndex20]);

  auto actual_seq_lengths_vector = ConvertActualSeqLengthsToVector(inputs[kIndex4]);
  // For interface aclnnIncreFlashAttentionV4, param inputs[kIndex5] (pse_shift) should follow param value_vector
  GetWorkspaceForResize(inputs[kIndex0], key_vector, value_vector, inputs[kIndex5], inputs[kIndex3],
                        actual_seq_lengths_vector, inputs[kIndex6], inputs[kIndex7], inputs[kIndex8], inputs[kIndex9],
                        inputs[kIndex10], inputs[kIndex11], inputs[kIndex12], inputs[kIndex13], inputs[kIndex14],
                        num_heads, scale_value_d, input_layout_str, num_key_value_heads, block_size, inner_precise,
                        outputs[kIndex0]);
}

bool IncreFlashAttentionAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  std::vector<KernelTensor *> key_vector{inputs[kIndex1]};
  MS_EXCEPTION_IF_NULL(inputs[kIndex2]);
  std::vector<KernelTensor *> value_vector{inputs[kIndex2]};
  MS_EXCEPTION_IF_NULL(inputs[kIndex15]);
  auto num_heads = transform::ConvertKernelTensor<int64_t>(inputs[kIndex15]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex16]);
  auto input_layout = transform::ConvertKernelTensor<int64_t>(inputs[kIndex16]);
  auto input_layout_str = transform::FASInputLayoutMode::ConvertEnumToString(input_layout);
  MS_EXCEPTION_IF_NULL(inputs[kIndex17]);
  auto scale_value = transform::ConvertKernelTensor<float>(inputs[kIndex17]);
  auto scale_value_d = static_cast<double>(scale_value);
  MS_EXCEPTION_IF_NULL(inputs[kIndex18]);
  auto num_key_value_heads = transform::ConvertKernelTensor<int64_t>(inputs[kIndex18]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex19]);
  auto block_size = transform::ConvertKernelTensor<int64_t>(inputs[kIndex19]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex20]);
  auto inner_precise = transform::ConvertKernelTensor<int64_t>(inputs[kIndex20]);

  auto actual_seq_lengths_vector = ConvertActualSeqLengthsToVector(inputs[kIndex4]);
  // For interface aclnnIncreFlashAttentionV4, param inputs[kIndex5] (pse_shift) should follow param value_vector
  ParseGenExecutor(GEN_EXECUTOR_BOOST(
    op_type_, hash_id_, inputs[kIndex0], key_vector, value_vector, inputs[kIndex5], inputs[kIndex3],
    actual_seq_lengths_vector, inputs[kIndex6], inputs[kIndex7], inputs[kIndex8], inputs[kIndex9], inputs[kIndex10],
    inputs[kIndex11], inputs[kIndex12], inputs[kIndex13], inputs[kIndex14], num_heads, scale_value_d, input_layout_str,
    num_key_value_heads, block_size, inner_precise, outputs[kIndex0]));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(IncreFlashAttention, IncreFlashAttentionAscend);
}  // namespace kernel
}  // namespace mindspore
