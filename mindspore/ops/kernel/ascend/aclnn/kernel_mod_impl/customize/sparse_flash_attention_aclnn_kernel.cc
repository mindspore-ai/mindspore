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

#include "kernel/ascend/aclnn/kernel_mod_impl/customize/sparse_flash_attention_aclnn_kernel.h"

#include <algorithm>
#include <vector>
#include <string>
#include <optional>

#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"

namespace mindspore {
namespace kernel {
namespace sparse_flash_attention {

namespace {
// Input indices
constexpr size_t kInputQueryIndex = 0;
constexpr size_t kInputKeyIndex = 1;
constexpr size_t kInputValueIndex = 2;
constexpr size_t kInputSparseIndicesIndex = 3;
constexpr size_t kInputScaleValueIndex = 4;
constexpr size_t kInputBlockTableIndex = 5;
constexpr size_t kInputActualSeqLengthsQueryIndex = 6;
constexpr size_t kInputActualSeqLengthsKvIndex = 7;
constexpr size_t kInputQueryRopeIndex = 8;
constexpr size_t kInputKeyRopeIndex = 9;
constexpr size_t kInputSparseBlockSizeIndex = 10;
constexpr size_t kInputLayoutQueryIndex = 11;
constexpr size_t kInputLayoutKvIndex = 12;
constexpr size_t kInputSparseModeIndex = 13;
constexpr size_t kInputPreTokensIndex = 14;
constexpr size_t kInputNextTokensIndex = 15;
constexpr size_t kInputAttentionModeIndex = 16;
constexpr size_t kInputReturnSoftmaxLseIndex = 17;

// Output indices
constexpr size_t kOutputAttentionOutIndex = 0;
constexpr size_t kOutputSoftmaxMaxIndex = 1;
constexpr size_t kOutputSoftmaxSumIndex = 2;
}  // namespace

void SparseFlashAttentionAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  scale_value_ = static_cast<double>(device::ascend::ConvertKernelTensor<float>(inputs[kInputScaleValueIndex]));
  sparse_block_size_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputSparseBlockSizeIndex]);
  auto layout_query_int = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputLayoutQueryIndex]);
  auto layout_kv_int = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputLayoutKvIndex]);
  layout_query_ = device::ascend::FASInputLayoutMode::ConvertEnumToString(layout_query_int);
  layout_kv_ = device::ascend::FASInputLayoutMode::ConvertEnumToString(layout_kv_int);
  sparse_mode_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputSparseModeIndex]);
  pre_tokens_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputPreTokensIndex]);
  next_tokens_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputNextTokensIndex]);
  attention_mode_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputAttentionModeIndex]);
  return_softmax_lse_ = device::ascend::ConvertKernelTensor<bool>(inputs[kInputReturnSoftmaxLseIndex]);

  const char *layout_query_ptr = layout_query_.c_str();
  const char *layout_kv_ptr = layout_kv_.c_str();

  GetWorkspaceForResize(
    inputs[kInputQueryIndex], inputs[kInputKeyIndex], inputs[kInputValueIndex], inputs[kInputSparseIndicesIndex],
    scale_value_, inputs[kInputBlockTableIndex], inputs[kInputActualSeqLengthsQueryIndex],
    inputs[kInputActualSeqLengthsKvIndex], inputs[kInputQueryRopeIndex], inputs[kInputKeyRopeIndex], sparse_block_size_,
    layout_query_ptr, layout_kv_ptr, sparse_mode_, pre_tokens_, next_tokens_, attention_mode_, return_softmax_lse_,
    outputs[kOutputAttentionOutIndex], outputs[kOutputSoftmaxMaxIndex], outputs[kOutputSoftmaxSumIndex]);
}

bool SparseFlashAttentionAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  MS_LOG(DEBUG) << "Run aclnnSparseFlashAttention in kernel_mod_impl";

  const char *layout_query_ptr = layout_query_.c_str();
  const char *layout_kv_ptr = layout_kv_.c_str();

  RunOp(stream_ptr, workspace, inputs[kInputQueryIndex], inputs[kInputKeyIndex], inputs[kInputValueIndex],
        inputs[kInputSparseIndicesIndex], scale_value_, inputs[kInputBlockTableIndex],
        inputs[kInputActualSeqLengthsQueryIndex], inputs[kInputActualSeqLengthsKvIndex], inputs[kInputQueryRopeIndex],
        inputs[kInputKeyRopeIndex], sparse_block_size_, layout_query_ptr, layout_kv_ptr, sparse_mode_, pre_tokens_,
        next_tokens_, attention_mode_, return_softmax_lse_, outputs[kOutputAttentionOutIndex],
        outputs[kOutputSoftmaxMaxIndex], outputs[kOutputSoftmaxSumIndex]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SparseFlashAttention, SparseFlashAttentionAscend);

}  // namespace sparse_flash_attention
}  // namespace kernel
}  // namespace mindspore
