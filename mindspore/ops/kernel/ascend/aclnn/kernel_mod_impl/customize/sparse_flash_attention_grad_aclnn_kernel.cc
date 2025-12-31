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

#include "kernel/ascend/aclnn/kernel_mod_impl/customize/sparse_flash_attention_grad_aclnn_kernel.h"

#include <algorithm>
#include <vector>
#include <string>

#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"

namespace mindspore {
namespace kernel {
namespace sparse_flash_attention_grad {

namespace {
// Input indices
constexpr size_t kInputQueryIndex = 0;
constexpr size_t kInputKeyIndex = 1;
constexpr size_t kInputValueIndex = 2;
constexpr size_t kInputSparseIndicesIndex = 3;
constexpr size_t kInputDOutIndex = 4;
constexpr size_t kInputOutIndex = 5;
constexpr size_t kInputSoftmaxMaxIndex = 6;
constexpr size_t kInputSoftmaxSumIndex = 7;
constexpr size_t kInputScaleValueIndex = 8;
constexpr size_t kInputSparseBlockSizeIndex = 9;
constexpr size_t kInputQueryRopeIndex = 10;
constexpr size_t kInputKeyRopeIndex = 11;
constexpr size_t kInputActualSeqQlenIndex = 12;
constexpr size_t kInputActualSeqKvlenIndex = 13;
constexpr size_t kInputLayoutIndex = 14;
constexpr size_t kInputSparseModeIndex = 15;
constexpr size_t kInputPreTokensIndex = 16;
constexpr size_t kInputNextTokensIndex = 17;

// Output indices
constexpr size_t kOutputDQueryIndex = 0;
constexpr size_t kOutputDKeyIndex = 1;
constexpr size_t kOutputDValueIndex = 2;
constexpr size_t kOutputDQueryRopeIndex = 3;
constexpr size_t kOutputDKeyRopeIndex = 4;
}  // namespace

void SparseFlashAttentionGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  scale_value_ = static_cast<double>(device::ascend::ConvertKernelTensor<float>(inputs[kInputScaleValueIndex]));
  sparse_block_size_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputSparseBlockSizeIndex]);
  auto layout_int = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputLayoutIndex]);
  layout_ = device::ascend::FASInputLayoutMode::ConvertEnumToString(layout_int);
  sparse_mode_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputSparseModeIndex]);
  pre_tokens_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputPreTokensIndex]);
  next_tokens_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputNextTokensIndex]);

  const char *layout_ptr = layout_.c_str();
  deterministic_ = true;

  GetWorkspaceForResize(inputs[kInputQueryIndex], inputs[kInputKeyIndex], inputs[kInputValueIndex],
                        inputs[kInputSparseIndicesIndex], inputs[kInputDOutIndex], inputs[kInputOutIndex],
                        inputs[kInputSoftmaxMaxIndex], inputs[kInputSoftmaxSumIndex], inputs[kInputActualSeqQlenIndex],
                        inputs[kInputActualSeqKvlenIndex], inputs[kInputQueryRopeIndex], inputs[kInputKeyRopeIndex],
                        scale_value_, sparse_block_size_, layout_ptr, sparse_mode_, pre_tokens_, next_tokens_,
                        deterministic_, outputs[kOutputDQueryIndex], outputs[kOutputDKeyIndex],
                        outputs[kOutputDValueIndex], outputs[kOutputDQueryRopeIndex], outputs[kOutputDKeyRopeIndex]);
}

bool SparseFlashAttentionGradAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  MS_LOG(DEBUG) << "Run aclnnSparseFlashAttentionGrad in kernel_mod_impl";

  const char *layout_ptr = layout_.c_str();

  RunOp(stream_ptr, workspace, inputs[kInputQueryIndex], inputs[kInputKeyIndex], inputs[kInputValueIndex],
        inputs[kInputSparseIndicesIndex], inputs[kInputDOutIndex], inputs[kInputOutIndex],
        inputs[kInputSoftmaxMaxIndex], inputs[kInputSoftmaxSumIndex], inputs[kInputActualSeqQlenIndex],
        inputs[kInputActualSeqKvlenIndex], inputs[kInputQueryRopeIndex], inputs[kInputKeyRopeIndex], scale_value_,
        sparse_block_size_, layout_ptr, sparse_mode_, pre_tokens_, next_tokens_, deterministic_,
        outputs[kOutputDQueryIndex], outputs[kOutputDKeyIndex], outputs[kOutputDValueIndex],
        outputs[kOutputDQueryRopeIndex], outputs[kOutputDKeyRopeIndex]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SparseFlashAttentionGrad, SparseFlashAttentionGradAscend);

}  // namespace sparse_flash_attention_grad
}  // namespace kernel
}  // namespace mindspore
