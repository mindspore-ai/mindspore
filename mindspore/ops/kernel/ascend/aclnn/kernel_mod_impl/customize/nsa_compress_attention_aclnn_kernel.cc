/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0
 */
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/nsa_compress_attention_aclnn_kernel.h"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
namespace nsa_compress_attention {

void NsaCompressAttentionAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  scale_value_ = static_cast<double>(device::ascend::ConvertKernelTensor<float>(inputs[kIndex3]));
  head_num_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex4]);
  compress_block_size_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex5]);
  compress_stride_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex6]);
  select_block_size_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex7]);
  select_block_count_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex8]);

  actual_seq_qlen_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex11]);
  actual_cmp_seq_kvlen_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex12]);
  actual_sel_seq_kvlen_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex13]);

  actual_seq_qlen_pair_ = std::make_pair(actual_seq_qlen_, true);
  actual_cmp_seq_kvlen_pair_ = std::make_pair(actual_cmp_seq_kvlen_, true);
  actual_sel_seq_kvlen_pair_ = std::make_pair(actual_sel_seq_kvlen_, true);

  input_layout_ = "TND";
  sparse_mode_ = 1;

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex10], inputs[kIndex9],
                        actual_seq_qlen_pair_, actual_cmp_seq_kvlen_pair_, actual_sel_seq_kvlen_pair_, scale_value_,
                        head_num_, input_layout_, sparse_mode_, compress_block_size_, compress_stride_,
                        select_block_size_, select_block_count_, outputs[kIndex2], outputs[kIndex3], outputs[kIndex0],
                        outputs[kIndex1]);
}

bool NsaCompressAttentionAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  MS_LOG(DEBUG) << "Run aclnnNsaCompressAttention in kernel_mod_impl";

  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex10], inputs[kIndex9],
        actual_seq_qlen_pair_, actual_cmp_seq_kvlen_pair_, actual_sel_seq_kvlen_pair_, scale_value_, head_num_,
        input_layout_, sparse_mode_, compress_block_size_, compress_stride_, select_block_size_, select_block_count_,
        outputs[kIndex2], outputs[kIndex3], outputs[kIndex0], outputs[kIndex1]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(NsaCompressAttention, NsaCompressAttentionAscend);

}  // namespace nsa_compress_attention
}  // namespace kernel
}  // namespace mindspore
