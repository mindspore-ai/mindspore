/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0
 */
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/nsa_compress_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <string>
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
namespace nsa_compress {

void NsaCompressAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  block_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex2]);
  stride_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  seq_len_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex4]);

  layout_ = "TND";
  seq_len_type_ = 0;
  constexpr size_t kTndInputRank = 3;

  // Runtime checks for common illegal values to provide early diagnostics in KernelMod
  // 1) D must be multiple of 16
  {
    auto *in_tensor = inputs[kIndex0];
    MS_EXCEPTION_IF_NULL(in_tensor);
    const auto in_shape = in_tensor->GetShapeVector();
    if (in_shape.size() == kTndInputRank) {
      const int64_t D = static_cast<int64_t>(in_shape[2]);
      if (D % 16 != 0) {
        MS_LOG(EXCEPTION) << "For 'NsaCompress', the last dimension D must be a multiple of 16, but got D=" << D << ".";
      }
    }
  }
  // 2) actual_seq_len last value must equal T
  if (!seq_len_.empty()) {
    auto *in_tensor = inputs[kIndex0];
    MS_EXCEPTION_IF_NULL(in_tensor);
    const auto in_shape = in_tensor->GetShapeVector();
    if (!in_shape.empty()) {
      const int64_t T = static_cast<int64_t>(in_shape[0]);
      const int64_t last = seq_len_.back();
      if (last != T) {
        MS_LOG(EXCEPTION) << "For 'NsaCompress', the last element of actual_seq_len must equal T. got last=" << last
                          << ", T=" << T << ".";
      }
    }
  }
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], seq_len_, layout_, block_, stride_, seq_len_type_,
                        outputs[kIndex0]);
}

bool NsaCompressAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  MS_LOG(DEBUG) << "Run aclnnNsaCompress in kernel_mod_impl";
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], seq_len_, layout_, block_, stride_, seq_len_type_,
        outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(NsaCompress, NsaCompressAscend);

}  // namespace nsa_compress
}  // namespace kernel
}  // namespace mindspore
