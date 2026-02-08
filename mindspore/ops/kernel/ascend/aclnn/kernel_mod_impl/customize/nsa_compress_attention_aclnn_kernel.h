/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0
 */
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_NSA_COMPRESS_ATTENTION_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_NSA_COMPRESS_ATTENTION_ACLNN_KERNEL_MOD_H_

#include <vector>
#include <string>
#include <utility>
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"

namespace mindspore {
namespace kernel {
namespace nsa_compress_attention {

class NsaCompressAttentionAscend : public AclnnKernelMod {
 public:
  NsaCompressAttentionAscend() : AclnnKernelMod(std::move("aclnnNsaCompressAttention")) {}
  ~NsaCompressAttentionAscend() = default;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 private:
  double scale_value_ = 1.0;
  int64_t head_num_ = 8;
  int64_t sparse_mode_ = 0;
  int64_t compress_block_size_ = 16;
  int64_t compress_stride_ = 16;
  int64_t select_block_size_ = 16;
  int64_t select_block_count_ = 1;
  std::string input_layout_ = "TND";
  std::vector<int64_t> actual_seq_qlen_;
  std::vector<int64_t> actual_cmp_seq_kvlen_;
  std::vector<int64_t> actual_sel_seq_kvlen_;
  std::pair<std::vector<int64_t>, bool> actual_seq_qlen_pair_{};
  std::pair<std::vector<int64_t>, bool> actual_cmp_seq_kvlen_pair_{};
  std::pair<std::vector<int64_t>, bool> actual_sel_seq_kvlen_pair_{};
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
};

}  // namespace nsa_compress_attention
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_NSA_COMPRESS_ATTENTION_ACLNN_KERNEL_MOD_H_
