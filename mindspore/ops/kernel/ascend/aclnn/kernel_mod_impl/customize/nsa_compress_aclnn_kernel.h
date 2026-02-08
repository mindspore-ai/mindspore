/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0
 */
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_NSA_COMPRESS_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_NSA_COMPRESS_ACLNN_KERNEL_MOD_H_

#include <vector>
#include <string>
#include <utility>
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"

namespace mindspore {
namespace kernel {
namespace nsa_compress {

class NsaCompressAscend : public AclnnKernelMod {
 public:
  NsaCompressAscend() : AclnnKernelMod(std::move("aclnnNsaCompress")) {}
  ~NsaCompressAscend() = default;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 private:
  int64_t block_ = 0;
  int64_t stride_ = 0;
  std::vector<int64_t> seq_len_{};
  std::string layout_ = "TND";
  int64_t seq_len_type_ = 0;
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
};

}  // namespace nsa_compress
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_NSA_COMPRESS_ACLNN_KERNEL_MOD_H_
