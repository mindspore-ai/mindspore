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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_MATMUL_KERNEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_MATMUL_KERNEL_H_
#include <string>
#include <vector>
#include <memory>
#include "extendrt/delegate/ascend_native/ascend_native_base_kernel.h"
namespace mindspore::kernel {
class AscendNativeMatmulKernel : public AscendNativeBaseKernel {
 public:
  AscendNativeMatmulKernel(const std::vector<InferTensor *> &inputs, const std::vector<InferTensor *> &outputs,
                           InferPrimitive prim, const InferContext *ctx, const void *stream, std::string name,
                           const void *acl_ctx_)
      : AscendNativeBaseKernel(inputs, outputs, prim, ctx, stream, name, acl_ctx_) {}

  ~AscendNativeMatmulKernel() {
    if (tile_data_d_) ascend_native::FreeDevice(tile_data_d_, const_cast<void *>(acl_ctx_));
    tile_data_d_ = nullptr;
    if (tile_data_h_) free(tile_data_h_);
    tile_data_h_ = nullptr;
    if (extra_d_) ascend_native::FreeDevice(extra_d_, const_cast<void *>(acl_ctx_));
    extra_d_ = nullptr;
#ifdef ACL_BLAS
    if (alpha_) ascend_native::FreeDevice(alpha_, const_cast<void *>(acl_ctx_));
    if (beta_) ascend_native::FreeDevice(beta_, const_cast<void *>(acl_ctx_));
#endif
  }

  int InferShape() override;

  int Prepare() override;

  int Run() override;

  int ReSize() override;

 private:
  void *tile_data_h_ = nullptr;
  void *tile_data_d_ = nullptr;
  ascend_native::MMExtra extra_h_;
  void *extra_d_ = nullptr;
  bool transpose_a_{false};
  bool transpose_b_{false};
  int m_{0};
  int n_{0};
  int k_{0};
#ifdef ACL_BLAS
  int PrepareBlas();
  void *alpha_;
  void *beta_;
#endif
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_MATMUL_KERNEL_H_
