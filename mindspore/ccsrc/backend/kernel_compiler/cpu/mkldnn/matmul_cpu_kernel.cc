/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/mkldnn/matmul_cpu_kernel.h"
#include <utility>
#include "common/thread_pool.h"
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "backend/kernel_compiler/cpu/nnacl/op_base.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMatMulInputsNum = 2;
constexpr size_t kMatMulOutputsNum = 1;
const size_t kIndexOffset = 2;
}  // namespace
void MatMulCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> a_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> b_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> o_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  const size_t rank_min = 2;
  if (a_shape.size() < rank_min || b_shape.size() < rank_min || o_shape.size() < rank_min) {
    MS_LOG(EXCEPTION) << "The tensor rank of MatMul should be greater than or equal to 2.";
  }
  bool trans_a = AnfAlgo::GetNodeAttr<bool>(kernel_node, TRANSPOSE_A);
  bool trans_b = AnfAlgo::GetNodeAttr<bool>(kernel_node, TRANSPOSE_B);
  rank_ = a_shape.size();
  batch_ = 1;
  for (size_t i = 0; i < rank_ - kIndexOffset; ++i) {
    batch_ *= a_shape[i];
  }
  size_mat_a_ = a_shape[rank_ - kIndexOffset] * a_shape[rank_ - 1];
  size_mat_b_ = b_shape[rank_ - kIndexOffset] * b_shape[rank_ - 1];
  size_mat_o_ = o_shape[rank_ - kIndexOffset] * o_shape[rank_ - 1];
  if (trans_a) {
    trans_a_ = TRANSPOSE_YES;
    dim_k_ = static_cast<dnnl_dim_t>(a_shape[rank_ - kIndexOffset]);
  } else {
    dim_k_ = static_cast<dnnl_dim_t>(a_shape[rank_ - 1]);
  }
  if (trans_b) {
    trans_b_ = TRANSPOSE_YES;
  }
  dim_m_ = static_cast<dnnl_dim_t>(o_shape[rank_ - kIndexOffset]);
  dim_n_ = static_cast<dnnl_dim_t>(o_shape[rank_ - 1]);
}

bool MatMulCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                             const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatMulInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMatMulOutputsNum, kernel_name_);
  const auto input_a = reinterpret_cast<float *>(inputs[0]->addr);
  const auto input_b = reinterpret_cast<float *>(inputs[1]->addr);
  auto output = reinterpret_cast<float *>(outputs[0]->addr);

  dnnl_dim_t lda = (trans_a_ == TRANSPOSE_YES ? dim_m_ : dim_k_);
  dnnl_dim_t ldb = (trans_b_ == TRANSPOSE_YES ? dim_k_ : dim_n_);
  dnnl_dim_t ldc = dim_n_;
  float alpha = 1.0;
  float beta = 0.0;
  for (size_t i = 0; i < batch_; i++) {
    (void)dnnl_sgemm(trans_a_, trans_b_, dim_m_, dim_n_, dim_k_, alpha, input_a + i * size_mat_a_, lda,
                     input_b + i * size_mat_b_, ldb, beta, output + i * size_mat_o_, ldc);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
