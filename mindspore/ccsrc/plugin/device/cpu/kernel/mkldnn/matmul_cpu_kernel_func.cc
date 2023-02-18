/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/mkldnn/matmul_cpu_kernel_func.h"
#include <utility>
#include <map>
#include <memory>
#include <string>

#include "mindspore/core/ops/mat_mul.h"
#include "include/common/utils/utils.h"
#include "kernel/common_utils.h"
#include "mkldnn/mkl_cpu_kernel.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include "ops/base_operator.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMatMulInputsNum = 2;
constexpr size_t kMatMulOutputsNum = 1;
constexpr size_t kIndexOffset = 2;
constexpr size_t kRankMin = 2;
using dims = dnnl::memory::dims;
}  // namespace

void MatMulCpuKernelFunc::InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MatMul>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast MatMul ops failed!";
  }
  trans_a_ = kernel_ptr->get_transpose_a();
  trans_b_ = kernel_ptr->get_transpose_b();
}

int MatMulCpuKernelFunc::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &) {
  auto a_shape = inputs[kIndex0]->GetShapeVector();
  auto b_shape = inputs[kIndex1]->GetShapeVector();
  auto o_shape = outputs[kIndex0]->GetShapeVector();
  if (a_shape.size() < kRankMin || b_shape.size() < kRankMin || o_shape.size() < kRankMin) {
    MS_LOG(EXCEPTION) << "The tensor rank of MatMul must be greater than or equal to " << kRankMin;
  }
  auto rank = a_shape.size();
  int64_t batch = 1;
  for (size_t i = 0; i < rank - kIndexOffset; ++i) {
    batch *= a_shape[i];
  }

  int64_t dim_m = o_shape[rank - kIndexOffset];
  int64_t dim_n = o_shape[rank - 1];
  int64_t dim_k = 1;
  if (trans_a_) {
    dim_k = a_shape[rank - kIndexOffset];
  } else {
    dim_k = a_shape[rank - 1];
  }

  dims src_dims, weights_dims, dst_dims, a_strides, b_strides, o_strides;
  if (batch > 1) {
    src_dims = {batch, dim_m, dim_k};
    weights_dims = {batch, dim_k, dim_n};
    dst_dims = {batch, dim_m, dim_n};
    a_strides = {trans_a_ ? dims{dim_m * dim_k, 1, dim_m} : dims{dim_m * dim_k, dim_k, 1}};
    b_strides = {trans_b_ ? dims{dim_n * dim_k, 1, dim_k} : dims{dim_n * dim_k, dim_n, 1}};
    o_strides = {dim_n * dim_m, dim_n, 1};
  } else {
    src_dims = {dim_m, dim_k};
    weights_dims = {dim_k, dim_n};
    dst_dims = {dim_m, dim_n};
    a_strides = {trans_a_ ? dims{1, dim_m} : dims{dim_k, 1}};
    b_strides = {trans_b_ ? dims{1, dim_k} : dims{dim_n, 1}};
    o_strides = {dim_n, 1};
  }

  auto src_md = CreateDesc<dnnl::memory::desc>(src_dims, dnnl::memory::data_type::f32, a_strides);
  auto weights_md = CreateDesc<dnnl::memory::desc>(weights_dims, dnnl::memory::data_type::f32, b_strides);
  auto dst_md = CreateDesc<dnnl::memory::desc>(dst_dims, dnnl::memory::data_type::f32, o_strides);
  auto matmul_desc = CreateDesc<dnnl::matmul::desc>(src_md, weights_md, dst_md);
  auto prim_desc = CreateDesc<dnnl::matmul::primitive_desc>(matmul_desc, engine_);
  primitive_ = CreatePrimitive<dnnl::matmul>(prim_desc);

  AddArgument(DNNL_ARG_SRC, src_md);
  AddArgument(DNNL_ARG_WEIGHTS, weights_md);
  AddArgument(DNNL_ARG_DST, dst_md);

  return KRET_OK;
}

bool MatMulCpuKernelFunc::RunFunc(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMatMulInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMatMulOutputsNum, kernel_name_);
  const auto input_a = reinterpret_cast<float *>(inputs[0]->addr);
  const auto input_b = reinterpret_cast<float *>(inputs[1]->addr);
  auto output = reinterpret_cast<float *>(outputs[0]->addr);

  SetArgumentHandle(DNNL_ARG_SRC, input_a);
  SetArgumentHandle(DNNL_ARG_WEIGHTS, input_b);
  SetArgumentHandle(DNNL_ARG_DST, output);
  ExecutePrimitive();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
