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

#include "backend/kernel_compiler/cpu/mkldnn/matmul_cpu_kernel.h"
#include <utility>
#include "common/thread_pool.h"
#include "backend/kernel_compiler/cpu/nnacl/op_base.h"
#include "backend/kernel_compiler/cpu/nnacl/matmul_parameter.h"
#include "backend/kernel_compiler/cpu/nnacl/fp32/matmul_fp32.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMatMulInputsNum = 2;
constexpr size_t kMatMulOutputsNum = 1;
constexpr size_t kIndexOffset = 2;
constexpr size_t kRankMin = 2;
using dims = dnnl::memory::dims;
}  // namespace

void MatMulCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> a_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> b_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> o_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  if (a_shape.size() < kRankMin || b_shape.size() < kRankMin || o_shape.size() < kRankMin) {
    MS_LOG(EXCEPTION) << "The tensor rank of MatMul should be greater than or equal to " << kRankMin;
  }
  bool trans_a = AnfAlgo::GetNodeAttr<bool>(kernel_node, TRANSPOSE_A);
  bool trans_b = AnfAlgo::GetNodeAttr<bool>(kernel_node, TRANSPOSE_B);
  auto rank = a_shape.size();
  int64_t batch = 1;
  for (size_t i = 0; i < rank - kIndexOffset; ++i) {
    batch *= SizeToLong(a_shape[i]);
  }

  int64_t dim_m = SizeToLong(o_shape[rank - kIndexOffset]);
  int64_t dim_n = SizeToLong(o_shape[rank - 1]);
  int64_t dim_k = 1;
  if (trans_a) {
    dim_k = SizeToLong(a_shape[rank - kIndexOffset]);
  } else {
    dim_k = SizeToLong(a_shape[rank - 1]);
  }

  dims src_dims, weights_dims, dst_dims, a_strides, b_strides, o_strides;
  if (batch > 1) {
    src_dims = {batch, dim_m, dim_k};
    weights_dims = {batch, dim_k, dim_n};
    dst_dims = {batch, dim_m, dim_n};
    a_strides = {trans_a ? dims{dim_m * dim_k, 1, dim_m} : dims{dim_m * dim_k, dim_k, 1}};
    b_strides = {trans_b ? dims{dim_n * dim_k, 1, dim_k} : dims{dim_n * dim_k, dim_n, 1}};
    o_strides = {dim_n * dim_m, dim_n, 1};
  } else {
    src_dims = {dim_m, dim_k};
    weights_dims = {dim_k, dim_n};
    dst_dims = {dim_m, dim_n};
    a_strides = {trans_a ? dims{1, dim_m} : dims{dim_k, 1}};
    b_strides = {trans_b ? dims{1, dim_k} : dims{dim_n, 1}};
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
}

bool MatMulCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
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
