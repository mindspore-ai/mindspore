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
#include "backend/kernel_compiler/cpu/mkldnn/pooling_cpu_kernel.h"
#include <string>
#include <algorithm>
#include "utils/ms_utils.h"
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
constexpr size_t kPoolingMinDim = 4;
constexpr size_t kPoolingMaxDim = 5;
constexpr size_t kPoolingOffsetDim = 2;
constexpr size_t kPoolingInputsNum = 1;
constexpr size_t kPoolingOutputsNum = 1;
void PoolingCPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  (void)workspace_size_list_.emplace_back(workspace_size_);
}

void PoolingCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> dst_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  dnnl::memory::desc dst_desc = GetDefaultMemDesc(dst_shape);
  std::vector<int> origin_kernel_sizes;
  std::vector<int> strides;
  std::vector<int64_t> kernel_sizes_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, KERNEL_SIZE);
  std::vector<int64_t> strides_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, STRIDES);
  (void)std::transform(kernel_sizes_me.begin(), kernel_sizes_me.end(), std::back_inserter(origin_kernel_sizes),
                       [](const int64_t &value) { return static_cast<int>(value); });
  (void)std::transform(strides_me.begin(), strides_me.end(), std::back_inserter(strides),
                       [](const int64_t &value) { return static_cast<int>(value); });
  auto dim = origin_kernel_sizes.size();
  if (dim < kPoolingMinDim || dim > kPoolingMaxDim || dim != strides.size()) {
    MS_LOG(EXCEPTION) << "Invalid kernel size " << origin_kernel_sizes.size() << " or stride size " << strides.size();
  }
  std::vector<int> stride;
  dnnl::memory::dims kernels_dims;
  dnnl::memory::dims strides_dims;
  std::vector<size_t> kernel_size;
  std::vector<int> dummy_dilation;
  for (size_t i = 2; i < dim; ++i) {
    (void)stride.emplace_back(strides[i]);
    (void)kernels_dims.emplace_back(origin_kernel_sizes[i]);
    (void)strides_dims.emplace_back(strides[i]);
    (void)kernel_size.emplace_back(IntToSize(origin_kernel_sizes[i]));
    (void)dummy_dilation.emplace_back(1);
  }

  std::vector<int> int_padding_l;
  std::vector<int> int_padding_r;
  const std::string pad_mode = AnfAlgo::GetNodeAttr<std::string>(kernel_node, PAD_MODE);
  GetPadding(kernel_node, pad_mode, src_shape, kernel_size, stride, &int_padding_l, &int_padding_r, dummy_dilation);
  if (int_padding_l.size() != dim - kPoolingOffsetDim || int_padding_r.size() != dim - kPoolingOffsetDim) {
    MS_LOG(EXCEPTION) << "Pooling get padding failed!";
  }
  dnnl::memory::dims padding_l;
  dnnl::memory::dims padding_r;
  for (size_t i = 0; i < dim - kPoolingOffsetDim; ++i) {
    (void)padding_l.emplace_back(int_padding_l[i]);
    (void)padding_r.emplace_back(int_padding_r[i]);
  }
  dnnl::pooling_forward::desc desc =
    dnnl::pooling_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::pooling_max, src_desc, dst_desc,
                                strides_dims, kernels_dims, padding_l, padding_r);
  if (kernel_name_ == prim::kPrimAvgPool->name()) {
    desc = dnnl::pooling_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::pooling_avg, src_desc,
                                       dst_desc, strides_dims, kernels_dims, padding_l, padding_r);
  }
  auto prim_desc = dnnl::pooling_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());
  workspace_size_ = prim_desc.workspace_desc().get_size();
  primitive_ = std::make_shared<dnnl::pooling_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, dst_desc);
  AddArgument(DNNL_ARG_WORKSPACE, prim_desc.workspace_desc());
}

bool PoolingCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                              const std::vector<kernel::AddressPtr> &workspace,
                              const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPoolingInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPoolingOutputsNum, kernel_name_);
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_WORKSPACE, workspace[0]->addr);
  ExecutePrimitive();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
