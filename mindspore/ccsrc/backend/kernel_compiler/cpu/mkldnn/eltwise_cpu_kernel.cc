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

#include "backend/kernel_compiler/cpu/mkldnn/eltwise_cpu_kernel.h"
#include <string>
#include <unordered_map>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 1;

struct DescParam {
  dnnl::algorithm algorithm{dnnl::algorithm::undef};
  float alpha{0.0f};
  float beta{0.0f};
};
}  // namespace

dnnl::eltwise_forward::desc EltWiseCPUKernel::GetForwardEltwiseDesc(const dnnl::memory::desc src_desc) {
  static const std::unordered_map<std::string, DescParam> eltWiseOpDescMap{
    {prim::kPrimRelu->name(), DescParam{dnnl::algorithm::eltwise_relu}},
    {prim::kPrimRelu6->name(), DescParam{dnnl::algorithm::eltwise_clip, 0.0f, 6.0f}},
    {prim::kPrimAbs->name(), DescParam{dnnl::algorithm::eltwise_abs}},
    {prim::kPrimExp->name(), DescParam{dnnl::algorithm::eltwise_exp}},
    {prim::kPrimLog->name(), DescParam{dnnl::algorithm::eltwise_log}},
    {prim::kPrimSigmoid->name(), DescParam{dnnl::algorithm::eltwise_logistic}},
    {prim::kPrimSqrt->name(), DescParam{dnnl::algorithm::eltwise_sqrt}},
    {prim::kPrimTanh->name(), DescParam{dnnl::algorithm::eltwise_tanh}},
    {prim::kPrimElu->name(), DescParam{dnnl::algorithm::eltwise_elu, 1.0f, 0.0f}},
    {prim::kPrimSoftplus->name(), DescParam{dnnl::algorithm::eltwise_soft_relu}},
  };
  const auto desc_pair = eltWiseOpDescMap.find(kernel_name_);
  if (desc_pair == eltWiseOpDescMap.end()) {
    MS_LOG(EXCEPTION) << "EltWiseCPUKernel does not support " << kernel_name_;
  }
  return dnnl::eltwise_forward::desc(dnnl_forward_, desc_pair->second.algorithm, src_desc, desc_pair->second.alpha,
                                     desc_pair->second.beta);
}

void EltWiseCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (src_shape.empty()) {
    (void)src_shape.insert(src_shape.begin(), 1);
  }
  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);

  auto desc = GetForwardEltwiseDesc(src_desc);
  auto prim_desc = dnnl::eltwise_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());
  primitive_ = std::make_shared<dnnl::eltwise_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, src_desc);
}

bool EltWiseCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                              const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  ExecutePrimitive();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
