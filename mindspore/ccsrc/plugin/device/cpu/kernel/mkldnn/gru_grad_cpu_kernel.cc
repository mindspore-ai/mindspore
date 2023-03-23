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

#include "plugin/device/cpu/kernel/mkldnn/gru_grad_cpu_kernel.h"
#include <cstddef>
#include <cstring>
#include <string>
#include "utils/ms_utils.h"
#include "mindspore/core/ops/grad/gru_v2_grad.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kGruGradInputsNum = 9;
constexpr size_t kGruGradOutputsNum = 3;
constexpr size_t kNumberOne = 1;
constexpr size_t kNumberTwo = 2;
constexpr size_t kGateNum = 3;
constexpr size_t kDims = 3;
constexpr int kMaxGRULayer = 100;

constexpr int kSrcLayerIdx = 0;
constexpr int kSrcIterIdx = 1;
constexpr int kDstLayerIdx = 4;
constexpr int kDstIterIdx = 5;
constexpr int kWorkSpaceIdx = 8;
constexpr int kDiffSrcLayerIdx = 0;
constexpr int kDiffSrcIterIdx = 1;
constexpr int kDiffDstLayerIdx = 6;
constexpr int kDiffDstIterIdx = 7;

using tag = dnnl::memory::format_tag;
using dim = dnnl::memory::dims;
using dt = dnnl::memory::data_type;
}  // namespace

bool GRUGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGruGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGruGradOutputsNum, kernel_name_);
  auto op_prim = std::dynamic_pointer_cast<ops::GRUV2Grad>(base_operator);
  MS_EXCEPTION_IF_NULL(op_prim);
  bidirectional_ = op_prim->get_bidirectional();
  input_size_ = op_prim->get_input_size();
  hidden_size_ = op_prim->get_hidden_size();
  num_layers_ = op_prim->get_num_layers();
  has_bias_ = op_prim->get_has_bias();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int GRUGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_size_list_[kIndex8] = reserve_size_;
  auto src_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
  auto src_h_shape = inputs[kIndex1]->GetDeviceShapeAdaptively();
  if (src_shape.size() != kDims || src_h_shape.size() != kDims) {
    MS_LOG(ERROR) << "GRU only support 3-D input!,but the src_shape dim is" << src_shape.size()
                  << ", the src_shape dim is" << src_h_shape.size();
    return KRET_RESIZE_FAILED;
  }
  batch_size_ = src_shape[1];
  seq_len_ = src_shape[0];
  num_directions_ = kNumberOne;
  if (bidirectional_) {
    num_directions_ = kNumberTwo;
  }
  const int64_t gate_size = kGateNum * hidden_size_;
  if (num_layers_ <= 0) {
    MS_LOG(ERROR) << "Layers must be greater than zero! but the num_layers is " << num_layers_;
    return KRET_RESIZE_FAILED;
  }
  if (num_layers_ > kMaxGRULayer) {
    MS_LOG(ERROR) << "Layers must be less than or equal to 100! but the num_layers_ is " << num_layers_;
    return KRET_RESIZE_FAILED;
  }

  for (int i = 0; i < num_layers_; ++i) {
    weight_size_ += gate_size * (i == 0 ? input_size_ : hidden_size_ * num_directions_);
    weight_h_size_ += gate_size * hidden_size_;
  }
  weight_size_ = weight_size_ * num_directions_;
  weight_h_size_ = weight_h_size_ * num_directions_;

  weights_dims_ = {num_layers_, num_directions_, input_size_, kGateNum, hidden_size_};
  weights_h_dims_ = {num_layers_, num_directions_, hidden_size_, kGateNum, hidden_size_};
  bias_dims_ = {num_layers_, num_directions_, kGateNum, hidden_size_};

  if (num_directions_ * num_layers_ != src_h_shape[0]) {
    MS_LOG(ERROR) << "Error iteration shape!, iteration shape[0] is required to be " << num_directions_ * num_layers_
                  << " but " << src_h_shape[0];
    return KRET_RESIZE_FAILED;
  }
  InitDnnl();
  return KRET_OK;
}

void GRUGradCpuKernelMod::InitDnnl() {
  auto eng = engine_;
  dnnl::rnn_direction direction =
    bidirectional_ ? dnnl::rnn_direction::bidirectional_concat : dnnl::rnn_direction::unidirectional;
  dim src_h_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  dim src_dims = {seq_len_, batch_size_, input_size_};
  weights_dims_ = {num_layers_, num_directions_, input_size_, kGateNum, hidden_size_};
  weights_h_dims_ = {num_layers_, num_directions_, hidden_size_, kGateNum, hidden_size_};
  bias_dims_ = {num_layers_, num_directions_, kGateNum, hidden_size_};
  dim dst_dims = {seq_len_, batch_size_, static_cast<int64_t>(hidden_size_) * num_directions_};
  dim dst_h_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};

  dnnl::memory::desc src_desc = formatted_md(src_dims, tag::tnc);
  dnnl::memory::desc src_h_desc = formatted_md(src_h_dims, tag::ldnc);
  dnnl::memory::desc bias_desc = formatted_md(bias_dims_, tag::ldgo);
  dnnl::memory::desc dst_desc = formatted_md(dst_dims, tag::tnc);
  dnnl::memory::desc dst_h_desc = formatted_md(dst_h_dims, tag::ldnc);

  auto weights_desc = formatted_md(weights_dims_, tag::any);
  auto weights_h_desc = formatted_md(weights_h_dims_, tag::any);

  auto forward_desc =
    CreatePrimitive<dnnl::gru_forward::desc>(dnnl::prop_kind::forward_training, direction, src_desc, src_h_desc,
                                             weights_desc, weights_h_desc, bias_desc, dst_desc, dst_h_desc);

  auto prim_forward_desc = CreateDesc<dnnl::gru_forward::primitive_desc>(*forward_desc, eng);
  auto backward_desc = CreatePrimitive<dnnl::gru_backward::desc>(
    dnnl::prop_kind::backward, direction, src_desc, src_h_desc, weights_desc, weights_h_desc, bias_desc, dst_desc,
    dst_h_desc, src_desc, src_h_desc, weights_desc, weights_h_desc, bias_desc, dst_desc, dst_h_desc);
  prim_backward_desc_ = CreateDesc<dnnl::gru_backward::primitive_desc>(*backward_desc, eng, prim_forward_desc);
  primitive_ = CreatePrimitive<dnnl::gru_backward>(prim_backward_desc_);
  auto wksp_desc = GetWorkspaceDesc(prim_forward_desc);
  reserve_size_ = GetSize(wksp_desc);
  AddArgumentOp(src_desc, src_h_desc, bias_desc, dst_desc, dst_h_desc, wksp_desc);

  // construct fw memory
  weights_layer_desc_ = GetWeightsLayerDesc(prim_backward_desc_);
  weights_iter_desc_ = GetWeightsIterDesc(prim_backward_desc_);
  bias_desc_ = GetBiasDesc(prim_backward_desc_);
  auto weights_mem_desc = CreateDesc<dnnl::memory::desc>(weights_dims_, dt::f32, tag::ldgoi);
  auto weights_h_mem_desc = CreateDesc<dnnl::memory::desc>(weights_h_dims_, dt::f32, tag::ldgoi);
  user_weights_memory_ = CreateDesc<dnnl::memory>(weights_mem_desc, eng);
  user_weights_h_memory_ = CreateDesc<dnnl::memory>(weights_h_mem_desc, eng);
  weights_memory_ = CreateDesc<dnnl::memory>(weights_layer_desc_, eng);
  weights_h_memory_ = CreateDesc<dnnl::memory>(weights_iter_desc_, eng);
  bias_memory_ = CreateDesc<dnnl::memory>(bias_desc_, eng);

  // construct bw memory
  diff_weights_layer_desc_ = GetDiffWeightsLayerDesc(prim_backward_desc_);
  diff_weights_iter_desc_ = GetDiffWeightsIterDesc(prim_backward_desc_);
  diff_bias_desc_ = GetDiffBiasDesc(prim_backward_desc_);
  diff_weights_memory_ = CreateDesc<dnnl::memory>(diff_weights_layer_desc_, eng);
  diff_weights_h_memory_ = CreateDesc<dnnl::memory>(diff_weights_iter_desc_, eng);
  diff_bias_memory_ = CreateDesc<dnnl::memory>(diff_bias_desc_, eng);
  user_diff_weights_memory_ = CreateDesc<dnnl::memory>(weights_mem_desc, eng);
  user_diff_weights_h_memory_ = CreateDesc<dnnl::memory>(weights_h_mem_desc, eng);
}

void GRUGradCpuKernelMod::AddArgumentOp(const dnnl::memory::desc &src_desc, const dnnl::memory::desc &src_h_desc,
                                        const dnnl::memory::desc &bias_desc, const dnnl::memory::desc &dst_desc,
                                        const dnnl::memory::desc &dst_h_desc, const dnnl::memory::desc &wksp_desc) {
  AddArgument(DNNL_ARG_SRC_LAYER, src_desc);
  AddArgument(DNNL_ARG_SRC_ITER, src_h_desc);
  AddArgument(DNNL_ARG_WEIGHTS_LAYER, weights_layer_desc_);
  AddArgument(DNNL_ARG_WEIGHTS_ITER, weights_iter_desc_);
  AddArgument(DNNL_ARG_BIAS, bias_desc);
  AddArgument(DNNL_ARG_DST_LAYER, dst_desc);
  AddArgument(DNNL_ARG_DST_ITER, dst_h_desc);
  AddArgument(DNNL_ARG_DIFF_SRC_LAYER, src_desc);
  AddArgument(DNNL_ARG_DIFF_SRC_ITER, src_h_desc);
  AddArgument(DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_weights_layer_desc_);
  AddArgument(DNNL_ARG_DIFF_WEIGHTS_ITER, diff_weights_iter_desc_);
  AddArgument(DNNL_ARG_DIFF_BIAS, bias_desc);
  AddArgument(DNNL_ARG_DIFF_DST_LAYER, dst_desc);
  AddArgument(DNNL_ARG_DIFF_DST_ITER, dst_h_desc);
  AddArgument(DNNL_ARG_WORKSPACE, wksp_desc);
}

void GRUGradCpuKernelMod::SetArgumentHandleOp(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  SetArgumentHandle(DNNL_ARG_SRC_LAYER, inputs[kSrcLayerIdx]->addr);
  SetArgumentHandle(DNNL_ARG_SRC_ITER, inputs[kSrcIterIdx]->addr);
  SetArgumentHandle(DNNL_ARG_WEIGHTS_LAYER, GetDataHandle(weights_memory_));
  SetArgumentHandle(DNNL_ARG_WEIGHTS_ITER, GetDataHandle(weights_h_memory_));
  SetArgumentHandle(DNNL_ARG_BIAS, GetDataHandle(bias_memory_));
  SetArgumentHandle(DNNL_ARG_DST_LAYER, inputs[kDstLayerIdx]->addr);
  SetArgumentHandle(DNNL_ARG_DST_ITER, inputs[kDstIterIdx]->addr);
  SetArgumentHandle(DNNL_ARG_WORKSPACE, inputs[kWorkSpaceIdx]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC_LAYER, outputs[kDiffSrcLayerIdx]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC_ITER, outputs[kDiffSrcIterIdx]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_WEIGHTS_LAYER, GetDataHandle(diff_weights_memory_));
  SetArgumentHandle(DNNL_ARG_DIFF_WEIGHTS_ITER, GetDataHandle(diff_weights_h_memory_));
  SetArgumentHandle(DNNL_ARG_DIFF_BIAS, GetDataHandle(diff_bias_memory_));
  SetArgumentHandle(DNNL_ARG_DIFF_DST_LAYER, inputs[kDiffDstLayerIdx]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_DST_ITER, inputs[kDiffDstIterIdx]->addr);
}

void GRUGradCpuKernelMod::ResetMemory(const dnnl::memory &mem, const string name) const {
  auto dst_ptr = GetDataHandle(mem);
  auto mem_desc = GetMemDesc(mem);
  auto size = GetSize(mem_desc);
  if (memset_s(dst_ptr, size, 0, size) != EOK) {
    MS_LOG(EXCEPTION) << name << " memset error";
  }
}

bool GRUGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                 const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGruGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGruGradOutputsNum, kernel_name_);
  SetDataHandle(user_weights_memory_, inputs[kIndex2]->addr);
  SetDataHandle(user_weights_h_memory_, reinterpret_cast<float *>(inputs[kIndex2]->addr) + weight_size_);
  Reorder(&user_weights_memory_, &weights_memory_);
  Reorder(&user_weights_h_memory_, &weights_h_memory_);
  if (has_bias_) {
    SetDataHandle(bias_memory_, reinterpret_cast<float *>(inputs[kIndex2]->addr) + weight_size_ + weight_h_size_);
  } else {
    auto dst_ptr = GetDataHandle(bias_memory_);
    auto size = GetSize(bias_desc_);
    if (memset_s(dst_ptr, size, 0, size) != EOK) {
      MS_LOG(EXCEPTION) << "Bias memset error";
    }
  }

  SetDataHandle(user_diff_weights_memory_, outputs[kIndex2]->addr);
  SetDataHandle(user_diff_weights_h_memory_, reinterpret_cast<float *>(outputs[kIndex2]->addr) + weight_size_);
  ResetMemory(user_diff_weights_memory_, "user weights grad");
  ResetMemory(user_diff_weights_h_memory_, "user weights iter grad");
  ResetMemory(diff_weights_memory_, "weights grad");
  ResetMemory(diff_weights_h_memory_, "weights iter grad");
  if (has_bias_) {
    SetDataHandle(diff_bias_memory_, reinterpret_cast<float *>(outputs[kIndex2]->addr) + weight_size_ + weight_h_size_);
  }
  auto dst_ptr = GetDataHandle(diff_bias_memory_);
  auto size = GetSize(diff_bias_desc_);
  if (memset_s(dst_ptr, size, 0, size) != EOK) {
    MS_LOG(EXCEPTION) << "Bias grad memset error";
  }
  SetArgumentHandleOp(inputs, outputs);
  ExecutePrimitive();
  Reorder(&diff_weights_memory_, &user_diff_weights_memory_);
  Reorder(&diff_weights_h_memory_, &user_diff_weights_h_memory_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GRUV2Grad, GRUGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
