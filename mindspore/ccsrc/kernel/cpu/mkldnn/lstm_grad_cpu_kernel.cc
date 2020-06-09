/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "kernel/cpu/mkldnn/lstm_grad_cpu_kernel.h"
#include <cstring>
#include <cmath>
#include <numeric>
#include <string>
#include "common/utils.h"
#include "kernel/cpu/mkldnn/mkl_kernel_engine.h"
#include "device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void LSTMGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  using tag = dnnl::memory::format_tag;
  using dim = dnnl::memory::dims;
  auto eng = MKLKernelEngine::Get().engine();
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> src_h_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  bidirectional_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "bidirectional");
  input_size_ = AnfAlgo::GetNodeAttr<int>(kernel_node, "input_size");
  hidden_size_ = AnfAlgo::GetNodeAttr<int>(kernel_node, "hidden_size");
  num_layers_ = AnfAlgo::GetNodeAttr<int>(kernel_node, "num_layers");
  has_bias_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "has_bias");
  batch_size_ = SizeToInt(src_shape[1]);
  seq_len_ = SizeToInt(src_shape[0]);
  num_directions_ = 1;
  if (bidirectional_) {
    num_directions_ = 2;
  }
  if (num_directions_ * num_layers_ != SizeToInt(src_h_shape[0])) {
    MS_LOG(EXCEPTION) << "error iteration shape!";
  }
  const int gate_size = 4 * hidden_size_;
  for (int i = 0; i < num_layers_; ++i) {
    weight_size_ += gate_size * (i == 0 ? input_size_ : hidden_size_ * num_directions_);
    weight_h_size_ += gate_size * hidden_size_;
  }
  weight_size_ = weight_size_ * num_directions_;
  weight_h_size_ = weight_h_size_ * num_directions_;
  dnnl::rnn_direction direction = dnnl::rnn_direction::unidirectional;
  if (bidirectional_) {
    direction = dnnl::rnn_direction::bidirectional_concat;
  }
  dim src_dims = {seq_len_, batch_size_, input_size_};
  dim src_h_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  dim src_c_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  weights_dims_ = {num_layers_, num_directions_, input_size_, 4, hidden_size_};
  weights_h_dims_ = {num_layers_, num_directions_, hidden_size_, 4, hidden_size_};
  bias_dims_ = {num_layers_, num_directions_, 4, hidden_size_};
  dim dst_dims = {seq_len_, batch_size_, hidden_size_ * num_directions_};
  dim dst_h_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  dim dst_c_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  dnnl::memory::desc src_desc = formatted_md(src_dims, tag::tnc);
  dnnl::memory::desc src_h_desc = formatted_md(src_h_dims, tag::ldnc);
  dnnl::memory::desc src_c_desc = formatted_md(src_c_dims, tag::ldnc);
  dnnl::memory::desc bias_desc = formatted_md(bias_dims_, tag::ldgo);
  dnnl::memory::desc dst_desc = formatted_md(dst_dims, tag::tnc);
  dnnl::memory::desc dst_h_desc = formatted_md(dst_h_dims, tag::ldnc);
  dnnl::memory::desc dst_c_desc = formatted_md(dst_c_dims, tag::ldnc);
  dnnl::lstm_forward::desc forward_desc =
    dnnl::lstm_forward::desc(dnnl::prop_kind::forward_training, direction, src_desc, src_h_desc, src_c_desc,
                             formatted_md(weights_dims_, tag::any), formatted_md(weights_h_dims_, tag::any), bias_desc,
                             dst_desc, dst_h_desc, dst_c_desc);
  auto prim_forward_desc = dnnl::lstm_forward::primitive_desc(forward_desc, eng);
  dnnl::lstm_backward::desc backward_desc = dnnl::lstm_backward::desc(
    dnnl::prop_kind::backward, direction, src_desc, src_h_desc, src_c_desc, formatted_md(weights_dims_, tag::any),
    formatted_md(weights_h_dims_, tag::any), bias_desc, dst_desc, dst_h_desc, dst_c_desc, src_desc, src_h_desc,
    src_c_desc, formatted_md(weights_dims_, tag::any), formatted_md(weights_h_dims_, tag::any), bias_desc, dst_desc,
    dst_h_desc, dst_c_desc);
  prim_backward_desc_ = dnnl::lstm_backward::primitive_desc(backward_desc, eng, prim_forward_desc);
  primitive_ = std::make_shared<dnnl::lstm_backward>(prim_backward_desc_);

  AddArgument(DNNL_ARG_SRC_LAYER, src_desc);
  AddArgument(DNNL_ARG_SRC_ITER, src_h_desc);
  AddArgument(DNNL_ARG_SRC_ITER_C, src_c_desc);
  AddArgument(DNNL_ARG_WEIGHTS_LAYER, prim_backward_desc_.weights_layer_desc());
  AddArgument(DNNL_ARG_WEIGHTS_ITER, prim_backward_desc_.weights_iter_desc());
  AddArgument(DNNL_ARG_BIAS, bias_desc);
  AddArgument(DNNL_ARG_DST_LAYER, dst_desc);
  AddArgument(DNNL_ARG_DST_ITER, dst_h_desc);
  AddArgument(DNNL_ARG_DST_ITER_C, dst_c_desc);
  AddArgument(DNNL_ARG_WORKSPACE, prim_forward_desc.workspace_desc());
  AddArgument(DNNL_ARG_DIFF_SRC_LAYER, src_desc);
  AddArgument(DNNL_ARG_DIFF_SRC_ITER, src_h_desc);
  AddArgument(DNNL_ARG_DIFF_SRC_ITER_C, src_c_desc);
  AddArgument(DNNL_ARG_DIFF_WEIGHTS_LAYER, prim_backward_desc_.diff_weights_layer_desc());
  AddArgument(DNNL_ARG_DIFF_WEIGHTS_ITER, prim_backward_desc_.diff_weights_iter_desc());
  AddArgument(DNNL_ARG_DIFF_BIAS, bias_desc);
  AddArgument(DNNL_ARG_DIFF_DST_LAYER, dst_desc);
  AddArgument(DNNL_ARG_DIFF_DST_ITER, dst_h_desc);
  AddArgument(DNNL_ARG_DIFF_DST_ITER_C, dst_c_desc);
}

bool LSTMGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> &workspace /*workspace*/,
                               const std::vector<kernel::AddressPtr> &outputs) {
  using dt = dnnl::memory::data_type;
  using tag = dnnl::memory::format_tag;
  auto eng = MKLKernelEngine::Get().engine();
  // construct fw memory
  auto user_weights_memory = dnnl::memory(dnnl::memory::desc{{weights_dims_}, dt::f32, tag::ldgoi}, eng);
  auto user_weights_h_memory = dnnl::memory(dnnl::memory::desc{{weights_h_dims_}, dt::f32, tag::ldgoi}, eng);
  auto weights_memory = dnnl::memory(prim_backward_desc_.weights_layer_desc(), eng);
  auto weights_h_memory = dnnl::memory(prim_backward_desc_.weights_iter_desc(), eng);
  auto bias_memory = dnnl::memory(prim_backward_desc_.bias_desc(), eng);
  user_weights_memory.set_data_handle(inputs[3]->addr);
  user_weights_h_memory.set_data_handle(reinterpret_cast<float *>(inputs[3]->addr) + weight_size_);
  Reorder(&user_weights_memory, &weights_memory);
  Reorder(&user_weights_h_memory, &weights_h_memory);
  if (has_bias_) {
    bias_memory.set_data_handle(reinterpret_cast<float *>(inputs[3]->addr) + weight_size_ + weight_h_size_);
  } else {
    std::memset(bias_memory.get_data_handle(), 0, prim_backward_desc_.bias_desc().get_size());
  }
  // construct bw memory
  auto diff_weights_memory = dnnl::memory(prim_backward_desc_.diff_weights_layer_desc(), eng);
  auto diff_weights_h_memory = dnnl::memory(prim_backward_desc_.diff_weights_iter_desc(), eng);
  auto diff_bias_memory = dnnl::memory(prim_backward_desc_.diff_bias_desc(), eng);
  auto user_diff_weights_memory = dnnl::memory(dnnl::memory::desc{{weights_dims_}, dt::f32, tag::ldgoi}, eng);
  auto user_diff_weights_h_memory = dnnl::memory(dnnl::memory::desc{{weights_h_dims_}, dt::f32, tag::ldgoi}, eng);
  user_diff_weights_memory.set_data_handle(outputs[3]->addr);
  user_diff_weights_h_memory.set_data_handle(reinterpret_cast<float *>(outputs[3]->addr) + weight_size_);
  std::memset(user_diff_weights_memory.get_data_handle(), 0, user_diff_weights_memory.get_desc().get_size());
  std::memset(user_diff_weights_h_memory.get_data_handle(), 0, user_diff_weights_h_memory.get_desc().get_size());
  if (has_bias_) {
    diff_bias_memory.set_data_handle(reinterpret_cast<float *>(outputs[3]->addr) + weight_size_ + weight_h_size_);
  }
  std::memset(diff_bias_memory.get_data_handle(), 0, prim_backward_desc_.diff_bias_desc().get_size());
  std::memset(diff_weights_memory.get_data_handle(), 0, diff_weights_memory.get_desc().get_size());
  std::memset(diff_weights_h_memory.get_data_handle(), 0, diff_weights_h_memory.get_desc().get_size());
  SetArgumentHandle(DNNL_ARG_SRC_LAYER, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_SRC_ITER, inputs[1]->addr);
  SetArgumentHandle(DNNL_ARG_SRC_ITER_C, inputs[2]->addr);
  SetArgumentHandle(DNNL_ARG_WEIGHTS_LAYER, weights_memory.get_data_handle());
  SetArgumentHandle(DNNL_ARG_WEIGHTS_ITER, weights_h_memory.get_data_handle());
  SetArgumentHandle(DNNL_ARG_BIAS, bias_memory.get_data_handle());
  SetArgumentHandle(DNNL_ARG_DST_LAYER, inputs[4]->addr);
  SetArgumentHandle(DNNL_ARG_DST_ITER, inputs[5]->addr);
  SetArgumentHandle(DNNL_ARG_DST_ITER_C, inputs[6]->addr);
  SetArgumentHandle(DNNL_ARG_WORKSPACE, inputs[10]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC_LAYER, outputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC_ITER, outputs[1]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC_ITER_C, outputs[2]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_weights_memory.get_data_handle());
  SetArgumentHandle(DNNL_ARG_DIFF_WEIGHTS_ITER, diff_weights_h_memory.get_data_handle());
  SetArgumentHandle(DNNL_ARG_DIFF_BIAS, diff_bias_memory.get_data_handle());
  SetArgumentHandle(DNNL_ARG_DIFF_DST_LAYER, inputs[7]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_DST_ITER, inputs[8]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_DST_ITER_C, inputs[9]->addr);
  ExecutePrimitive();
  Reorder(&diff_weights_memory, &user_diff_weights_memory);
  Reorder(&diff_weights_h_memory, &user_diff_weights_h_memory);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
