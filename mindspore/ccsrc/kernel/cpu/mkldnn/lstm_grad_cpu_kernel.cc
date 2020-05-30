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
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  bidirectional_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "bidirectional");
  input_size_ = AnfAlgo::GetNodeAttr<int>(kernel_node, "input_size");
  hidden_size_ = AnfAlgo::GetNodeAttr<int>(kernel_node, "hidden_size");
  num_layers_ = AnfAlgo::GetNodeAttr<int>(kernel_node, "num_layers");
  batch_size_ = SizeToInt(src_shape[1]);
  seq_len_ = SizeToInt(src_shape[0]);
  num_directions_ = 1;
  if (bidirectional_) {
    num_directions_ = 2;
  }
  int gate_size = 4 * hidden_size_;
  for (int i = 0; i < num_layers_; ++i) {
    weight_size_ += gate_size * (i == 0 ? input_size_ : hidden_size_ * num_directions_);
    weight_h_size_ += gate_size * hidden_size_;
  }
  weight_size_ = weight_size_ * num_directions_;
  weight_h_size_ = weight_h_size_ * num_directions_;
}

bool LSTMGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> &workspace /*workspace*/,
                               const std::vector<kernel::AddressPtr> &outputs) {
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;
  using dim = dnnl::memory::dims;
  auto eng = MKLKernelEngine::Get().engine();
  dnnl::stream s(eng);
  auto formatted_md = [](dim dimensions, tag layout) { return dnnl::memory::desc{{dimensions}, dt::f32, layout}; };
  auto generic_md = [](dim dimensions) { return dnnl::memory::desc{{dimensions}, dt::f32, tag::any}; };
  dnnl::rnn_direction direction = dnnl::rnn_direction::unidirectional;
  if (bidirectional_) {
    direction = dnnl::rnn_direction::bidirectional_concat;
  }
  dim src_dims = {seq_len_, batch_size_, input_size_};
  dim src_h_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  dim src_c_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  dim weights_dims = {num_layers_, num_directions_, input_size_, 4, hidden_size_};
  dim weights_h_dims = {num_layers_, num_directions_, hidden_size_, 4, hidden_size_};
  dim bias_dims = {num_layers_, num_directions_, 4, hidden_size_};
  dim dst_dims = {seq_len_, batch_size_, hidden_size_ * num_directions_};
  dim dst_h_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  dim dst_c_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  dnnl::memory::desc src_desc = formatted_md(src_dims, tag::tnc);
  dnnl::memory::desc src_h_desc = formatted_md(src_h_dims, tag::ldnc);
  dnnl::memory::desc src_c_desc = formatted_md(src_c_dims, tag::ldnc);
  dnnl::memory::desc bias_desc = formatted_md(bias_dims, tag::ldgo);
  dnnl::memory::desc dst_desc = formatted_md(dst_dims, tag::tnc);
  dnnl::memory::desc dst_h_desc = formatted_md(dst_h_dims, tag::ldnc);
  dnnl::memory::desc dst_c_desc = formatted_md(dst_c_dims, tag::ldnc);
  dnnl::lstm_forward::desc forward_desc = dnnl::lstm_forward::desc(
    dnnl::prop_kind::forward_training, direction, src_desc, src_h_desc, src_c_desc, generic_md(weights_dims),
    generic_md(weights_h_dims), bias_desc, dst_desc, dst_h_desc, dst_c_desc);
  auto prim_forward_desc = dnnl::lstm_forward::primitive_desc(forward_desc, eng);
  dnnl::lstm_backward::desc backward_desc = dnnl::lstm_backward::desc(
    dnnl::prop_kind::backward, direction, src_desc, src_h_desc, src_c_desc, generic_md(weights_dims),
    generic_md(weights_h_dims), bias_desc, dst_desc, dst_h_desc, dst_c_desc, src_desc, src_h_desc, src_c_desc,
    generic_md(weights_dims), generic_md(weights_h_dims), bias_desc, dst_desc, dst_h_desc, dst_c_desc);
  auto prim_backward_desc = dnnl::lstm_backward::primitive_desc(backward_desc, eng, prim_forward_desc);

  // construct fw memory
  auto src_memory = dnnl::memory(formatted_md(src_dims, tag::tnc), eng);
  src_memory.set_data_handle(inputs[0]->addr);
  auto src_h_memory = dnnl::memory(formatted_md(src_h_dims, tag::ldnc), eng);
  auto src_c_memory = dnnl::memory(formatted_md(src_c_dims, tag::ldnc), eng);
  src_h_memory.set_data_handle(inputs[1]->addr);
  src_c_memory.set_data_handle(inputs[2]->addr);
  auto user_weights_memory = dnnl::memory(formatted_md(weights_dims, tag::ldgoi), eng);
  auto user_weights_h_memory = dnnl::memory(formatted_md(weights_h_dims, tag::ldgoi), eng);
  auto bias_memory = dnnl::memory(formatted_md(bias_dims, tag::ldgo), eng);
  user_weights_memory.set_data_handle(inputs[3]->addr);
  user_weights_h_memory.set_data_handle(reinterpret_cast<float *>(inputs[3]->addr) + weight_size_);
  bias_memory.set_data_handle(reinterpret_cast<float *>(inputs[3]->addr) + weight_size_ + weight_h_size_);
  auto weights_memory = dnnl::memory(prim_backward_desc.weights_layer_desc(), eng);
  auto weights_h_memory = dnnl::memory(prim_backward_desc.weights_iter_desc(), eng);
  std::vector<float> net_x(src_memory.get_desc().get_size(), 0.0f);
  std::vector<float> net_w(weights_memory.get_desc().get_size(), 0.0f);
  std::vector<float> net_wh(weights_h_memory.get_desc().get_size(), 0.0f);
  dnnl::reorder(user_weights_memory, weights_memory).execute(s, user_weights_memory, weights_memory);
  dnnl::reorder(user_weights_h_memory, weights_h_memory).execute(s, user_weights_h_memory, weights_h_memory);
  auto dst_memory = dnnl::memory(formatted_md(dst_dims, tag::tnc), eng);
  dst_memory.set_data_handle(inputs[4]->addr);
  auto dst_h_memory = dnnl::memory(formatted_md(dst_h_dims, tag::ldnc), eng);
  auto dst_c_memory = dnnl::memory(formatted_md(dst_c_dims, tag::ldnc), eng);
  dst_h_memory.set_data_handle(inputs[5]->addr);
  dst_c_memory.set_data_handle(inputs[6]->addr);
  auto workspace_memory = dnnl::memory(prim_forward_desc.workspace_desc(), eng);
  workspace_memory.set_data_handle(inputs[10]->addr);
  auto diff_src_memory = dnnl::memory(formatted_md(src_dims, tag::tnc), eng);
  auto diff_src_h_memory = dnnl::memory(formatted_md(src_h_dims, tag::ldnc), eng);
  auto diff_src_c_memory = dnnl::memory(formatted_md(src_c_dims, tag::ldnc), eng);
  auto user_diff_weights_memory = dnnl::memory(formatted_md(weights_dims, tag::ldgoi), eng);
  auto user_diff_weights_h_memory = dnnl::memory(formatted_md(weights_h_dims, tag::ldgoi), eng);
  auto diff_weights_memory = dnnl::memory(prim_backward_desc.diff_weights_layer_desc(), eng);
  auto diff_weights_h_memory = dnnl::memory(prim_backward_desc.diff_weights_iter_desc(), eng);
  write_to_dnnl_memory(net_w.data(), diff_weights_memory);
  write_to_dnnl_memory(net_wh.data(), diff_weights_h_memory);
  auto diff_bias_memory = dnnl::memory(formatted_md(bias_dims, tag::ldgo), eng);
  auto diff_dst_memory = dnnl::memory(formatted_md(dst_dims, tag::tnc), eng);
  diff_dst_memory.set_data_handle(inputs[7]->addr);
  auto diff_dst_h_memory = dnnl::memory(formatted_md(dst_h_dims, tag::ldnc), eng);
  diff_dst_h_memory.set_data_handle(inputs[8]->addr);
  auto diff_dst_c_memory = dnnl::memory(formatted_md(dst_c_dims, tag::ldnc), eng);
  diff_dst_c_memory.set_data_handle(inputs[9]->addr);
  diff_src_memory.set_data_handle(outputs[0]->addr);
  diff_src_h_memory.set_data_handle(outputs[1]->addr);
  diff_src_c_memory.set_data_handle(outputs[2]->addr);
  user_diff_weights_memory.set_data_handle(outputs[3]->addr);
  user_diff_weights_h_memory.set_data_handle(reinterpret_cast<float *>(outputs[3]->addr) + weight_size_);
  diff_bias_memory.set_data_handle(reinterpret_cast<float *>(outputs[3]->addr) + weight_size_ + weight_h_size_);
  write_to_dnnl_memory(net_w.data(), user_diff_weights_memory);
  write_to_dnnl_memory(net_wh.data(), user_diff_weights_h_memory);
  write_to_dnnl_memory(net_w.data(), diff_bias_memory);
  write_to_dnnl_memory(net_x.data(), diff_src_memory);
  write_to_dnnl_memory(net_x.data(), diff_src_h_memory);
  write_to_dnnl_memory(net_x.data(), diff_src_c_memory);
  dnnl::lstm_backward bwd_layer(prim_backward_desc);
  bwd_layer.execute(s, {{DNNL_ARG_SRC_LAYER, src_memory},
                        {DNNL_ARG_SRC_ITER, src_h_memory},
                        {DNNL_ARG_SRC_ITER_C, src_c_memory},
                        {DNNL_ARG_WEIGHTS_LAYER, weights_memory},
                        {DNNL_ARG_WEIGHTS_ITER, weights_h_memory},
                        {DNNL_ARG_BIAS, bias_memory},
                        {DNNL_ARG_DST_LAYER, dst_memory},
                        {DNNL_ARG_DST_ITER, dst_h_memory},
                        {DNNL_ARG_DST_ITER_C, dst_c_memory},
                        {DNNL_ARG_DIFF_SRC_LAYER, diff_src_memory},
                        {DNNL_ARG_DIFF_SRC_ITER, diff_src_h_memory},
                        {DNNL_ARG_DIFF_SRC_ITER_C, diff_src_c_memory},
                        {DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_weights_memory},
                        {DNNL_ARG_DIFF_WEIGHTS_ITER, diff_weights_h_memory},
                        {DNNL_ARG_DIFF_BIAS, diff_bias_memory},
                        {DNNL_ARG_DIFF_DST_LAYER, diff_dst_memory},
                        {DNNL_ARG_DIFF_DST_ITER, diff_dst_h_memory},
                        {DNNL_ARG_DIFF_DST_ITER_C, diff_dst_c_memory},
                        {DNNL_ARG_WORKSPACE, workspace_memory}});
  dnnl::reorder(diff_weights_memory, user_diff_weights_memory)
    .execute(s, diff_weights_memory, user_diff_weights_memory);
  dnnl::reorder(diff_weights_h_memory, user_diff_weights_h_memory)
    .execute(s, diff_weights_h_memory, user_diff_weights_h_memory);
  s.wait();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
