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
#include "kernel/cpu/mkldnn/lstm_cpu_kernel.h"
#include <string>
#include "common/utils.h"
#include "kernel/cpu/mkldnn/mkl_kernel_engine.h"
#include "device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void LstmCPUKernel::InitKernel(const CNodePtr &kernel_node) {
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

bool LstmCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> & /*workspace*/,
                           const std::vector<kernel::AddressPtr> &outputs) {
  using dt = dnnl::memory::data_type;
  using tag = dnnl::memory::format_tag;
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
  dnnl::lstm_forward::desc desc = dnnl::lstm_forward::desc(
    dnnl::prop_kind::forward_training, direction, src_desc, src_h_desc, src_c_desc, generic_md(weights_dims),
    generic_md(weights_h_dims), bias_desc, dst_desc, dst_h_desc, dst_c_desc);
  auto prim_desc = dnnl::lstm_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());

  auto workspace_memory = dnnl::memory(prim_desc.workspace_desc(), eng);
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
  auto weights_memory = dnnl::memory(prim_desc.weights_layer_desc(), eng);
  auto weights_h_memory = dnnl::memory(prim_desc.weights_iter_desc(), eng);
  dnnl::reorder(user_weights_memory, weights_memory).execute(s, user_weights_memory, weights_memory);
  dnnl::reorder(user_weights_h_memory, weights_h_memory).execute(s, user_weights_h_memory, weights_h_memory);
  auto dst_memory = dnnl::memory(formatted_md(dst_dims, tag::tnc), eng);
  auto dst_h_memory = dnnl::memory(formatted_md(dst_h_dims, tag::ldnc), eng);
  auto dst_c_memory = dnnl::memory(formatted_md(dst_c_dims, tag::ldnc), eng);
  dnnl::lstm_forward fw_layer(prim_desc);
  workspace_memory.set_data_handle(outputs[3]->addr);
  dst_memory.set_data_handle(outputs[0]->addr);
  dst_h_memory.set_data_handle(outputs[1]->addr);
  dst_c_memory.set_data_handle(outputs[2]->addr);
  fw_layer.execute(s, {{DNNL_ARG_SRC_LAYER, src_memory},
                       {DNNL_ARG_SRC_ITER, src_h_memory},
                       {DNNL_ARG_SRC_ITER_C, src_c_memory},
                       {DNNL_ARG_WEIGHTS_LAYER, weights_memory},
                       {DNNL_ARG_WEIGHTS_ITER, weights_h_memory},
                       {DNNL_ARG_BIAS, bias_memory},
                       {DNNL_ARG_DST_LAYER, dst_memory},
                       {DNNL_ARG_DST_ITER, dst_h_memory},
                       {DNNL_ARG_DST_ITER_C, dst_c_memory},
                       {DNNL_ARG_WORKSPACE, workspace_memory}});
  s.wait();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
