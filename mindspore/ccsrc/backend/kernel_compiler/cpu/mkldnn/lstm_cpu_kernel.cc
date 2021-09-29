/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/mkldnn/lstm_cpu_kernel.h"
#include <string>
#include "utils/ms_utils.h"
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#ifdef PLATFORM_86
#include <pmmintrin.h>
#endif

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLstmInputsNum = 4;
constexpr size_t kLstmOutputsNum = 5;
constexpr int kMaxLSTMLayer = 100;
constexpr int kOutputWorkSpaceIndex = 3;
constexpr int kGateNum = 4;

using tag = dnnl::memory::format_tag;
using dim = dnnl::memory::dims;
using dt = dnnl::memory::data_type;
}  // namespace

void LstmCPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  output_size_list_[kOutputWorkSpaceIndex] = reserve_size_;
  auto output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  auto output_type = AnfAlgo::GetOutputInferDataType(kernel_node, 0);
  auto output_types = std::vector<TypeId>(output_num, output_type);
  std::vector<std::vector<size_t>> output_shapes;
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    auto shape = AnfAlgo::GetOutputInferShape(kernel_node, output_index);
    (void)output_shapes.emplace_back(shape);
  }
  size_t len = reserve_size_ / IntToSize(kGateNum);
  output_shapes[kOutputWorkSpaceIndex] = {len, 1};
  AnfAlgo::SetOutputInferTypeAndShape(output_types, output_shapes, kernel_node.get());
}

void LstmCPUKernel::InitKernel(const CNodePtr &kernel_node) {
#ifdef PLATFORM_86
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  CheckParam(kernel_node);
  auto eng = MKLKernelEngine::Get().engine();
  dnnl::rnn_direction direction = dnnl::rnn_direction::unidirectional;
  if (bidirectional_) {
    direction = dnnl::rnn_direction::bidirectional_concat;
  }
  dim src_dims = {seq_len_, batch_size_, input_size_};
  dim src_h_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  dim src_c_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  weights_dims_ = {num_layers_, num_directions_, input_size_, kGateNum, hidden_size_};
  weights_h_dims_ = {num_layers_, num_directions_, hidden_size_, kGateNum, hidden_size_};
  bias_dims_ = {num_layers_, num_directions_, kGateNum, hidden_size_};
  dim dst_dims = {seq_len_, batch_size_, static_cast<int64_t>(hidden_size_) * num_directions_};
  dim dst_h_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  dim dst_c_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  dnnl::memory::desc src_desc = formatted_md(src_dims, tag::tnc);
  dnnl::memory::desc src_h_desc = formatted_md(src_h_dims, tag::ldnc);
  dnnl::memory::desc src_c_desc = formatted_md(src_c_dims, tag::ldnc);
  dnnl::memory::desc bias_desc = formatted_md(bias_dims_, tag::ldgo);
  dnnl::memory::desc dst_desc = formatted_md(dst_dims, tag::tnc);
  dnnl::memory::desc dst_h_desc = formatted_md(dst_h_dims, tag::ldnc);
  dnnl::memory::desc dst_c_desc = formatted_md(dst_c_dims, tag::ldnc);
  if (kernel_node->HasAttr(kAttrIsTraining)) {
    is_training = GetValue<bool>(kernel_node->GetAttr(kAttrIsTraining));
  } else {
    is_training = true;
  }
  auto prop_kind = dnnl::prop_kind::forward_training;
  if (!is_training) {
    prop_kind = dnnl::prop_kind::forward_inference;
  }
  auto desc = std::make_shared<dnnl::lstm_forward::desc>(
    prop_kind, direction, src_desc, src_h_desc, src_c_desc, formatted_md(weights_dims_, tag::any),
    formatted_md(weights_h_dims_, tag::any), bias_desc, dst_desc, dst_h_desc, dst_c_desc);
  prim_desc_ = dnnl::lstm_forward::primitive_desc(*desc, eng);
  primitive_ = std::make_shared<dnnl::lstm_forward>(prim_desc_);
  if (is_training) {
    reserve_size_ = static_cast<size_t>(prim_desc_.workspace_desc().get_size());
    AddArgument(DNNL_ARG_WORKSPACE, prim_desc_.workspace_desc());
  } else {
    reserve_size_ = 1;
  }
  AddArgument(DNNL_ARG_SRC_LAYER, src_desc);
  AddArgument(DNNL_ARG_SRC_ITER, src_h_desc);
  AddArgument(DNNL_ARG_SRC_ITER_C, src_c_desc);
  AddArgument(DNNL_ARG_WEIGHTS_LAYER, prim_desc_.weights_layer_desc());
  AddArgument(DNNL_ARG_WEIGHTS_ITER, prim_desc_.weights_iter_desc());
  AddArgument(DNNL_ARG_BIAS, bias_desc);
  AddArgument(DNNL_ARG_DST_LAYER, dst_desc);
  AddArgument(DNNL_ARG_DST_ITER, dst_h_desc);
  AddArgument(DNNL_ARG_DST_ITER_C, dst_c_desc);
}

void LstmCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> src_h_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> src_c_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 2);
  bidirectional_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "bidirectional");
  input_size_ = LongToInt(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "input_size"));
  hidden_size_ = LongToInt(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "hidden_size"));
  num_layers_ = LongToInt(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "num_layers"));
  has_bias_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "has_bias");
  batch_size_ = SizeToInt(src_shape[1]);
  seq_len_ = SizeToInt(src_shape[0]);
  num_directions_ = 1;
  if (bidirectional_) {
    num_directions_ = 2;
  }
  const int gate_size = kGateNum * hidden_size_;
  if (num_layers_ <= 0) {
    MS_LOG(EXCEPTION) << "Layers must be greater than zero!";
  }
  if (num_layers_ > kMaxLSTMLayer) {
    MS_LOG(EXCEPTION) << "Layers must be lower than 100!";
  }
  for (int i = 0; i < num_layers_; ++i) {
    weight_size_ += gate_size * (i == 0 ? input_size_ : hidden_size_ * num_directions_);
    weight_h_size_ += gate_size * hidden_size_;
  }
  weight_size_ = weight_size_ * num_directions_;
  weight_h_size_ = weight_h_size_ * num_directions_;
  if (num_directions_ * num_layers_ != SizeToInt(src_h_shape[0])) {
    MS_LOG(EXCEPTION) << "Error iteration shape!";
  }
  if (src_shape.size() != 3 || src_h_shape.size() != 3 || src_c_shape.size() != 3) {
    MS_LOG(EXCEPTION) << "Lstm only support 3-D input!";
  }
}

bool LstmCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                           const std::vector<kernel::AddressPtr> &outputs) {
  using dt = dnnl::memory::data_type;
  using tag = dnnl::memory::format_tag;
  auto eng = MKLKernelEngine::Get().engine();
  auto user_weights_memory = dnnl::memory(dnnl::memory::desc{{weights_dims_}, dt::f32, tag::ldgoi}, eng);
  auto user_weights_h_memory = dnnl::memory(dnnl::memory::desc{{weights_h_dims_}, dt::f32, tag::ldgoi}, eng);
  auto weights_memory = dnnl::memory(prim_desc_.weights_layer_desc(), eng);
  auto weights_h_memory = dnnl::memory(prim_desc_.weights_iter_desc(), eng);
  user_weights_memory.set_data_handle(inputs[3]->addr);
  user_weights_h_memory.set_data_handle(reinterpret_cast<float *>(inputs[3]->addr) + weight_size_);
  Reorder(&user_weights_memory, &weights_memory);
  Reorder(&user_weights_h_memory, &weights_h_memory);
  auto bias_memory = dnnl::memory(prim_desc_.bias_desc(), eng);
  if (has_bias_) {
    bias_memory.set_data_handle(reinterpret_cast<float *>(inputs[3]->addr) + weight_size_ + weight_h_size_);
  } else {
    if (memset_s(bias_memory.get_data_handle(), prim_desc_.bias_desc().get_size(), 0,
                 prim_desc_.bias_desc().get_size())) {
      MS_LOG(EXCEPTION) << "Bias memset error";
    }
  }
  // set handle
  SetArgumentHandle(DNNL_ARG_SRC_LAYER, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_SRC_ITER, inputs[1]->addr);
  SetArgumentHandle(DNNL_ARG_SRC_ITER_C, inputs[2]->addr);
  SetArgumentHandle(DNNL_ARG_WEIGHTS_LAYER, weights_memory.get_data_handle());
  SetArgumentHandle(DNNL_ARG_WEIGHTS_ITER, weights_h_memory.get_data_handle());
  SetArgumentHandle(DNNL_ARG_BIAS, bias_memory.get_data_handle());
  SetArgumentHandle(DNNL_ARG_DST_LAYER, outputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DST_ITER, outputs[1]->addr);
  SetArgumentHandle(DNNL_ARG_DST_ITER_C, outputs[2]->addr);
  if (is_training) {
    SetArgumentHandle(DNNL_ARG_WORKSPACE, outputs[3]->addr);
  }
  ExecutePrimitive();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
