/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/fse_decode_kernel.h"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <thread>
#include <functional>
#include "proto/aicpu_tensor.pb.h"
#include "aicpu_sharder/aicpu_sharder.h"
#include "mindspore/core/mindapi/base/type_id.h"

namespace aicpu {
namespace {
const size_t hw_h = 1;
const size_t hw_w = 2;
const size_t fnz_w1 = 4;
const size_t fnz_h1 = 3;
const size_t fnz_h0 = 2;
const size_t fnz_w0 = 1;
const size_t C0NUM = 0;
const size_t C1NUM = 1;
const size_t C2NUM = 2;
const size_t C3NUM = 3;
const size_t C4NUM = 4;
const size_t C5NUM = 5;
const size_t C6NUM = 6;
const size_t C7NUM = 7;

bool TransShapeToHW_NZ(const std::vector<int> &host_shape, std::vector<int> *hw_shape) {
  if (host_shape.empty()) {
    return false;
  }
  switch (host_shape.size()) {
    case 1:
      hw_shape->push_back(1);
      hw_shape->push_back(1);
      hw_shape->push_back(host_shape[0]);
      return true;
    default:
      auto size = host_shape.size();
      if (size < C2NUM) {
        return false;
      }
      int64_t times = 1;
      for (size_t i = 0; i != size - C2NUM; i++) {
        times *= host_shape[i];
      }
      hw_shape->push_back(times);
      hw_shape->push_back(host_shape[size - C2NUM]);
      hw_shape->push_back(host_shape[size - 1]);
      return true;
  }
}

int NCHW_TO_FRAC_NZ(const std::vector<int> &host_shape, const std::vector<int> &device_shape,
                    std::vector<int> *result) {
  std::vector<int> hw_shape;
  if (!TransShapeToHW_NZ(host_shape, &hw_shape)) {
    return -1;
  }
  auto element_num =
    std::accumulate(device_shape.begin(), device_shape.end(), static_cast<int>(1), std::multiplies<int>());
  result->resize(element_num);
  auto times = hw_shape.at(0);
  auto h = hw_shape.at(hw_h);
  auto w = hw_shape.at(hw_w);
  auto hw = h * w;

  auto shape_size = device_shape.size();
  auto w1 = device_shape[shape_size - fnz_w1];
  auto h1 = device_shape[shape_size - fnz_h1];
  auto h0 = device_shape[shape_size - fnz_h0];
  auto w0 = device_shape[shape_size - fnz_w0];
  auto h1h0w0 = h1 * h0 * w0;
  auto w1h1h0w0 = w1 * h1h0w0;
  auto num_w1 = w / w0;

  for (int64_t times_idx = 0; times_idx < times; times_idx++) {
    auto times_head = times_idx * w1h1h0w0;
    auto src_times_head = times_idx * hw;
    for (int64_t h1h0_idx = 0; h1h0_idx < h; h1h0_idx++) {
      auto h1h0_head = times_head + h1h0_idx * w0;
      auto src_h_head = src_times_head + h1h0_idx * w;
      for (int64_t w1_idx = 0; w1_idx < num_w1; w1_idx++) {
        for (int64_t i = 0; i < w0; ++i) {
          int64_t src_idx = src_h_head + w1_idx * w0 + i;
          int64_t dst_idx = h1h0_head + w1_idx * h1h0w0 + i;
          result->at(src_idx) = dst_idx;
        }
      }
      auto w1_head = num_w1 * w0;
      for (int64_t w0_idx = 0; w1_head + w0_idx < w; w0_idx++) {
        auto src_w_idx = w1_head + w0_idx;
        int64_t dst_idx = h1h0_head + num_w1 * h1h0w0 + w0_idx;
        int64_t src_idx = src_h_head + src_w_idx;
        result->at(src_idx) = dst_idx;
      }
    }
  }
  return 0;
}

std::vector<int> GetShape(const ::aicpuops::TensorShape &shape) {
  std::vector<int> res;
  for (int i = 0; i < shape.dim_size(); ++i) {
    res.push_back(shape.dim(i).size());
  }
  return res;
}
}  // namespace
bool FSEDecodeKernel::CheckParams() const { return true; }

uint64_t FSEDecodeKernel::Pop(const uint64_t *chunks, uint64_t bit_count) {
  const int kMaxBitCount = 64;
  uint64_t right = curr_chunk_ >> static_cast<size_t>(kMaxBitCount - curr_bit_count_);
  uint64_t res = right & ((1u << bit_count) - 1);
  curr_bit_count_ -= static_cast<int8_t>(bit_count);
  if (curr_bit_count_ > 0) {
    return res;
  }
  if (curr_bit_count_ == 0) {
    if (curr_chunk_index_ > -1) {
      curr_bit_count_ = kMaxBitCount;
      curr_chunk_ = chunks[curr_chunk_index_--];
    }
    return res;
  }
  curr_bit_count_ += static_cast<int8_t>(bit_count);
  curr_chunk_ = chunks[curr_chunk_index_--];
  right |= (curr_chunk_ & ((1u << (static_cast<int8_t>(bit_count) - curr_bit_count_)) - 1)) << curr_bit_count_;
  curr_bit_count_ = kMaxBitCount - (static_cast<int8_t>(bit_count) - curr_bit_count_);
  return right;
}

uint32_t FSEDecodeKernel::FixedBitFloatDequantTask() {
  uint64_t *chunks = reinterpret_cast<uint64_t *>(io_addrs_[C0NUM]);
  uint16_t *states_table = reinterpret_cast<uint16_t *>(io_addrs_[C1NUM]);
  uint8_t *bit_count_table = reinterpret_cast<uint8_t *>(io_addrs_[C2NUM]);
  uint16_t *symbol_table = reinterpret_cast<uint16_t *>(io_addrs_[C3NUM]);
  float *centroids = reinterpret_cast<float *>(io_addrs_[C4NUM]);
  float *output = reinterpret_cast<float *>(io_addrs_[C6NUM]);

  int out_count = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<int>());
  uint64_t state = Pop(chunks, table_log_);
  while ((curr_chunk_index_ >= 0) || (bit_count_table[state] == 0) || (curr_bit_count_ > 0)) {
    if (out_count == 0) {
      return kAicpuKernelStateSucess;
    }
    output[--out_count] = static_cast<float>(centroids[symbol_table[state]]);
    // state = newStateBaseline + rest
    state = states_table[state] + Pop(chunks, bit_count_table[state]);
  }
  return kAicpuKernelStateSucess;
}

uint32_t FSEDecodeKernel::FixedBitHalfDequantTask() {
  uint64_t *chunks = reinterpret_cast<uint64_t *>(io_addrs_[C0NUM]);
  uint16_t *states_table = reinterpret_cast<uint16_t *>(io_addrs_[C1NUM]);
  uint8_t *bit_count_table = reinterpret_cast<uint8_t *>(io_addrs_[C2NUM]);
  uint16_t *symbol_table = reinterpret_cast<uint16_t *>(io_addrs_[C3NUM]);
  float *centroids = reinterpret_cast<float *>(io_addrs_[C4NUM]);
  int *intput_shape = reinterpret_cast<int *>(io_addrs_[C5NUM]);
  Eigen::half *output = reinterpret_cast<Eigen::half *>(io_addrs_[C6NUM]);

  int out_count = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<int>());
  std::vector<int> input_shape_vector(intput_shape, intput_shape + input_shape_size_);
  std::vector<int> index_maps;
  auto ret = NCHW_TO_FRAC_NZ(input_shape_vector, output_shape_, &index_maps);
  if (ret != 0) {
    return kAicpuKernelStateFailed;
  }
  uint64_t state = Pop(chunks, table_log_);
  while ((curr_chunk_index_ >= 0) || (bit_count_table[state] == 0) || (curr_bit_count_ > 0)) {
    if (out_count == 0) {
      return kAicpuKernelStateSucess;
    }
    auto out_index = index_maps.at(--out_count);
    output[out_index] = static_cast<Eigen::half>(centroids[symbol_table[state]]);
    // state = newStateBaseline + rest
    state = states_table[state] + Pop(chunks, bit_count_table[state]);
  }
  return kAicpuKernelStateSucess;
}

uint32_t FSEDecodeKernel::FSEDecodeTask() {
  if (io_addrs_.empty() || io_addrs_.size() != C7NUM) {
    return kAicpuKernelStateFailed;
  }
  if (dst_type_ == mindspore::kNumberTypeFloat32) {
    FixedBitFloatDequantTask();
  } else if (dst_type_ == mindspore::kNumberTypeFloat16) {
    FixedBitHalfDequantTask();
  } else {
    return kAicpuKernelStateInvalid;
  }
  return kAicpuKernelStateSucess;
}

uint32_t FSEDecodeKernel::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  // get value of attr axis
  dst_type_ = attrs["dst_t"].i();
  curr_chunk_ = attrs["curr_chunk"].i();
  curr_chunk_index_ = attrs["curr_chunk_index"].i();
  curr_bit_count_ = attrs["curr_bit_count"].i();
  table_log_ = attrs["table_log"].i();

  // get input tensors shape
  if (node_def_.inputs_size() != C6NUM) {
    AICPU_LOGE("For 'FSEDecode', input tensor number must be 1, but got %d", node_def_.inputs_size());
    return kAicpuKernelStateInvalid;
  }
  input_shape_size_ = node_def_.inputs(C5NUM).tensor_shape().dim(0).size();

  // get output tensor shape
  if (node_def_.outputs_size() != 1) {
    AICPU_LOGE("For 'FSEDecode', output tensor number must be 1, but got %d", node_def_.outputs_size());
    return kAicpuKernelStateInvalid;
  }
  aicpuops::Tensor output_tensor = node_def_.outputs(0);
  output_shape_ = GetShape(output_tensor.tensor_shape());

  return kAicpuKernelStateSucess;
}

uint32_t FSEDecodeKernel::DoCompute() { return FSEDecodeTask(); }
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t FSEDecode(void *param) {
  aicpu::FSEDecodeKernel fse_decode_kernel;
  return fse_decode_kernel.Compute(param);
}
}
