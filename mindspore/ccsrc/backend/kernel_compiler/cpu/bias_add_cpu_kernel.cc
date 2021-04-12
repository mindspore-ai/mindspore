/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/bias_add_cpu_kernel.h"

namespace mindspore {
namespace kernel {
void BiasAddCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  bias_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  data_shape_ = input_shape_.size();
  if (input_shape_.size() < 2) {
    MS_LOG(EXCEPTION) << "Input tensor's rank must be at least 2 for 'BiasAdd' Op, but input tensor's rank is "
                      << input_shape_.size();
  }
  if (bias_shape_.size() != 1) {
    MS_LOG(EXCEPTION) << "Bias's rank must be 1 for 'BiasAdd' Op, but bias' rank is" << bias_shape_.size();
  }
  if (input_shape_[1] != bias_shape_[0]) {
    MS_LOG(EXCEPTION) << "Bias shape not match, bias shape must be equal to C channel's shape";
  }
}

bool BiasAddCPUKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> & /*workspace*/,
                              const std::vector<AddressPtr> &outputs) {
  if (inputs.size() != 2 || outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "inputs outputs size not supoort";
  }

  auto src_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto bias_addr = reinterpret_cast<float *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);

  if (input_shape_.size() > 2) {
    size_t hw_size = 1;
    for (size_t i = 2; i < input_shape_.size(); ++i) {
      hw_size *= input_shape_[i];
    }

    size_t c_size = input_shape_[1];
    for (size_t n = 0; n < input_shape_[0]; ++n) {
      for (size_t c = 0; c < c_size; ++c) {
        size_t offset = n * c_size * hw_size + c * hw_size;
        size_t hw = 0;
#ifdef ENABLE_AVX
        const size_t C8NUM = 8;
        size_t hw8 = hw_size / C8NUM * C8NUM;
        const float *in_ptr = src_addr + offset;
        float *out_ptr = output_addr + offset;
        for (; hw < hw8; hw += C8NUM) {
          __m256 src_r1 = _mm256_loadu_ps(in_ptr);
          __m256 bias_r2 = _mm256_set1_ps(bias_addr[c]);
          __m256 dst_r3 = _mm256_add_ps(src_r1, bias_r2);
          _mm256_storeu_ps(out_ptr, dst_r3);

          in_ptr += C8NUM;
          out_ptr += C8NUM;
        }
#endif
        for (; hw < hw_size; ++hw) {
          output_addr[offset + hw] = src_addr[offset + hw] + bias_addr[c];
        }
      }
    }
  } else {
    size_t n_offset = 0;
    for (size_t n = 0; n < input_shape_[0]; ++n) {
      for (size_t c = 0; c < input_shape_[1]; ++c) {
        output_addr[n_offset + c] = src_addr[n_offset + c] + bias_addr[c];
      }
      n_offset += input_shape_[1];
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
