/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/adam_weight_decay_cpu_kernel.h"

#include <cmath>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
template <typename T>
void AdamWeightDecayCPUKernel::LaunchAdamWeightDecay(T *var, T *m, T *v, float lr, float beta1, float beta2,
                                                     float epsilon, T *decay, const T *gradient, size_t size) {
  float beta1_minus = 1 - beta1;
  float beta2_minus = 1 - beta2;
#if defined(ENABLE_AVX512)
  MS_FLOAT32X16 beta1_16 = MS_MOV512_F32(beta1);
  MS_FLOAT32X16 beta2_16 = MS_MOV512_F32(beta2);
  MS_FLOAT32X16 beta1_minus_16 = MS_MOV512_F32(beta1_minus);
  MS_FLOAT32X16 beta2_minus_16 = MS_MOV512_F32(beta2_minus);
  MS_FLOAT32X16 lr_neg_16 = MS_MOV512_F32(-lr);
  MS_FLOAT32X16 epsilon_16 = MS_MOV512_F32(epsilon);
  MS_FLOAT32X16 decay_16 = MS_MOV512_F32(*decay);
#endif
#if defined(ENABLE_NEON)
  MS_FLOAT32X4 epsilon_4 = MS_MOVQ_F32(epsilon);
  float lr_neg = -lr;
#endif

  auto task = [&](size_t start, size_t end) {
    size_t i = start;
#if defined(ENABLE_AVX512)
    if (end >= MS_AVX512_WIDTH) {
      for (; i <= end - MS_AVX512_WIDTH; i += MS_AVX512_WIDTH) {
        MS_FLOAT32X16 var_16 = MS_LD512_F32(var + i);
        MS_FLOAT32X16 m_16 = MS_LD512_F32(m + i);
        MS_FLOAT32X16 v_16 = MS_LD512_F32(v + i);
        MS_FLOAT32X16 g_16 = MS_LD512_F32(gradient + i);
        m_16 = MS_MUL512_F32(m_16, beta1_16);
        m_16 = MS_FMA512_F32(g_16, beta1_minus_16, m_16);
        v_16 = MS_MUL512_F32(v_16, beta2_16);
        v_16 = MS_MUL512_F32(g_16, g_16);
        v_16 = MS_FMA512_F32(g_16, beta2_minus_16, v_16);
        g_16 = MS_SQRT512_F32(v_16);
        g_16 = MS_DIV512_F32(m_16, MS_ADD512_F32(g_16, epsilon_16));
        g_16 = MS_FMA512_F32(var_16, decay_16, g_16);
        var_16 = MS_FMA512_F32(g_16, lr_neg_16, var_16);
        MS_ST512_F32(var + i, var_16);
        MS_ST512_F32(m + i, m_16);
        MS_ST512_F32(v + i, v_16);
      }
    }
#endif
#if defined(ENABLE_NEON)
    if (end >= MS_NEON_WIDTH) {
      for (; i <= end - MS_NEON_WIDTH; i += MS_NEON_WIDTH) {
        MS_FLOAT32X4 var_4 = MS_LDQ_F32(var + i);
        MS_FLOAT32X4 m_4 = MS_LDQ_F32(m + i);
        MS_FLOAT32X4 v_4 = MS_LDQ_F32(v + i);
        MS_FLOAT32X4 g_4 = MS_LDQ_F32(gradient + i);
        m_4 = MS_MULQ_N_F32(m_4, beta1);
        m_4 = MS_MLAQ_N_F32(m_4, g_4, beta1_minus);
        v_4 = MS_MULQ_N_F32(v_4, beta2);
        g_4 = MS_MULQ_F32(g_4, g_4);
        v_4 = MS_MLAQ_N_F32(v_4, g_4, beta2_minus);
        g_4 = MS_SQRT_F32(v_4);
        g_4 = MS_DIVQ_F32(m_4, MS_ADDQ_F32(g_4, epsilon_4));
        g_4 = MS_MLAQ_N_F32(g_4, var_4, *decay);
        var_4 = MS_MLAQ_N_F32(var_4, g_4, lr_neg);
        MS_STQ_F32(var + i, var_4);
        MS_STQ_F32(m + i, m_4);
        MS_STQ_F32(v + i, v_4);
      }
    }
#endif
    for (; i < end; i++) {
      m[i] += (gradient[i] - m[i]) * beta1_minus;
      v[i] += (gradient[i] * gradient[i] - v[i]) * beta2_minus;
      T update = m[i] / (std::sqrt(v[i]) + epsilon);
      update += decay[0] * var[i];
      var[i] -= lr * update;
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

void AdamWeightDecayCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 9) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but AdamWeightDecay needs 9 inputs.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 3) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but AdamWeightDecay needs 3 outputs.";
  }
}

bool AdamWeightDecayCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> & /*workspace*/,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != 9) {
    MS_LOG(EXCEPTION) << "Input number is " << inputs.size() << ", but AdamWeightDecay needs 9 inputs.";
  }
  if (outputs.size() != 3) {
    MS_LOG(EXCEPTION) << "Output number is " << outputs.size() << ", but AdamWeightDecay needs 3 outputs.";
  }
  if (inputs[0]->size != inputs[1]->size || inputs[0]->size != inputs[2]->size || inputs[0]->size != inputs[8]->size) {
    MS_LOG(EXCEPTION) << "Error input data size!";
  }
  size_t f_size = sizeof(float);
  if (inputs[3]->size != f_size || inputs[4]->size != f_size || inputs[5]->size != f_size ||
      inputs[6]->size != f_size || inputs[7]->size != f_size) {
    MS_LOG(EXCEPTION) << "The attribute beta, lr and epsilon must be float!";
  }
  auto var = reinterpret_cast<float *>(inputs[0]->addr);
  auto m = reinterpret_cast<float *>(inputs[1]->addr);
  auto v = reinterpret_cast<float *>(inputs[2]->addr);
  float lr = reinterpret_cast<float *>(inputs[3]->addr)[0];
  float beta1 = reinterpret_cast<float *>(inputs[4]->addr)[0];
  float beta2 = reinterpret_cast<float *>(inputs[5]->addr)[0];
  float epsilon = reinterpret_cast<float *>(inputs[6]->addr)[0];
  auto decay = reinterpret_cast<float *>(inputs[7]->addr);
  auto gradient = reinterpret_cast<float *>(inputs[8]->addr);

  // multithreading
  size_t lens = inputs[0]->size > 0 ? static_cast<size_t>(inputs[0]->size / sizeof(float)) : 1;
  LaunchAdamWeightDecay<float>(var, m, v, lr, beta1, beta2, epsilon, decay, gradient, lens);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
