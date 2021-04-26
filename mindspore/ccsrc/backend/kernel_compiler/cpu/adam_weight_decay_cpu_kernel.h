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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAM_WEIGHT_DECAY_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAM_WEIGHT_DECAY_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

#if defined(ENABLE_AVX512)
#include <x86intrin.h>
#endif

#ifdef ENABLE_NEON
#define MS_FLOAT32X4 float32x4_t
#define MS_LDQ_F32 vld1q_f32
#define MS_MOVQ_F32 vmovq_n_f32
#define MS_STQ_F32 vst1q_f32
#define MS_ADDQ_F32(src1, src2) vaddq_f32(src1, src2)
#define MS_MULQ_F32(src1, src2) vmulq_f32(src1, src2)
#define MS_MULQ_N_F32(src1, src2) vmulq_n_f32(src1, src2)
#define MS_DIVQ_F32(src1, src2) vdivq_f32(src1, src2)
#define MS_MLAQ_F32(src1, src2, src3) vmlaq_f32(src1, src2, src3)
#define MS_MLAQ_N_F32(src1, src2, src3) vmlaq_n_f32(src1, src2, src3)
#define MS_SQRT_F32(src) vsqrtq_f32(src)
#define MS_CAST_F32_F16(src) vreinterpretq_f32_f16(src)
#define MS_NEON_WIDTH 4
#endif

#if defined(ENABLE_AVX512)
#define MS_FLOAT32X16 __m512
#define MS_LD512_F32 _mm512_loadu_ps
#define MS_ST512_F32 _mm512_storeu_ps
#define MS_MOV512_F32 _mm512_set1_ps
#define MS_ADD512_F32(src1, src2) _mm512_add_ps(src1, src2)
#define MS_MUL512_F32(src1, src2) _mm512_mul_ps(src1, src2)
#define MS_DIV512_F32(src1, src2) _mm512_div_ps(src1, src2)
#define MS_FMA512_F32(src1, src2, src3) _mm512_fmadd_ps(src1, src2, src3)
#define MS_SQRT512_F32(src) _mm512_sqrt_ps(src)
#define MS_CAST512_F32_S32(src) _mm512_castsi512_ps(src)
#define MS_AVX512_WIDTH 16
#endif

namespace mindspore {
namespace kernel {
class AdamWeightDecayCPUKernel : public CPUKernel {
 public:
  AdamWeightDecayCPUKernel() = default;
  ~AdamWeightDecayCPUKernel() override = default;
  template <typename T>
  void LaunchAdamWeightDecay(T *var, T *m, T *v, float lr, float beta1, float beta2, float epsilon, T *decay,
                             const T *gradient, size_t size);
  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
};

MS_REG_CPU_KERNEL(AdamWeightDecay,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  AdamWeightDecayCPUKernel)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAM_WEIGHT_DECAY_CPU_KERNEL_H_
