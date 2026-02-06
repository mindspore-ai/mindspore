/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_GAMMA_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_GAMMA_CPU_KERNEL_H_
#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL

#include <vector>
#include <string>
#include <map>
#include "kernel/cpu/cpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "mindspore/ops/kernel/cpu/utils/random_util.h"
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"
#include "mindspore/ops/infer/random_gamma.h"

namespace mindspore {
namespace kernel {
class GammaCpuKernelMod : public NativeCpuKernelMod {
 public:
  GammaCpuKernelMod() = default;
  ~GammaCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using Normal = random::MSNormalDistribution<random::PhiloxRandom, double>;
  using Uniform = random::MSUniformDistribution<random::PhiloxRandom, double>;

  template <typename T>
  void Generate(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  template <typename T>
  void InferShape(const std::vector<KernelTensor *> &inputs);

  template <typename T>
  void GenerateSamplesForRange(int64_t start_output, int64_t limit_output, int64_t samples_per_alpha,
                               int64_t num_alphas, const random::PhiloxRandom &rng, T *samples_flat,
                               const T *alpha_flat);

  template <typename T>
  void GenerateExponentialSamples(int64_t *output_idx, int64_t limit_output, int64_t samples_per_alpha,
                                  int64_t num_alphas, const random::PhiloxRandom &rng, T *samples_alpha_offset,
                                  Uniform *uniform, typename Uniform::ResType *uniform_res);

  template <typename T>
  void GenerateGammaSamples(int64_t *output_idx, int64_t limit_output, int64_t samples_per_alpha, int64_t num_alphas,
                            const random::PhiloxRandom &rng, T *samples_alpha_offset, double alpha_value,
                            Normal *normal, Uniform *uniform, typename Normal::ResType *norm_res,
                            typename Uniform::ResType *uniform_res);

  double GenerateSingleGammaSample(random::PhiloxRandom *gen, double alpha_value, bool alpha_less_than_one, double su,
                                   double cut, Normal *normal, Uniform *uniform, typename Normal::ResType *norm_res,
                                   typename Uniform::ResType *uniform_res);

  double GetNextUniformRandom(Uniform *uniform, random::PhiloxRandom *gen, typename Uniform::ResType *uniform_res,
                              int64_t *uniform_remaining);

  int64_t seed_{0};
  int64_t seed2_{0};

  ShapeVector output_shape_;
  ShapeVector shape_shape_;
  ShapeVector alpha_shape_;
  TypeId shape_dtype_{kTypeUnknown};
  TypeId alpha_dtype_{kTypeUnknown};

  random::GuardedPhiloxRandom rng_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_GAMMA_CPU_KERNEL_H_
