/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/apply_proximal_adagrad_cpu_kernel.h"
#include <algorithm>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/cpu/kernel/nnacl/intrinsics/ms_simd_instructions.h"

namespace mindspore {
namespace kernel {
constexpr size_t kApplyProximalAdagradInputsNum = 6;
constexpr size_t kVarIndex = 0;
constexpr size_t kAccIndex = 1;
constexpr size_t kLRIndex = 2;
constexpr size_t kL1Index = 3;
constexpr size_t kL2Index = 4;
constexpr size_t kGradIndex = 5;

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define ApplyProximalAdagradCalc(block_size, block_num, input_elements, var, accum, lr, l1, l2, grad)               \
  do {                                                                                                              \
    MS_FLOAT_32xN(block_num) lr_vec = MS_MOVN_F32(block_size, lr);                                                  \
    MS_FLOAT_32xN(block_num) l1_vec = MS_MOVN_F32(block_size, l1);                                                  \
    MS_FLOAT_32xN(block_num) l2_vec = MS_MOVN_F32(block_size, l2);                                                  \
    for (size_t block_max_size = input_elements - block_num + 1; i < block_max_size; i += block_num) {              \
      MS_FLOAT_32xN(block_num) tmp_vec1 = MS_LD_F32(block_size, grad + i);                                          \
      MS_FLOAT_32xN(block_num) accum_vec = MS_LD_F32(block_size, accum + i);                                        \
      MS_FLOAT_32xN(block_num) prox_v_vec = MS_LD_F32(block_size, var + i);                                         \
      MS_FMADD_F32(block_size, accum_vec, tmp_vec1, tmp_vec1);                                                      \
      MS_FLOAT_32xN(block_num) learn_rate_vec = MS_DIV_F32(block_size, lr_vec, MS_SQRT_F32(block_size, accum_vec)); \
      MS_FMSUB_F32(block_size, prox_v_vec, tmp_vec1, learn_rate_vec);                                               \
      MS_ST_F32(block_size, accum + i, accum_vec);                                                                  \
      tmp_vec1 = MS_MOVN_F32(block_size, 1);                                                                        \
      MS_FMSUB_F32(block_size, tmp_vec1, l2_vec, learn_rate_vec);                                                   \
      if (l1 > 0) {                                                                                                 \
        learn_rate_vec = MS_MUL_F32(block_size, learn_rate_vec, l1_vec);                                            \
        tmp_vec1 = MS_DIV_F32(block_size, MS_SIN_F32(block_size, prox_v_vec), tmp_vec1);                            \
        learn_rate_vec = MS_MUL_F32(block_size, MS_ABS_F32(block_size, prox_v_vec), learn_rate_vec);                \
        learn_rate_vec = MS_MAX_F32(block_size, learn_rate_vec, MS_MOVN_F32(block_size, 0.0f));                     \
        prox_v_vec = MS_MUL_F32(block_size, learn_rate_vec, tmp_vec1);                                              \
      } else {                                                                                                      \
        prox_v_vec = MS_DIV_F32(block_size, prox_v_vec, tmp_vec1);                                                  \
      }                                                                                                             \
      MS_ST_F32(block_size, var + i, prox_v_vec);                                                                   \
    }                                                                                                               \
  } while (0)

bool ApplyProximalAdagradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();

  auto input_type_id = inputs[0]->GetDtype();
  if (input_type_id != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "',  does not support " << TypeIdToString(input_type_id);
    return false;
  }
  unit_size_ = sizeof(float);

  return true;
}

int ApplyProximalAdagradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  if (input_size_list_.size() != kApplyProximalAdagradInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 6.";
    return KRET_RESIZE_FAILED;
  }
  std::vector<int64_t> var_shape = inputs[kVarIndex]->GetShapeVector();
  std::vector<int64_t> accum_shape = inputs[kAccIndex]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kLRIndex]->GetShapeVector();
  std::vector<int64_t> l1_shape = inputs[kL1Index]->GetShapeVector();
  std::vector<int64_t> l2_shape = inputs[kL2Index]->GetShapeVector();
  std::vector<int64_t> grad_shape = inputs[kGradIndex]->GetShapeVector();
  if (var_shape.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'var' must be at least 1-D, but got scalar or None.";
  }
  if (!IsSameShape(var_shape, accum_shape)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape of 'accum' must be the same as the shape of 'var', "
                         "but got the shape of 'accum': "
                      << Vector2Str(accum_shape) << " and the shape of 'var': " << Vector2Str(var_shape);
  }
  if (!IsSameShape(var_shape, grad_shape)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape of 'grad' must be the same as the shape of 'var', "
                         "but got the shape of 'grad': "
                      << Vector2Str(grad_shape) << " and the shape of 'var': " << Vector2Str(var_shape);
  }

  if (!lr_shape.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', 'lr' must be a scalar,and dimension of 'lr' must be 0,but got the dimension of 'lr': "
                      << Vector2Str(lr_shape);
  }
  if (!l1_shape.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', 'l1' must be a scalar,and dimension of 'l1' must be 0,but got the dimension of 'l1': "
                      << Vector2Str(l1_shape);
  }
  if (!l2_shape.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', 'l2' must be a scalar,and dimension of 'l2' must be 0,but got the dimension of 'l2': "
                      << Vector2Str(l2_shape);
  }

  input_elements_ = input_size_list_[0] / unit_size_;
  return ret;
}

bool ApplyProximalAdagradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &workspace,
                                              const std::vector<kernel::AddressPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyProximalAdagradInputsNum, kernel_name_);
  auto var = reinterpret_cast<float *>(inputs[kVarIndex]->addr);
  auto accum = reinterpret_cast<float *>(inputs[kAccIndex]->addr);
  auto lr = reinterpret_cast<float *>(inputs[kLRIndex]->addr)[0];
  auto l1 = reinterpret_cast<float *>(inputs[kL1Index]->addr)[0];
  auto l2 = reinterpret_cast<float *>(inputs[kL2Index]->addr)[0];
  auto grad = reinterpret_cast<float *>(inputs[kGradIndex]->addr);
  size_t i = 0;
  MS_SIMD_RUN_NO_SCALAR(ApplyProximalAdagradCalc, input_elements_, var, accum, lr, l1, l2, grad);

  for (; i < input_elements_; ++i) {
    accum[i] += grad[i] * grad[i];
    auto learning_rate = lr / std::sqrt(accum[i]);
    auto prox_v = var[i];
    prox_v -= grad[i] * learning_rate;

    if (l1 > 0) {
      var[i] = sinf(prox_v) * std::fmax(std::fabs(prox_v) - learning_rate * l1, 0.0) / (1 + l2 * learning_rate);
    } else {
      var[i] = prox_v / (1 + l2 * learning_rate);
    }
  }

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyProximalAdagrad, ApplyProximalAdagradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
