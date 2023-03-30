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

#include "plugin/device/cpu/kernel/affine_grid_grad_cpu_kernel.h"

#include <algorithm>
#include <string>
#include <map>

#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kAffineGridGradInputsNum = 2;
constexpr size_t kAffineGridGradOutputsNum = 1;

#define AFFINEGRIDGRAD_LAUNCH_CASE(DTYPE, TYPE, DTYPE0, INPUTS, OUTPUTS) \
  case DTYPE: {                                                          \
    if ((DTYPE0) == kNumberTypeInt32) {                                  \
      (void)LaunchKernel<TYPE, int32_t>(INPUTS, OUTPUTS);                \
    } else {                                                             \
      (void)LaunchKernel<TYPE, int64_t>(INPUTS, OUTPUTS);                \
    }                                                                    \
    break;                                                               \
  }

const int kRowNum1 = 1;
const int kColNum1 = 1;
const int kRowNum2 = 2;
const int kColNum2 = 2;
const int kRowNum3 = 3;
const int kColNum3 = 3;
const int kRowNum4 = 4;
const int kColNum4 = 4;
const int kLenXSize3D = 4;
const int kLenXSize4D = 5;
const int kXSizeH3D = 2;
const int kXSizeW3D = 3;
const int kXSizeD4D = 2;
const int kXSizeH4D = 3;
const int kXSizeW4D = 4;
}  // namespace

bool AffineGridGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto prim = base_operator->GetPrim();
  align_corners_ = GetValue<bool>(prim->GetAttr("align_corners"));
  auto type_id = inputs[0]->GetDtype();
  input_info_.push_back(type_id);
  type_id = inputs[1]->GetDtype();
  input_info_.push_back(type_id);
  return true;
}

int AffineGridGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  x_size_dims_ = inputs[1]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

template <typename T, typename T0>
void AffineGridGradCpuKernelMod::LaunchKernel_3D(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  auto x_size_data = reinterpret_cast<T0 *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(x_size_data);

  int64_t H = x_size_data[kXSizeH3D];
  int64_t W = x_size_data[kXSizeW3D];
  Eigen::VectorXf vecX, vecY;
  (void)vecX.setZero(W, 1);
  (void)vecY.setZero(H, 1);
  if (W != 1) {
    vecX = Eigen::VectorXf::LinSpaced(vecX.size(), -1.0, 1.0);
  }
  if (H != 1) {
    vecY = Eigen::VectorXf::LinSpaced(vecY.size(), -1.0, 1.0);
  }
  if (!align_corners_) {
    float x_ = static_cast<float>((W - 1)) / static_cast<float>(W);
    float y_ = static_cast<float>((H - 1)) / static_cast<float>(H);
    for (int64_t i = 0; i < W; i++) {
      vecX[i] = vecX[i] * x_;
    }
    for (int64_t i = 0; i < H; i++) {
      vecY[i] = vecY[i] * y_;
    }
  }

  Eigen::MatrixXf all(kRowNum3, W * H);
  all = make_base_grid_3D<T0>(inputs, vecX, vecY);
  DoCompute_3D<T, T0>(inputs, outputs, all);
}

template <typename T0>
Eigen::MatrixXf AffineGridGradCpuKernelMod::make_base_grid_3D(const std::vector<kernel::AddressPtr> &inputs,
                                                              Eigen::VectorXf vecX, Eigen::VectorXf vecY) {
  auto x_size_data = reinterpret_cast<T0 *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(x_size_data);
  int64_t H = x_size_data[kXSizeH3D];
  int64_t W = x_size_data[kXSizeW3D];
  Eigen::MatrixXf all(kRowNum3, W * H);
  int64_t datanums = H * W;
  auto task1 = [&](const int64_t start, const int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      int64_t j = i % W;
      int64_t k = i / W;
      all(0, k * W + j) = vecX(j);
      all(kRowNum1, k * W + j) = vecY(k);
      all(kRowNum2, k * W + j) = 1.0;
    }
  };
  ParallelLaunchAutoSearch(task1, datanums, this, &parallel_search_info_);
  return all;
}

template <typename T, typename T0>
void AffineGridGradCpuKernelMod::DoCompute_3D(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs, Eigen::MatrixXf all) {
  auto data_y_grad = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(data_y_grad);
  auto x_size_data = reinterpret_cast<T0 *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(x_size_data);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output);
  int64_t N = x_size_data[0];
  int64_t H = x_size_data[kXSizeH3D];
  int64_t W = x_size_data[kXSizeW3D];

  Eigen::MatrixXf y_grad(H * W, kColNum2);
  Eigen::MatrixXf result(kRowNum3, kColNum2);
  float result_0;
  float result_1;
  float result_2;
  int64_t k_num = 0;

  for (int64_t n = 0; n < N; n++) {
    for (int64_t k = 0; k < H * W; k++) {
      y_grad(k, 0) = static_cast<float>(*(data_y_grad + (n * H * W * kColNum2 + k * kColNum2)));
      y_grad(k, kColNum1) = static_cast<float>(*(data_y_grad + (n * H * W * kColNum2 + k * kColNum2) + kColNum1));
    }
    result = all * y_grad;

    for (int64_t k = 0; k < kColNum2; k++) {
      result_0 = result(0, k);
      result_1 = result(kRowNum1, k);
      result_2 = result(kRowNum2, k);
      *(output + k_num) = static_cast<T>(result_0);
      *(output + k_num + kColNum1) = static_cast<T>(result_1);
      *(output + k_num + kColNum2) = static_cast<T>(result_2);
      k_num += kColNum3;
    }
  }
}

template <typename T, typename T0>
void AffineGridGradCpuKernelMod::LaunchKernel_4D(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  auto x_size_data = reinterpret_cast<T0 *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(x_size_data);
  int64_t D = x_size_data[kXSizeD4D];
  int64_t H = x_size_data[kXSizeH4D];
  int64_t W = x_size_data[kXSizeW4D];

  Eigen::VectorXf vecX, vecY, vecZ;
  (void)vecX.setZero(W, 1);
  (void)vecY.setZero(H, 1);
  (void)vecZ.setZero(D, 1);
  if (W != 1) {
    vecX = Eigen::VectorXf::LinSpaced(vecX.size(), -1.0, 1.0);
  }
  if (H != 1) {
    vecY = Eigen::VectorXf::LinSpaced(vecY.size(), -1.0, 1.0);
  }
  if (D != 1) {
    vecZ = Eigen::VectorXf::LinSpaced(vecZ.size(), -1.0, 1.0);
  }
  if (!align_corners_) {
    float x_ = static_cast<float>((W - 1)) / static_cast<float>(W);
    float y_ = static_cast<float>((H - 1)) / static_cast<float>(H);
    float z_ = static_cast<float>((D - 1)) / static_cast<float>(D);
    for (int64_t i = 0; i < W; i++) {
      vecX[i] = vecX[i] * x_;
    }
    for (int64_t i = 0; i < H; i++) {
      vecY[i] = vecY[i] * y_;
    }
    for (int64_t i = 0; i < D; i++) {
      vecZ[i] = vecZ[i] * z_;
    }
  }

  Eigen::MatrixXf all(kRowNum4, D * W * H);
  all = make_base_grid_4D<T0>(inputs, vecX, vecY, vecZ);
  DoCompute_4D<T, T0>(inputs, outputs, all);
}

template <typename T0>
Eigen::MatrixXf AffineGridGradCpuKernelMod::make_base_grid_4D(const std::vector<kernel::AddressPtr> &inputs,
                                                              Eigen::VectorXf vecX, Eigen::VectorXf vecY,
                                                              Eigen::VectorXf vecZ) {
  auto x_size_data = reinterpret_cast<T0 *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(x_size_data);
  int64_t D = x_size_data[kXSizeD4D];
  int64_t H = x_size_data[kXSizeH4D];
  int64_t W = x_size_data[kXSizeW4D];
  Eigen::MatrixXf all(kRowNum4, D * W * H);
  int64_t datanums = D * H * W;
  auto task1 = [&](const int64_t start, const int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      int64_t m = i / (H * W);
      int64_t j = (i % (H * W)) / W;
      int64_t k = (i % (H * W)) % W;
      all(0, m * H * W + j * W + k) = vecX(k);
      all(kRowNum1, m * H * W + j * W + k) = vecY(j);
      all(kRowNum2, m * H * W + j * W + k) = vecZ(m);
      all(kRowNum3, m * H * W + j * W + k) = 1.0;
    }
  };
  ParallelLaunchAutoSearch(task1, datanums, this, &parallel_search_info_);
  return all;
}

template <typename T, typename T0>
void AffineGridGradCpuKernelMod::DoCompute_4D(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs, Eigen::MatrixXf all) {
  auto data_y_grad = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(data_y_grad);
  auto x_size_data = reinterpret_cast<T0 *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(x_size_data);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output);
  int64_t N = x_size_data[0];
  int64_t D = x_size_data[kXSizeD4D];
  int64_t H = x_size_data[kXSizeH4D];
  int64_t W = x_size_data[kXSizeW4D];

  Eigen::MatrixXf y_grad(D * H * W, kColNum3);
  Eigen::MatrixXf result(kRowNum4, kColNum3);
  float result_0;
  float result_1;
  float result_2;
  float result_3;
  int64_t k_num = 0;

  for (int64_t n = 0; n < N; n++) {
    for (int64_t k = 0; k < D * H * W; k++) {
      y_grad(k, 0) = static_cast<float>(*(data_y_grad + (n * D * H * W * kColNum3 + k * kColNum3) + 0));
      y_grad(k, kColNum1) = static_cast<float>(*(data_y_grad + (n * D * H * W * kColNum3 + k * kColNum3) + kColNum1));
      y_grad(k, kColNum2) = static_cast<float>(*(data_y_grad + (n * D * H * W * kColNum3 + k * kColNum3) + kColNum2));
    }
    result = all * y_grad;
    for (int64_t k = 0; k < kColNum3; k++) {
      result_0 = result(0, k);
      result_1 = result(kRowNum1, k);
      result_2 = result(kRowNum2, k);
      result_3 = result(kRowNum3, k);
      *(output + k_num) = static_cast<T>(result_0);
      *(output + k_num + kColNum1) = static_cast<T>(result_1);
      *(output + k_num + kColNum2) = static_cast<T>(result_2);
      *(output + k_num + kColNum3) = static_cast<T>(result_3);
      k_num += kColNum4;
    }
  }
}

template <typename T, typename T0>
bool AffineGridGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  if (x_size_dims_[0] == kLenXSize3D) {
    LaunchKernel_3D<T, T0>(inputs, outputs);
  } else if (x_size_dims_[0] == kLenXSize4D) {
    LaunchKernel_4D<T, T0>(inputs, outputs);
  }
  return true;
}

bool AffineGridGradCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &outputs) {
  constexpr int INPUTSNUM = 1;
  TypeId input_type = input_info_[0];
  TypeId x_size_type = input_info_[INPUTSNUM];
  switch (input_type) {
    AFFINEGRIDGRAD_LAUNCH_CASE(kNumberTypeFloat16, float16, x_size_type, inputs, outputs)
    AFFINEGRIDGRAD_LAUNCH_CASE(kNumberTypeFloat32, float, x_size_type, inputs, outputs)
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', unsupported input data type: " << TypeIdLabel(input_type)
                        << ".";
  }
  return true;
}

using AffineGridGradPair = std::pair<KernelAttr, AffineGridGradCpuKernelMod::KernelRunFunc>;
const std::vector<AffineGridGradPair> &AffineGridGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, AffineGridGradCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddSkipCheckAttr(true), &AffineGridGradCpuKernelMod::Launch},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AffineGridGrad, AffineGridGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
