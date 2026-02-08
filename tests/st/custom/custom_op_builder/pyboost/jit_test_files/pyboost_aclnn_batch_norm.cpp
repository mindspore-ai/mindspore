/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include <vector>
#include <utility>
#include "ms_extension/all.h"

namespace custom {

static std::tuple<ShapeVector, ShapeVector, ShapeVector> InferBatchNormShape(const ShapeVector &x_shape) {
  const int64_t rank = static_cast<int64_t>(x_shape.size());
  if (rank < 2) {
    MS_LOG(EXCEPTION) << "BatchNorm input rank must >= 2";
  }
  const int64_t channels = x_shape[1];
  ShapeVector saved_shape{channels};
  return {x_shape, saved_shape, saved_shape};
}

static std::tuple<ms::Tensor, ms::Tensor, ms::Tensor> GenBatchNormResultTensors(const ms::Tensor &x, ms::TypeId dtype) {
  ShapeVector x_shape = x.shape();
  auto [y_shape, sm_shape, sv_shape] = InferBatchNormShape(x_shape);

  ms::Tensor y(dtype, y_shape);
  ms::Tensor saved_mean(ms::TypeId::kNumberTypeFloat32, sm_shape);
  ms::Tensor saved_var(ms::TypeId::kNumberTypeFloat32, sv_shape);
  return {std::move(y), std::move(saved_mean), std::move(saved_var)};
}

std::vector<ms::Tensor> npu_batch_norm(const ms::Tensor &x, const ms::Tensor &scale, const ms::Tensor &bias,
                                       const ms::Tensor &mean, const ms::Tensor &var, bool training, double momentum,
                                       double eps) {
  auto [y, save_mean, save_var] = GenBatchNormResultTensors(x, x.data_type());

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("aclnnBatchNorm");
  runner->SetLaunchFunc(
    LAUNCH_ACLNN_FUNC(aclnnBatchNorm, x, scale, bias, mean, var, training, momentum, eps, y, save_mean, save_var));
  runner->Run({x, scale, bias, mean, var}, {y, save_mean, save_var});
  return {y, save_mean, save_var};
}

}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_batch_norm", PYBOOST_CALLER(3, custom::npu_batch_norm)); }