/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include "ms_extension/all.h"

namespace custom {

static int64_t AvgPoolOutSize(int64_t in_size, int64_t kernel, int64_t stride, int64_t padding, bool ceil_mode) {
  int64_t out_size = (in_size + 2 * padding - kernel) / stride + 1;
  if (ceil_mode && (out_size - 1) * stride >= in_size + padding) {
    --out_size;
  }
  return out_size > 0 ? out_size : 0;
}

static ShapeVector InferAvgPool2dShape(const ShapeVector &in_shape, const std::vector<int64_t> &kernel_size,
                                       const std::vector<int64_t> &stride, const std::vector<int64_t> &padding,
                                       bool ceil_mode) {
  if (in_shape.size() < 2) {
    MS_LOG(EXCEPTION) << "AvgPool2d input rank must be at least 2";
  }
  ShapeVector out_shape = in_shape;
  auto rank = in_shape.size();
  out_shape[rank - 2] = AvgPoolOutSize(in_shape[rank - 2], kernel_size[0], stride[0], padding[0], ceil_mode);
  out_shape[rank - 1] = AvgPoolOutSize(in_shape[rank - 1], kernel_size[1], stride[1], padding[1], ceil_mode);
  return out_shape;
}

ms::Tensor npu_avgpool2d_v2(const ms::Tensor &x, const std::vector<int64_t> &kernel_size,
                            const std::vector<int64_t> &stride, const std::vector<int64_t> &padding, bool ceil_mode,
                            bool count_include_pad, int64_t divisor_override, bool cube_math_type) {
  auto result =
    ms::Tensor(x.data_type(), InferAvgPool2dShape(x.shape(), kernel_size, stride, padding, ceil_mode));
  ms::TensorToDevice(x);
  ms::TensorAllocate(result);
  MS_DISPATCH_ACLNN(aclnnAvgPool2d, x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override,
                    cube_math_type, result);
  return result;
}

}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_avgpool2d_v2", PYBOOST_CALLER(1, custom::npu_avgpool2d_v2)); }
