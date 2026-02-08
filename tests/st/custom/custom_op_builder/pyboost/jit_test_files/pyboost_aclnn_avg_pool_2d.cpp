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
#include "ms_extension/all.h"

namespace custom {

std::vector<int64_t> InferAvgPool2dShape(const std::vector<int64_t> &in_shape, const std::vector<int64_t> &kernel_size,
                                         const std::vector<int64_t> &stride, const std::vector<int64_t> &padding,
                                         bool ceil_mode) {
  if (in_shape.size() < 2) {
    MS_LOG(EXCEPTION) << "AvgPool2d input rank must >= 2";
  }

  const int64_t rank = static_cast<int64_t>(in_shape.size());
  std::vector<int64_t> out_shape = in_shape;  // 先复制

  int64_t H = in_shape[rank - 2];
  int64_t W = in_shape[rank - 1];

  auto compute_out = [=](int64_t in, int64_t k, int64_t s, int64_t p) -> int64_t {
    int64_t tmp = in + 2 * p - (ceil_mode ? k - 1 : k);
    int64_t o = tmp / s + 1;
    if (ceil_mode && tmp % s != 0) ++o;
    return o > 0 ? o : 0;
  };

  out_shape[rank - 2] = compute_out(H, kernel_size[0], stride[0], padding[0]);
  out_shape[rank - 1] = compute_out(W, kernel_size[1], stride[1], padding[1]);
  return out_shape;
}

ms::Tensor GenAvgPool2dResultTensor(const ms::Tensor &x, const std::vector<int64_t> &kernel_size,
                                    const std::vector<int64_t> &stride, const std::vector<int64_t> &padding,
                                    bool ceil_mode) {
  auto out_shape = InferAvgPool2dShape(x.shape(), kernel_size, stride, padding, ceil_mode);
  return ms::Tensor(x.data_type(), out_shape);
}

ms::Tensor npu_avgpool2d(const ms::Tensor &x, const std::vector<int64_t> &kernel_size,
                         const std::vector<int64_t> &stride, const std::vector<int64_t> &padding, bool ceil_mode,
                         bool count_include_pad, int64_t divisor_override, bool cube_math_type) {
  auto y = GenAvgPool2dResultTensor(x, kernel_size, stride, padding, ceil_mode);

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("aclnnAvgPool2d");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAvgPool2d, x, kernel_size, stride, padding, ceil_mode, count_include_pad,
                                          divisor_override, cube_math_type, y));
  runner->Run({x}, {y});
  return y;
}

}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_avgpool2d", PYBOOST_CALLER(1, custom::npu_avgpool2d)); }