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

#include <set>
#include <optional>
#include "mindspore/include/custom_op_api.h"

namespace custom {
using namespace mindspore;
constexpr size_t kQuantBatchMatmulOutputNum = 1;
constexpr size_t kQbmmMatSize = 2;

ms::Tensor GetTensorOrEmpty(const std::optional<ms::Tensor> &opt_tensor) {
  return opt_tensor.has_value() ? opt_tensor.value() : ms::Tensor();
}

ShapeVector BatchMatMulMakeShape(const ShapeVector x1_shape, const ShapeVector x2_shape, bool transpose_x1,
                                 bool transpose_x2, size_t offset) {
  ShapeVector out_shape;
  ShapeVector long_shape = x1_shape.size() > x2_shape.size() ? x1_shape : x2_shape;
  ShapeVector short_shape = x1_shape.size() > x2_shape.size() ? x2_shape : x1_shape;
  size_t size_diff = long_shape.size() - short_shape.size();
  for (size_t i = 0; i < long_shape.size() - offset; i++) {
    if (long_shape[i] < 0) {
      out_shape.push_back(abstract::Shape::kShapeDimAny);
    } else if (i >= size_diff) {
      out_shape.push_back(long_shape[i] > short_shape[i - size_diff] ? long_shape[i] : short_shape[i - size_diff]);
    } else {
      out_shape.push_back(long_shape[i]);
    }
  }
  size_t x1_offset = x1_shape.size() - offset;
  size_t x2_offset = x2_shape.size() - offset;
  out_shape.push_back(x1_shape[x1_offset + (transpose_x1 ? 1 : 0)]);
  out_shape.push_back(x2_shape[x2_offset + (transpose_x2 ? 0 : 1)]);
  return out_shape;
}

void set_nz_storage(const ms::Tensor &tensor, const std::string &nz_format) {
  tensor.set_format(nz_format);
  auto nd_shape = tensor.shape();
  auto nz_shape =
    mindspore::trans::DeviceShapeTransfer().GetDeviceShapeByFormat(nd_shape, nz_format, tensor.data_type());

  constexpr int64_t kStrideBase = 1;
  constexpr int kStrideOffset = 2;
  auto strides = nd_shape;
  if (!strides.empty()) {
    strides.erase(strides.begin());
  }
  strides.push_back(kStrideBase);
  for (int i = static_cast<int>(strides.size()) - kStrideOffset; i >= 0; i--) {
    strides[i] = strides[i] * strides[i + 1];
  }
  auto storage_info = std::make_shared<TensorStorageInfo>(nd_shape, strides, nz_shape, strides, true);
  MS_EXCEPTION_IF_NULL(tensor.tensor());
  MS_EXCEPTION_IF_NULL(tensor.tensor()->device_address());
  tensor.tensor()->set_storage_info(storage_info);
}

ms::Tensor quant_batch_matmul_custom(const ms::Tensor &x1, const ms::Tensor &x2, const ms::Tensor &scale,
                                     const std::optional<ms::Tensor> &offset, const std::optional<ms::Tensor> &bias,
                                     const std::optional<ms::Tensor> &pertoken_scale, bool transpose_x1,
                                     bool transpose_x2, const std::string x2_format, const int64_t output_dtype) {
  auto x1_shape = x1.shape();
  auto x2_shape = x2.shape();
  auto output_shape = BatchMatMulMakeShape(x1.shape(), x2.shape(), transpose_x1, transpose_x2, kQbmmMatSize);
  TypeId out_dtype = static_cast<TypeId>(output_dtype);
  auto out = ms::Tensor(out_dtype, output_shape);

  const auto nz_format = "FRACTAL_NZ";
  if (x2_format == nz_format) {
    set_nz_storage(x2, nz_format);
  }
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("QuantMatmulV4");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnQuantMatmulV4, x1, x2, scale, offset, pertoken_scale, bias, transpose_x1,
                                          transpose_x2, out));
  runner->Run({x1, x2, scale, GetTensorOrEmpty(offset), GetTensorOrEmpty(pertoken_scale), GetTensorOrEmpty(bias)},
              {out});
  return out;
}
}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("quant_batch_matmul", PYBOOST_CALLER(custom::kQuantBatchMatmulOutputNum, custom::quant_batch_matmul_custom));
}
