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
#include "common/graph_kernel/adapter/fake_abstract_shape.h"

#include "utils/utils.h"

namespace mindspore::graphkernel {
namespace {
class AbstractShapeCreator {
 public:
  using AbstractShapeTransferFunc = std::function<ShapeVector(const ShapeVector &)>;
  /**
   * Get an abstract shape.
   * For a given device_shape and format, the available abstract_shape is not unique,
   * this interface only returns a legal abstract_shape without considering padding
   * so that the AnfAlgo's get device shape interface can get the right device_shape.
   */
  static ShapeVector GetFakeAbstractShape(const ShapeVector &device_shape, const std::string &format) {
    const std::map<std::string, AbstractShapeTransferFunc> fmap{
      {kOpFormat_NCHW, NchwAbstractShape},
      {kOpFormat_NHWC, NhwcAbstractShape},
      {kOpFormat_FRAC_NZ, FractalNzAbstractShape},
    };
    if (format == kOpFormat_ND || format == kOpFormat_DEFAULT) {
      return device_shape;
    }
    auto iter = fmap.find(format);
    if (iter == fmap.end()) {
      MS_LOG(WARNING) << "Unexpected format[" << format << "]";
      return device_shape;
    }
    return iter->second(device_shape);
  }

 private:
  static ShapeVector NchwAbstractShape(const ShapeVector &device_shape) { return device_shape; }
  static ShapeVector NhwcAbstractShape(const ShapeVector &device_shape) {
    const size_t nhwc_size = 4;
    if (device_shape.size() != nhwc_size) {
      MS_LOG(EXCEPTION) << "Shape size of NHWC should be 4, but got " << device_shape.size();
    }
    return {device_shape[0], device_shape[3], device_shape[1], device_shape[2]};
  }
  static ShapeVector FractalNzAbstractShape(const ShapeVector &device_shape) {
    if (device_shape.size() == 1 && (device_shape[0] == 1 || static_cast<size_t>(device_shape[0]) % kCubeSize == 0)) {
      return device_shape;
    }
    const size_t nz_size = 4;
    if (device_shape.size() < nz_size) {
      MS_LOG(EXCEPTION) << "Shape size of FRACTAL_NZ should >= 4, but got " << device_shape.size();
    }
    ShapeVector shape;
    size_t dims = device_shape.size();
    size_t batch = dims - 4;
    for (size_t i = 0; i < batch; ++i) {
      shape.push_back(device_shape[i]);
    }
    int64_t m = device_shape[dims - 3] * device_shape[dims - 2];
    int64_t n = device_shape[dims - 4] * device_shape[dims - 1];
    shape.push_back(m);
    shape.push_back(n);

    return shape;
  }
};
}  // namespace

ShapeVector GetFakeAbstractShape(const ShapeVector &device_shape, const std::string &format) {
  return AbstractShapeCreator::GetFakeAbstractShape(device_shape, format);
}
}  // namespace mindspore::graphkernel
