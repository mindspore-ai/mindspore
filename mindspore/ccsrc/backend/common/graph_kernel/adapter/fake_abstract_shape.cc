/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/adapter/fake_abstract_shape.h"
#include <sstream>
#include "include/common/utils/utils.h"

namespace mindspore::graphkernel {
namespace {
class FakeAbstractShape {
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
      std::stringstream ss;
      ss << "[";
      size_t i = 0;
      for (const auto &it : fmap) {
        if (i > 0) {
          ss << ", ";
        }
        ++i;
        ss << it.first;
      }
      ss << "]";
      MS_LOG(WARNING) << "Encounter unsupported format when infer the abstract shape, because the format " << format
                      << " is not in supported list " << ss.str();
      return device_shape;
    }
    return iter->second(device_shape);
  }

 private:
  static ShapeVector NchwAbstractShape(const ShapeVector &device_shape) { return device_shape; }
  static ShapeVector NhwcAbstractShape(const ShapeVector &device_shape) {
    const size_t nhwc_size = 4;
    const size_t index_n = 0;
    const size_t index_h = 1;
    const size_t index_w = 2;
    const size_t index_c = 3;
    if (device_shape.size() != nhwc_size) {
      MS_LOG(EXCEPTION) << "Shape size of NHWC should be 4, but got " << device_shape.size();
    }
    return {device_shape[index_n], device_shape[index_c], device_shape[index_h], device_shape[index_w]};
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
    size_t batch = dims - nz_size;
    for (size_t i = 0; i < batch; ++i) {
      shape.push_back(device_shape[i]);
    }
    const size_t index_m1 = 3;
    const size_t index_m2 = 2;
    const size_t index_n1 = 4;
    const size_t index_n2 = 1;
    int64_t m = device_shape[dims - index_m1] * device_shape[dims - index_m2];
    int64_t n = device_shape[dims - index_n1] * device_shape[dims - index_n2];
    shape.push_back(m);
    shape.push_back(n);

    return shape;
  }
};
}  // namespace

ShapeVector GetFakeAbstractShape(const ShapeVector &device_shape, const std::string &format) {
  return FakeAbstractShape::GetFakeAbstractShape(device_shape, format);
}
}  // namespace mindspore::graphkernel
