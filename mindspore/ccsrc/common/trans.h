/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_COMMON_TRANS_H
#define MINDSPORE_CCSRC_COMMON_TRANS_H

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "ir/dtype.h"
#include "kernel/kernel.h"
#include "ir/dtype/type.h"

namespace mindspore {
namespace trans {
struct TypeIdArgs {
  const void *data;
  size_t host_shape_size;  // Multiply each dimension elements. [a, b, c, d] => a*b*c*d
  TypeId host_data_type;
  TypeId device_data_type;
};

struct FormatArgs {
  const void *data;
  const size_t device_size;
  std::string host_format;
  std::string device_format;
  std::vector<size_t> host_shape;
  std::vector<size_t> device_shape;
  TypeId src_data_type;
};

size_t TypeIdSize(const TypeId data_type);
size_t ShapeSize(const std::vector<size_t> &shape);
size_t CubeSizeByType(const TypeId data_type);

std::vector<size_t> PaddingShapeTo4d(const std::vector<size_t> &shape,
                                     const std::vector<kernel::Axis> &padding_axis = {});
std::vector<int> GetRuntimePaddingShape(const AnfNodePtr &node, size_t index);
bool IsNeedPadding(const std::string &format, const size_t shape_size);
std::vector<size_t> TransShapeToDevice(const std::vector<size_t> &shape, const std::string &format);
bool TransDataType(const TypeIdArgs &args, void *result);
bool TransFormat(const FormatArgs &args, void *result);
bool TransFormatFromDeviceToHost(const FormatArgs &args, void *result);

// host to device
bool NchwToFracZ(const FormatArgs &args, void *result);
bool NchwToFracNz(const FormatArgs &args, void *result);
bool NchwToNc1hwc0(const FormatArgs &args, void *result);
// device to host
bool FracZToNchw(const FormatArgs &args, void *result);
bool FracNzToNchw(const FormatArgs &args, void *result);
bool Nc1hwc0ToNchw(const FormatArgs &args, void *result);
}  // namespace trans
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_COMMON_TRANS_H
