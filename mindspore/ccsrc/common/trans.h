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
#include "backend/kernel_compiler/kernel.h"
#include "ir/dtype/type.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace trans {
enum Axis5D : int {
  N_ncdhw = 0,
  C_ncdhw,
  D_ncdhw,
  H_ncdhw,
  W_ncdhw,
};
struct TypeIdArgs {
  const void *data;
  size_t host_shape_size;  // Multiply each dimension elements. [a, b, c, d] => a*b*c*d
  TypeId host_data_type;
  TypeId device_data_type;
  size_t data_size;
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

size_t CubeSizeByType(const TypeId data_type);

std::vector<size_t> PaddingShape(const std::vector<size_t> &shape, const std::string &format,
                                 const std::string &pad_index = {""});
std::vector<size_t> PaddingShapeTo4d(const std::vector<size_t> &shape, const std::string &padding_axis = {""});
std::vector<size_t> PaddingShapeTo5d(const std::vector<size_t> &shape, const std::string &padding_axis = {""});
std::vector<size_t> PaddingShapeTo5dDefault(const std::vector<size_t> &shape);
void StringToAxisVector4D(const std::string &reshape_type_str, std::vector<Axis> *reshape_type_vec);
void StringToAxisVector5D(const std::string &reshape_type_str, std::vector<Axis5D> *reshape_type_vec);
ShapeVector GetRuntimePaddingShape(const AnfNodePtr &node, size_t index);
bool IsNeedPadding(const std::string &format, const size_t shape_size);
std::vector<size_t> TransShapeToDevice(const std::vector<size_t> &shape, const std::string &format);
bool TransDataType(const TypeIdArgs &args, void *result);
bool TransFormat(const FormatArgs &args, void *result);
bool TransFormatFromDeviceToHost(const FormatArgs &args, void *result);

// host to device
bool NchwTo4D(const FormatArgs &args, void *result);
bool NchwToFracZ(const FormatArgs &args, void *result);
bool NchwToFracNz(const FormatArgs &args, void *result);
bool NchwToNc1hwc0(const FormatArgs &args, void *result);
bool NcdhwToFracZ3D(const FormatArgs &args, void *result);
bool NchwToFracZc04(const FormatArgs &args, void *result);
bool NchwToNc1hwc04(const FormatArgs &args, void *result);
bool NchwToC1hwncoc0(const FormatArgs &args, void *result);
bool NcdhwToNdc1hwc0(const FormatArgs &args, void *result);

// device to host
bool ToNchw(const FormatArgs &args, void *result);
bool FracZToNchw(const FormatArgs &args, void *result);
bool FracNzToNchw(const FormatArgs &args, void *result);
bool Nc1hwc0ToNchw(const FormatArgs &args, void *result);
bool Nc1hwc04ToNchw(const FormatArgs &args, void *result);
bool FracZ3DToNcdhw(const FormatArgs &args, void *result);
bool C1hwncoc0ToNchw(const FormatArgs &args, void *result);
bool Ndc1hwc0ToNcdhw(const FormatArgs &args, void *result);
using FormatTransfer = std::function<bool(const FormatArgs &, void *)>;
const std::map<std::string, FormatTransfer> kTransFormatMapOfHostToDevice{
  {kOpFormat_FRAC_Z, NchwToFracZ},           {kOpFormat_FRAC_NZ, NchwToFracNz},
  {kOpFormat_NC1HWC0, NchwToNc1hwc0},        {kOpFormat_C1HWNCoC0, NchwToC1hwncoc0},
  {kOpFormat_FRACTAL_Z_C04, NchwToFracZc04}, {kOpFormat_NC1HWC0_C04, NchwToNc1hwc04},
  {kOpFormat_NDC1HWC0, NcdhwToNdc1hwc0},     {kOpFormat_FRACTAL_Z_3D, NcdhwToFracZ3D}};
}  // namespace trans
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_COMMON_TRANS_H
