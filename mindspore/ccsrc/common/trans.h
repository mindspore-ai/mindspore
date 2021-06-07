/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
constexpr size_t kDims2 = 2;
constexpr size_t H_hw = 1;
constexpr size_t W_hw = 2;
constexpr size_t HW_hw = 3;
constexpr size_t N0_fz = 1;
constexpr size_t Ni_fz = 2;
constexpr size_t C0_fz = 3;
constexpr size_t W0_nz = 1;
constexpr size_t H0_nz = 2;
constexpr size_t H1_nz = 3;
constexpr size_t W1_nz = 4;
constexpr size_t NZ_nz = 4;
enum Axis5D : int {
  N_ncdhw = 0,
  C_ncdhw,
  D_ncdhw,
  H_ncdhw,
  W_ncdhw,
};
enum AxisNc1hwc0 : int {
  N_nc1hwc0 = 0,
  C1_nc1hwc0,
  H_nc1hwc0,
  W_nc1hwc0,
  C0_nc1hwc0,
};

enum AxisNdc1hwc0 : int {
  N_ndc1hwc0 = 0,
  D_ndc1hwc0,
  C1_ndc1hwc0,
  H_ndc1hwc0,
  W_ndc1hwc0,
  C0_ndc1hwc0,
};
enum AxisFZ3D : int {
  Di_fz3d = 0,
  Ni_fz3d = 1,
  N0_fz3d = 2,
  C0_fz3d = 3,
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

std::vector<size_t> PaddingShape(const std::vector<size_t> &shape, const std::string &format,
                                 const std::string &pad_index = {""});
std::vector<size_t> PaddingShapeTo4d(const std::vector<size_t> &shape, const std::string &padding_axis = {""});
std::vector<size_t> PaddingShapeTo5d(const std::vector<size_t> &shape, const std::string &padding_axis = {""});
std::vector<size_t> PaddingShapeTo5dDefault(const std::vector<size_t> &shape);
void StringToAxisVector4D(const std::string &reshape_type_str, std::vector<Axis> *reshape_type_vec);
void StringToAxisVector5D(const std::string &reshape_type_str, std::vector<Axis5D> *reshape_type_vec);
ShapeVector GetRuntimePaddingShape(const AnfNodePtr &node, size_t index);
bool IsNeedPadding(const std::string &format, const size_t shape_size);
int64_t GetNodeGroups(const AnfNodePtr &node);
std::vector<size_t> TransShapeToDevice(const std::vector<size_t> &shape, const std::string &format,
                                       const int64_t groups = 1);
std::vector<size_t> TransShapeToDevice(const std::vector<size_t> &shape, const std::string &format,
                                       const AnfNodePtr &node, const size_t index);
bool TransDataType(const TypeIdArgs &args, void *result);
bool TransFormat(const FormatArgs &args, void *result, int64_t groups = 1);
bool TransFormat(const FormatArgs &args, void *result, const AnfNodePtr &node, const size_t index);
bool TransFormatFromDeviceToHost(const FormatArgs &args, void *result, int64_t groups = 1);
bool TransFormatFromDeviceToHost(const FormatArgs &args, void *result, const AnfNodePtr &node, const size_t index);

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
bool NchwToFracZWithGroups(const FormatArgs &args, void *result, int64_t groups);

// device to host
bool ToNchw(const FormatArgs &args, void *result);
bool FracZToNchw(const FormatArgs &args, void *result);
bool FracNzToNchw(const FormatArgs &args, void *result);
bool Nc1hwc0ToNchw(const FormatArgs &args, void *result);
bool Nc1hwc04ToNchw(const FormatArgs &args, void *result);
bool FracZ3DToNcdhw(const FormatArgs &args, void *result);
bool C1hwncoc0ToNchw(const FormatArgs &args, void *result);
bool Ndc1hwc0ToNcdhw(const FormatArgs &args, void *result);
bool FracZToNchwWithGroups(const FormatArgs &args, void *result, int64_t groups);
using FormatTransfer = std::function<bool(const FormatArgs &, void *)>;
const std::map<std::string, FormatTransfer> kTransFormatMapOfHostToDevice{
  {kOpFormat_FRAC_Z, NchwToFracZ},           {kOpFormat_FRAC_NZ, NchwToFracNz},
  {kOpFormat_NC1HWC0, NchwToNc1hwc0},        {kOpFormat_C1HWNCoC0, NchwToC1hwncoc0},
  {kOpFormat_FRACTAL_Z_C04, NchwToFracZc04}, {kOpFormat_NC1HWC0_C04, NchwToNc1hwc04},
  {kOpFormat_NDC1HWC0, NcdhwToNdc1hwc0},     {kOpFormat_FRACTAL_Z_3D, NcdhwToFracZ3D}};
}  // namespace trans
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_COMMON_TRANS_H
