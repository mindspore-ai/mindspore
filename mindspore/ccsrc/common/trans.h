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
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace trans {
constexpr int64_t kAlign16 = 16;

enum kAxis : int { kN = 0, kC, kH, kW, kNchwDims };

enum Axis5D : int {
  N_ncdhw = 0,
  C_ncdhw,
  D_ncdhw,
  H_ncdhw,
  W_ncdhw,
  kNcdhw,
  N_ndc1hwc0 = 0,
  D_ndc1hwc0,
  C1_ndc1hwc0,
  H_ndc1hwc0,
  W_ndc1hwc0,
  C0_ndc1hwc0
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

int64_t GetAttrGroups(const AnfNodePtr &node, const size_t index);
std::vector<int64_t> GetAttrInputAndHiddenSize(const AnfNodePtr &node);
void StringToAxisVector4D(const std::string &reshape_type_str, std::vector<Axis> *reshape_type_vec);
void StringToAxisVector5D(const std::string &reshape_type_str, std::vector<Axis5D> *reshape_type_vec);
ShapeVector GetRuntimePaddingShape(const AnfNodePtr &node, size_t index);
bool IsNeedPadding(const std::string &format, const size_t shape_size);
int64_t GetNodeGroups(const AnfNodePtr &node);
std::vector<size_t> TransShapeToDevice(const std::vector<size_t> &shape, const std::string &format,
                                       const int64_t groups = 1,
                                       const std::vector<int64_t> &input_hidden_size = {kAlign16, kAlign16});
std::vector<int64_t> TransShapeToDevice(const std::vector<int64_t> &shape, const std::string &format,
                                        const int64_t groups = 1,
                                        const std::vector<int64_t> &input_hidden_size = {kAlign16, kAlign16});
template <typename T>
std::vector<T> TransShapeToDevice(const std::vector<T> &shape, const std::string &format, const AnfNodePtr &node,
                                  const size_t index, bool is_output = true) {
  int64_t groups = 1;
  if (format == kOpFormat_FRAC_Z) {
    groups = GetAttrGroups(node, index);
  }
  std::vector<int64_t> input_hidden_size = {kAlign16, kAlign16};
  if (format == kOpFormat_FRACTAL_ZN_RNN || format == kOpFormat_ND_RNN_BIAS) {
    input_hidden_size = GetAttrInputAndHiddenSize(node);
  }
  if (node != nullptr) {
    MS_LOG(DEBUG) << "Start trans infer shape to device shape for node: " << node->DebugString()
                  << ", format: " << format;
  }

  return TransShapeToDevice(shape, format, groups, input_hidden_size);
}
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

template <typename T>
std::vector<T> PaddingShapeTo5dDefault(const std::vector<T> &shape) {
  if (shape.size() >= kNcdhw) {
    return shape;
  }
  std::vector<T> shape_5d(kNcdhw, 1);
  switch (shape.size()) {
    case N_ncdhw:
      return shape_5d;
    case C_ncdhw:
      shape_5d[C_ncdhw] = shape[N_ncdhw];
      break;
    case D_ncdhw:
      shape_5d[C_ncdhw] = shape[N_ncdhw];
      shape_5d[D_ncdhw] = shape[C_ncdhw];
      break;
    case H_ncdhw:
      shape_5d[C_ncdhw] = shape[N_ncdhw];
      shape_5d[D_ncdhw] = shape[C_ncdhw];
      shape_5d[H_ncdhw] = shape[D_ncdhw];
      break;
    case W_ncdhw:
      shape_5d[C_ncdhw] = shape[N_ncdhw];
      shape_5d[D_ncdhw] = shape[C_ncdhw];
      shape_5d[H_ncdhw] = shape[D_ncdhw];
      shape_5d[W_ncdhw] = shape[H_ncdhw];
      break;
    default:
      MS_LOG(EXCEPTION) << "Unexpected shape size = " << shape.size();
  }
  return shape_5d;
}

template <typename T>
std::vector<T> PaddingShapeTo4dDefault(const std::vector<T> &shape) {
  std::vector<T> shape_4d(kNchwDims, 1);
  switch (shape.size()) {
    case kN:
      return shape_4d;
    case kC:
      shape_4d[kC] = shape[kN];
      break;
    case kH:
      shape_4d[kC] = shape[kN];
      shape_4d[kH] = shape[kC];
      break;
    case kW:
      shape_4d[kC] = shape[kN];
      shape_4d[kH] = shape[kC];
      shape_4d[kW] = shape[kH];
      break;
    case kNchwDims:
      std::copy(shape.begin(), shape.end(), shape_4d.begin());
      break;
    default:
      MS_LOG(EXCEPTION) << "Unexpected shape size = " << shape.size();
  }
  return shape_4d;
}

template <typename T>
std::vector<T> PaddingShapeTo5d(const std::vector<T> &shape, const std::string &padding_str = {""}) {
  std::vector<Axis5D> padding_axis;
  StringToAxisVector5D(padding_str, &padding_axis);
  if (padding_axis.empty() || shape.size() != padding_axis.size()) {
    return PaddingShapeTo5dDefault(shape);
  }
  std::vector<T> shape_5d(kNcdhw, 1);
  for (size_t index = 0; index < padding_axis.size(); index++) {
    shape_5d[padding_axis[index]] = shape[index];
  }
  return shape_5d;
}

template <typename T>
std::vector<T> PaddingShapeTo4d(const std::vector<T> &shape, const std::string &padding_str = {""}) {
  std::vector<Axis> padding_axis;
  StringToAxisVector4D(padding_str, &padding_axis);
  if (padding_axis.empty() || shape.size() != padding_axis.size()) {
    return PaddingShapeTo4dDefault(shape);
  }
  std::vector<T> shape_4d(kNchwDims, 1);
  for (size_t index = 0; index < padding_axis.size(); index++) {
    shape_4d[padding_axis[index]] = shape[index];
  }
  return shape_4d;
}

template <typename T>
std::vector<T> PaddingShape(const std::vector<T> &shape, const std::string &format,
                            const std::string &pad_index = {""}) {
  std::vector<T> host_shape;
  if (k3DFormatSet.find(format) != k3DFormatSet.end()) {
    if (shape.size() >= kNcdhw) {
      return shape;
    }
    host_shape = trans::PaddingShapeTo5d(shape, pad_index);
  } else {
    host_shape = trans::PaddingShapeTo4d(shape, pad_index);
  }
  return host_shape;
}
}  // namespace trans
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_COMMON_TRANS_H
