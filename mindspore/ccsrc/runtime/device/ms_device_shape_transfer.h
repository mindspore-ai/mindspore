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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_MS_DEVICE_SHAPE_TRANSFER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_MS_DEVICE_SHAPE_TRANSFER_H_
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <set>
#include <utility>
#include <vector>
#include <numeric>
#include <optional>
#include "kernel/oplib/oplib.h"
#include "ir/dtype.h"
#include "kernel/kernel.h"
#include "ir/dtype/type.h"
#include "utils/shape_utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_utils.h"
#include "abstract/utils.h"
#include "runtime/device/convert_tensor_utils.h"
#include "include/common/utils/convert_utils.h"
#include "utils/log_adapter.h"
#include "include/common/utils/utils.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace trans {
constexpr int64_t kAlign16 = 16;
enum kAxis4D : int { kN = 0, kC, kH, kW, kNchwDims };
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
using ShapeVector = std::vector<int64_t>;
using RangePair = std::vector<std::pair<int64_t, int64_t>>;
/**
 * Args when trans node's data type
 * */
struct TypeIdArgs {
  const void *data;
  int64_t src_shape_size;  // Multiply each dimension elements. [a, b, c, d] => a*b*c*d
  TypeId src_data_type;
  TypeId dst_data_type;
  size_t data_size;
};

/**
 * Args when trans node's data at host
 * */
struct FormatArgs {
  const void *data;
  const size_t device_size;
  std::string host_format;
  std::string device_format;
  ShapeVector host_shape;
  ShapeVector device_shape;
  TypeId src_data_type;
};

/**
 * Trans data type at host from src type to dst type
 * */
class DataTypeTransfer {
 public:
  DataTypeTransfer() = default;
  ~DataTypeTransfer() = default;
  bool TransDataType(const TypeIdArgs &args, void *result) const;

 private:
  enum class DataTypeTransMode {
    FROM_BOOL_TO_UINT8,
    FROM_BOOL_TO_INT32,
    FROM_BOOL_TO_FLOAT16,
    FROM_BOOL_TO_FLOAT,
    FROM_INT8_TO_INT32,
    FROM_INT8_TO_FLOAT,
    FROM_INT8_TO_FLOAT16,
    FROM_UINT8_TO_INT32,
    FROM_UINT8_TO_FLOAT16,
    FROM_UINT8_TO_FLOAT,
    FROM_UINT16_TO_INT32,
    FROM_INT32_TO_BOOL,
    FROM_INT32_TO_INT8,
    FROM_INT32_TO_UINT8,
    FROM_INT32_TO_INT64,
    FROM_INT32_TO_FLOAT16,
    FROM_INT32_TO_FLOAT,
    FROM_INT64_TO_INT32,
    FROM_FLOAT16_TO_UINT8,
    FROM_FLOAT16_TO_INT32,
    FROM_FLOAT16_TO_FLOAT,
    FROM_FLOAT_TO_INT32,
    FROM_FLOAT_TO_FLOAT16,
    FROM_FLOAT32_TO_FLOAT64,
    FROM_FLOAT64_TO_FLOAT32
  };
  const std::map<std::pair<TypeId, TypeId>, DataTypeTransMode> mode_map = {
    {std::pair<TypeId, TypeId>(kNumberTypeFloat64, kNumberTypeFloat32), DataTypeTransMode::FROM_FLOAT64_TO_FLOAT32},
    {std::pair<TypeId, TypeId>(kNumberTypeFloat32, kNumberTypeFloat64), DataTypeTransMode::FROM_FLOAT32_TO_FLOAT64},
    {std::pair<TypeId, TypeId>(kNumberTypeFloat32, kNumberTypeFloat16), DataTypeTransMode::FROM_FLOAT_TO_FLOAT16},
    {std::pair<TypeId, TypeId>(kNumberTypeFloat32, kNumberTypeInt32), DataTypeTransMode::FROM_FLOAT_TO_INT32},
    {std::pair<TypeId, TypeId>(kNumberTypeFloat16, kNumberTypeFloat32), DataTypeTransMode::FROM_FLOAT16_TO_FLOAT},
    {std::pair<TypeId, TypeId>(kNumberTypeFloat16, kNumberTypeInt32), DataTypeTransMode::FROM_FLOAT16_TO_INT32},
    {std::pair<TypeId, TypeId>(kNumberTypeFloat16, kNumberTypeUInt8), DataTypeTransMode::FROM_FLOAT16_TO_UINT8},
    {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeFloat32), DataTypeTransMode::FROM_INT32_TO_FLOAT},
    {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeFloat16), DataTypeTransMode::FROM_INT32_TO_FLOAT16},
    {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeUInt8), DataTypeTransMode::FROM_INT32_TO_UINT8},
    {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeInt8), DataTypeTransMode::FROM_INT32_TO_INT8},
    {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeInt64), DataTypeTransMode::FROM_INT32_TO_INT64},
    {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeBool), DataTypeTransMode::FROM_INT32_TO_BOOL},
    {std::pair<TypeId, TypeId>(kNumberTypeUInt8, kNumberTypeFloat32), DataTypeTransMode::FROM_UINT8_TO_FLOAT},
    {std::pair<TypeId, TypeId>(kNumberTypeUInt8, kNumberTypeInt32), DataTypeTransMode::FROM_UINT8_TO_INT32},
    {std::pair<TypeId, TypeId>(kNumberTypeUInt8, kNumberTypeFloat16), DataTypeTransMode::FROM_UINT8_TO_FLOAT16},
    {std::pair<TypeId, TypeId>(kNumberTypeInt8, kNumberTypeFloat32), DataTypeTransMode::FROM_INT8_TO_FLOAT},
    {std::pair<TypeId, TypeId>(kNumberTypeInt8, kNumberTypeFloat16), DataTypeTransMode::FROM_INT8_TO_FLOAT16},
    {std::pair<TypeId, TypeId>(kNumberTypeInt8, kNumberTypeInt32), DataTypeTransMode::FROM_INT8_TO_INT32},
    {std::pair<TypeId, TypeId>(kNumberTypeInt64, kNumberTypeInt32), DataTypeTransMode::FROM_INT64_TO_INT32},
    {std::pair<TypeId, TypeId>(kNumberTypeUInt16, kNumberTypeInt32), DataTypeTransMode::FROM_UINT16_TO_INT32},
    {std::pair<TypeId, TypeId>(kNumberTypeBool, kNumberTypeInt32), DataTypeTransMode::FROM_BOOL_TO_INT32},
    {std::pair<TypeId, TypeId>(kNumberTypeBool, kNumberTypeFloat), DataTypeTransMode::FROM_BOOL_TO_FLOAT},
    {std::pair<TypeId, TypeId>(kNumberTypeBool, kNumberTypeUInt8), DataTypeTransMode::FROM_BOOL_TO_UINT8},
    {std::pair<TypeId, TypeId>(kNumberTypeBool, kNumberTypeFloat16), DataTypeTransMode::FROM_BOOL_TO_FLOAT16}};

  bool CastKernel(const TypeIdArgs &args, void *dst, int64_t data_size, DataTypeTransMode mode) const;
};

/**
 * Trans host shape to device shape according to node's format
 * */
class BACKEND_EXPORT DeviceShapeTransfer {
 public:
  DeviceShapeTransfer() = default;
  ~DeviceShapeTransfer() = default;
  ShapeVector GetDeviceShapeByFormat(const ShapeVector &shape, const std::string &format, const AnfNodePtr &node,
                                     size_t index, const TypeId &type, bool is_output = true) const;

  ShapeVector GetDeviceShapeByFormat(const ShapeVector &shape, const std::string &format, const TypeId &type,
                                     int64_t groups = 1,
                                     const ShapeVector &input_hidden_size = {kAlign16, kAlign16}) const;

 private:
  ShapeVector GetAttrInputAndHiddenSize(const AnfNodePtr &node) const;
  std::optional<ShapeVector> GetFixedDeviceShape(const ShapeVector &, const AnfNodePtr &node, size_t index,
                                                 bool is_output = true) const;
  ShapeVector TransCore(const ShapeVector &shape, const std::string &format, const TypeId &type, int64_t groups = 1,
                        const ShapeVector &input_hidden_size = {kAlign16, kAlign16}) const;

  // trans functions
  static ShapeVector NCHWDeviceShape(const ShapeVector &shape, const TypeId &);
  static ShapeVector NHWCDeviceShape(const ShapeVector &shape, const TypeId &);
  static ShapeVector HWCNDeviceShape(const ShapeVector &shape, const TypeId &);
  static ShapeVector NCDHWDeviceShape(const ShapeVector &shape, const TypeId &);
  static ShapeVector NC1HWC04DeviceShape(const ShapeVector &shape, const TypeId &);
  static ShapeVector FRAC_ZC04DeviceShape(const ShapeVector &shape, const TypeId &);
  static ShapeVector ChannelLastDeviceShape(const ShapeVector &shape, const TypeId &);
  static ShapeVector FRAC_ZN_LSTMDeviceShape(const ShapeVector &shape, const TypeId &);
  static ShapeVector FRAC_ZDeviceShape(const ShapeVector &shape, const TypeId &type);
  static ShapeVector FRAC_NZDeviceShape(const ShapeVector &shape, const TypeId &type);
  static ShapeVector NC1HWC0DeviceShape(const ShapeVector &shape, const TypeId &type);
  static ShapeVector NDC1HWC0DeviceShape(const ShapeVector &shape, const TypeId &type);
  static ShapeVector FRAC_Z3DDeviceShape(const ShapeVector &shape, const TypeId &type);
  static ShapeVector C1HWNCOC0DeviceShape(const ShapeVector &shape, const TypeId &type);
  static ShapeVector NDRNNBiasDeviceShape(const ShapeVector &shape, const TypeId &type, int64_t hidden_size = 16);
  static ShapeVector FRAC_ZDeviceShapeWithGroups(const ShapeVector &shape, const TypeId &type, int64_t groups = 1);
  static ShapeVector FRAC_ZN_RNNDeviceShape(const ShapeVector &shape, const TypeId &type,
                                            const ShapeVector &input_hidden_size = {kAlign16, kAlign16});
};

/**
 * Trans data at host according to the node's format
 * */
class FormatTransfer {
 public:
  FormatTransfer() = default;
  ~FormatTransfer() = default;

  bool TransDataByFormat(const FormatArgs &args, void *result, const AnfNodePtr &node, size_t index, bool is_forward);
  bool TransDataForwardCore(const FormatArgs &args, void *result, int64_t groups = 1);
  bool TransDataBackwordCore(const FormatArgs &args, void *result, int64_t groups = 1);

 private:
  using TransferCore = std::function<bool(const FormatArgs &, void *)>;
  // fp map
  const std::map<std::string, TransferCore> format_trans_fp_map = {{kOpFormat_HWCN, NCHW_TO_4D},
                                                                   {kOpFormat_NHWC, NCHW_TO_4D},
                                                                   {kOpFormat_FRAC_Z, NCHW_TO_FRAC_Z},
                                                                   {kOpFormat_FRAC_NZ, NCHW_TO_FRAC_NZ},
                                                                   {kOpFormat_NC1HWC0, NCHW_TO_NC1HWC0},
                                                                   {kOpFormat_NDC1HWC0, NCDHW_TO_NDC1HWC0},
                                                                   {kOpFormat_C1HWNCoC0, NCHW_TO_C1HWNCOC0},
                                                                   {kOpFormat_NC1HWC0_C04, NCHW_TO_NC1HWC04},
                                                                   {kOpFormat_FRACTAL_Z_3D, NCDHW_TO_FRAC_Z3D},
                                                                   {kOpFormat_FRACTAL_Z_C04, NCHW_TO_FRAC_ZC04}};
  // bp map
  const std::map<std::string, TransferCore> format_trans_bp_map = {{kOpFormat_HWCN, TO_NCHW},
                                                                   {kOpFormat_NHWC, TO_NCHW},
                                                                   {kOpFormat_FRAC_Z, FRAC_Z_TO_NCHW},
                                                                   {kOpFormat_FRAC_NZ, FRAC_NZ_TO_NCHW},
                                                                   {kOpFormat_NC1HWC0, NC1HWC0_TO_NCHW},
                                                                   {kOpFormat_NDC1HWC0, NDC1HWC0_TO_NCDHW},
                                                                   {kOpFormat_C1HWNCoC0, C1HWNCOC0_TO_NCHW},
                                                                   {kOpFormat_NC1HWC0_C04, NC1HWC04_TO_NCHW},
                                                                   {kOpFormat_FRACTAL_Z_3D, FRAC_Z3D_TO_NCDHW}};

  static bool CheckArgs(const FormatArgs &args, int64_t *size);
  static bool TransShapeToHW_NZ(const ShapeVector &host_shape, ShapeVector *hw_shape);
  // HOST TO DEVICE
  static bool NCHW_TO_4D(const FormatArgs &args, void *result);
  static bool NCHW_TO_FRAC_Z(const FormatArgs &args, void *result);
  static bool NCHW_TO_NC1HWC0(const FormatArgs &args, void *result);
  static bool NCHW_TO_FRAC_NZ(const FormatArgs &args, void *result);
  static bool NCHW_TO_NC1HWC04(const FormatArgs &args, void *result);
  static bool NCHW_TO_FRAC_ZC04(const FormatArgs &args, void *result);
  static bool NCHW_TO_C1HWNCOC0(const FormatArgs &args, void *result);
  static bool NCDHW_TO_NDC1HWC0(const FormatArgs &args, void *result);
  static bool NCDHW_TO_FRAC_Z3D(const FormatArgs &args, void *result);
  static bool NCHW_TO_FRAC_Z_WITH_GROUPS(const FormatArgs &args, void *result, bool to_device, int64_t groups);

  // DEVICE TO HOST
  static bool TO_NCHW(const FormatArgs &args, void *result);
  static bool FRAC_Z_TO_NCHW(const FormatArgs &args, void *result);
  static bool FRAC_NZ_TO_NCHW(const FormatArgs &args, void *result);
  static bool NC1HWC0_TO_NCHW(const FormatArgs &args, void *result);
  static bool NC1HWC04_TO_NCHW(const FormatArgs &args, void *result);
  static bool C1HWNCOC0_TO_NCHW(const FormatArgs &args, void *result);
  static bool FRAC_Z3D_TO_NCDHW(const FormatArgs &args, void *result);
  static bool NDC1HWC0_TO_NCDHW(const FormatArgs &args, void *result);
  static bool FRAC_Z_TO_NCHW_WITH_GROUPS(const FormatArgs &args, void *result, int64_t groups);

  // common check_func
  static int64_t Common4DCheck(const FormatArgs &args);
};

/**
 * Range trans function
 * */
class BACKEND_EXPORT ShapeRangeTransfer {
 public:
  ShapeRangeTransfer() = default;
  ~ShapeRangeTransfer() = default;
  RangePair GetRealRange(const RangePair &ori_range, const std::string &format, const TypeId &type,
                         const std::string &padding_str = {""}) const;

 private:
  static RangePair NHWCRange(const RangePair &ori_range, const TypeId &);
  static RangePair HWCNRange(const RangePair &ori_range, const TypeId &);
  static RangePair NC1HWC04Range(const RangePair &ori_range, const TypeId &);
  static RangePair FRAC_ZC04Range(const RangePair &ori_range, const TypeId &);
  static RangePair FRAC_ZN_LSTMRange(const RangePair &ori_range, const TypeId &);
  static RangePair FRAC_ZRange(const RangePair &ori_range, const TypeId &type);
  static RangePair FRAC_NZRange(const RangePair &ori_range, const TypeId &type);
  static RangePair NC1HWC0Range(const RangePair &ori_range, const TypeId &type);
  static RangePair NDC1HWC0Range(const RangePair &ori_range, const TypeId &type);
  static RangePair C1HWNCOC0Range(const RangePair &ori_range, const TypeId &type);
  static RangePair FRAC_Z_3DRange(const RangePair &ori_range, const TypeId &type);
};

/**
 * If you want extend format, make sure it has a data trans function at host in class
 * 'FormatTransfer.format_trans_fp_map'
 * */
static const std::set<std::string> kFormatWithTransFunc = {
  kOpFormat_HWCN,     kOpFormat_NHWC,      kOpFormat_FRAC_Z,      kOpFormat_FRAC_NZ,      kOpFormat_NC1HWC0,
  kOpFormat_NDC1HWC0, kOpFormat_C1HWNCoC0, kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_3D, kOpFormat_FRACTAL_Z_C04};

/**
 * Interface of datatype trans
 * */
BACKEND_EXPORT bool TransDataType(const TypeIdArgs &args, void *result);

/**
 * Interface of data format trans from host to device
 * */
BACKEND_EXPORT bool TransFormat(const FormatArgs &args, void *result, const AnfNodePtr &node, size_t index);

/**
 * Interface of data format trans from host to device
 * */
BACKEND_EXPORT bool TransFormatFromDeviceToHost(const FormatArgs &args, void *result, int64_t groups = 1);

/**
 * Interface of data format trans from device to host
 * */
BACKEND_EXPORT bool TransFormatFromDeviceToHost(const FormatArgs &args, void *result, const AnfNodePtr &node,
                                                size_t index);

/**
 * 4D reshape type trans, trans reshape_type from string to int
 * */
BACKEND_EXPORT void StringToAxisVector4D(const std::string &reshape_type_str, std::vector<Axis> *reshape_type_vec);

/**
 * 5D reshape type trans, trans reshape_type from string to int
 * */
BACKEND_EXPORT void StringToAxisVector5D(const std::string &reshape_type_str, std::vector<Axis5D> *reshape_type_vec);

/**
 * Get shape after padding
 * */
BACKEND_EXPORT ShapeVector GetRuntimePaddingShape(const AnfNodePtr &node, size_t index);

/**
 *  If need padding
 * */
BACKEND_EXPORT bool IsNeedPadding(const std::string &format, size_t shape_size);

/**
 * Padding shape to 5D by default mode
 * */
template <typename T>
std::vector<T> PaddingShapeTo5dDefault(const std::vector<T> &shape, const AnfNodePtr &node = nullptr) {
  if (shape.size() >= kDim5) {
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
      auto node_info = (node != nullptr) ? ". Node: " + node->fullname_with_scope() : " .";
      MS_LOG(EXCEPTION) << "Unexpected shape :" << shape << node_info;
  }
  return shape_5d;
}

/**
 * Padding shape to 4D by default mode
 * */
template <typename T>
std::vector<T> PaddingShapeTo4dDefault(const std::vector<T> &shape, const AnfNodePtr &node = nullptr) {
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
      return shape;
    default:
      auto node_info = (node != nullptr) ? ". Node: " + node->fullname_with_scope() : " .";
      MS_LOG(EXCEPTION) << "Unexpected shape : " << shape << node_info;
  }
  return shape_4d;
}

/**
 * Padding shape to 5D according to reshape type
 * */
template <typename T>
std::vector<T> PaddingShapeTo5d(const std::vector<T> &shape, const std::string &padding_str = {""}) {
  std::vector<Axis5D> padding_axis;
  StringToAxisVector5D(padding_str, &padding_axis);
  if (padding_axis.empty() || shape.size() > padding_axis.size()) {
    return PaddingShapeTo5dDefault(shape);
  }
  std::vector<T> shape_5d(kNcdhw, 1);
  for (size_t index = 0; index < shape.size(); index++) {
    shape_5d[padding_axis[index]] = shape[index];
  }
  return shape_5d;
}

/**
 * Padding shape to 4D according to reshape type
 * */
template <typename T>
std::vector<T> PaddingShapeTo4d(const std::vector<T> &shape, const std::string &padding_str = {""}) {
  std::vector<Axis> padding_axis;
  StringToAxisVector4D(padding_str, &padding_axis);
  if (padding_axis.empty() || shape.size() > padding_axis.size()) {
    return PaddingShapeTo4dDefault(shape);
  }
  std::vector<T> shape_4d(kNchwDims, 1);
  for (size_t index = 0; index < shape.size(); index++) {
    shape_4d[padding_axis[index]] = shape[index];
  }
  return shape_4d;
}

/**
 * Interface of padding shape
 * */
template <typename T>
std::vector<T> PaddingShape(const std::vector<T> &shape, const std::string &format, const std::string &pad_index = {""},
                            const AnfNodePtr &node = nullptr) {
  if (node != nullptr) {
    MS_LOG(DEBUG) << "Start padding shape for node: [" << node->fullname_with_scope() << "], format: " << format
                  << ", detail info: " << node->DebugString();
  }
  std::vector<T> host_shape;
  if (IsOneOf3DFormat(format)) {
    if (shape.size() >= kDim5) {
      return shape;
    }
    host_shape = PaddingShapeTo5d(shape, pad_index);
  } else {
    host_shape = PaddingShapeTo4d(shape, pad_index);
  }
  return host_shape;
}

/**
 * Interface of transform pad_index string to AxisVector
 * */
template <typename T>
std::vector<int> StringToAxisVector(const std::vector<T> &shape, const std::string &format,
                                    const std::string &pad_index = {""}, const AnfNodePtr &node = nullptr) {
  if (node != nullptr) {
    MS_LOG(DEBUG) << "Start transform  pad_index to axis_vecor for node: [" << node->fullname_with_scope()
                  << "], format: " << format << ", detail info: " << node->DebugString();
  }

  std::vector<int> padding_axis;
  if (IsOneOf3DFormat(format)) {
    if (shape.size() >= kDim5) {
      return padding_axis;
    }
    std::vector<Axis5D> padding_axis_5d;
    StringToAxisVector5D(pad_index, &padding_axis_5d);

    if (padding_axis_5d.empty() || shape.size() != padding_axis_5d.size()) {
      for (int index = 0; index < static_cast<int>(shape.size()); ++index) {
        padding_axis.push_back(index);
      }
    } else {
      (void)std::transform(padding_axis_5d.begin(), padding_axis_5d.end(), std::back_inserter(padding_axis),
                           [](Axis5D x) { return static_cast<int>(x); });
    }
  } else {
    std::vector<Axis> padding_axis_4d;
    StringToAxisVector4D(pad_index, &padding_axis_4d);

    if (padding_axis_4d.empty() || shape.size() != padding_axis_4d.size()) {
      for (int index = 0; index < static_cast<int>(shape.size()); ++index) {
        padding_axis.push_back(index);
      }
    } else {
      (void)std::transform(padding_axis_4d.begin(), padding_axis_4d.end(), std::back_inserter(padding_axis),
                           [](Axis x) { return static_cast<int>(x); });
    }
  }

  return padding_axis;
}

/**
 * Interface of device shape trance
 * */
template <typename T>
std::vector<T> TransShapeToDevice(const std::vector<T> &shape, const std::string &format, const AnfNodePtr &node,
                                  size_t index, TypeId type, bool is_output = true) {
  ShapeVector shape_before;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_before),
                       [](T num) { return static_cast<int64_t>(num); });
  DeviceShapeTransfer deviceShapeTransfer;
  auto res = deviceShapeTransfer.GetDeviceShapeByFormat(shape_before, format, node, index, type, is_output);
  std::vector<T> out_shape;
  (void)std::transform(res.begin(), res.end(), std::back_inserter(out_shape),
                       [](int64_t num) { return static_cast<T>(num); });
  return out_shape;
}

template <typename T>
std::vector<T> TransShapeToDevice(const std::vector<T> &shape, const std::string &format, TypeId type,
                                  int64_t groups = 1, const ShapeVector &input_hidden_size = {kAlign16, kAlign16}) {
  ShapeVector shape_before;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_before),
                       [](T num) { return static_cast<int64_t>(num); });
  DeviceShapeTransfer deviceShapeTransfer;
  auto res = deviceShapeTransfer.GetDeviceShapeByFormat(shape_before, format, type, groups, input_hidden_size);
  std::vector<T> out_shape;
  (void)std::transform(res.begin(), res.end(), std::back_inserter(out_shape),
                       [](int64_t num) { return static_cast<T>(num); });
  return out_shape;
}
}  // namespace trans
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_MS_DEVICE_SHAPE_TRANSFER_H_
