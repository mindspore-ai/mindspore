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
#include "common/trans.h"
#include <functional>
#include <numeric>
#include <utility>
#include "common/utils.h"
#include "session/anf_runtime_algorithm.h"
#include "kernel/kernel.h"
#include "device/convert_tensor_utils.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/utils.h"

namespace mindspore {
namespace trans {
enum kAxis : int { kN = 0, kC, kH, kW, kNchwDims, kNdhwc };
const std::map<TypeId, size_t> type_map = {{kNumberTypeBool, 1},    {kNumberTypeInt, 4},     {kNumberTypeInt8, 1},
                                           {kNumberTypeInt16, 2},   {kNumberTypeInt32, 4},   {kNumberTypeInt64, 8},
                                           {kNumberTypeUInt, 4},    {kNumberTypeUInt8, 1},   {kNumberTypeUInt16, 2},
                                           {kNumberTypeUInt32, 4},  {kNumberTypeUInt64, 8},  {kNumberTypeFloat, 4},
                                           {kNumberTypeFloat16, 2}, {kNumberTypeFloat32, 4}, {kNumberTypeFloat64, 8}};

inline void SetData(size_t size, bool pad_zero, size_t src_idx, size_t dst_idx, const FormatArgs &args, void *result) {
  switch (size) {
    case 1:
      static_cast<uint8_t *>(result)[dst_idx] = pad_zero ? 0 : static_cast<const uint8_t *>(args.data)[src_idx];
      break;
    case 2:
      static_cast<uint16_t *>(result)[dst_idx] = pad_zero ? 0 : static_cast<const uint16_t *>(args.data)[src_idx];
      break;
    case 4:
      static_cast<uint32_t *>(result)[dst_idx] = pad_zero ? 0 : static_cast<const uint32_t *>(args.data)[src_idx];
      break;
    case 8:
      static_cast<uint64_t *>(result)[dst_idx] = pad_zero ? 0 : static_cast<const uint64_t *>(args.data)[src_idx];
      break;
    default:
      MS_LOG(EXCEPTION) << "Trans data not support size " << size;
  }
}

template <typename T>
T DivCeil(T n1, T n2) {
  if (n2 != 0) {
    return (n1 - 1) / n2 + 1;
  }
  return 0;
}

enum DataTypeTransMode {
  FROM_FLOAT_TO_FLOAT16,
  FROM_FLOAT_TO_INT32,
  FROM_FLOAT16_TO_FLOAT,
  FROM_FLOAT16_TO_INT32,
  FROM_FLOAT16_TO_UINT8,
  FROM_INT32_TO_FLOAT,
  FROM_INT32_TO_FLOAT16,
  FROM_INT32_TO_UINT8,
  FROM_INT32_TO_INT8,
  FROM_INT32_TO_BOOL,
  FROM_UINT8_TO_FLOAT,
  FROM_UINT8_TO_INT32,
  FROM_UINT8_TO_FLOAT16,
  FROM_INT8_TO_FLOAT,
  FROM_INT8_TO_FLOAT16,
  FROM_INT8_TO_INT32,
  FROM_INT64_TO_INT32,
  FROM_UINT16_TO_INT32,
  FROM_BOOL_TO_FLOAT,
  FROM_BOOL_TO_INT32,
  FROM_BOOL_TO_UINT8,
  FROM_BOOL_TO_FLOAT16,
  FROM_FLOAT64_TO_FLOAT32,
  FROM_FLOAT32_TO_FLOAT64
};

const std::map<std::pair<TypeId, TypeId>, DataTypeTransMode> mode_map{
  {std::pair<TypeId, TypeId>(kNumberTypeFloat64, kNumberTypeFloat32), FROM_FLOAT64_TO_FLOAT32},
  {std::pair<TypeId, TypeId>(kNumberTypeFloat32, kNumberTypeFloat64), FROM_FLOAT32_TO_FLOAT64},
  {std::pair<TypeId, TypeId>(kNumberTypeFloat32, kNumberTypeFloat16), FROM_FLOAT_TO_FLOAT16},
  {std::pair<TypeId, TypeId>(kNumberTypeFloat32, kNumberTypeInt32), FROM_FLOAT_TO_INT32},
  {std::pair<TypeId, TypeId>(kNumberTypeFloat16, kNumberTypeFloat32), FROM_FLOAT16_TO_FLOAT},
  {std::pair<TypeId, TypeId>(kNumberTypeFloat16, kNumberTypeInt32), FROM_FLOAT16_TO_INT32},
  {std::pair<TypeId, TypeId>(kNumberTypeFloat16, kNumberTypeUInt8), FROM_FLOAT16_TO_UINT8},
  {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeFloat32), FROM_INT32_TO_FLOAT},
  {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeFloat16), FROM_INT32_TO_FLOAT16},
  {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeUInt8), FROM_INT32_TO_UINT8},
  {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeInt8), FROM_INT32_TO_INT8},
  {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeBool), FROM_INT32_TO_BOOL},
  {std::pair<TypeId, TypeId>(kNumberTypeUInt8, kNumberTypeFloat32), FROM_UINT8_TO_FLOAT},
  {std::pair<TypeId, TypeId>(kNumberTypeUInt8, kNumberTypeInt32), FROM_UINT8_TO_INT32},
  {std::pair<TypeId, TypeId>(kNumberTypeUInt8, kNumberTypeFloat16), FROM_UINT8_TO_FLOAT16},
  {std::pair<TypeId, TypeId>(kNumberTypeInt8, kNumberTypeFloat32), FROM_INT8_TO_FLOAT},
  {std::pair<TypeId, TypeId>(kNumberTypeInt8, kNumberTypeFloat16), FROM_INT8_TO_FLOAT16},
  {std::pair<TypeId, TypeId>(kNumberTypeInt8, kNumberTypeInt32), FROM_INT8_TO_INT32},
  {std::pair<TypeId, TypeId>(kNumberTypeInt64, kNumberTypeInt32), FROM_INT64_TO_INT32},
  {std::pair<TypeId, TypeId>(kNumberTypeUInt16, kNumberTypeInt32), FROM_UINT16_TO_INT32},
  {std::pair<TypeId, TypeId>(kNumberTypeBool, kNumberTypeInt32), FROM_BOOL_TO_INT32},
  {std::pair<TypeId, TypeId>(kNumberTypeBool, kNumberTypeFloat), FROM_BOOL_TO_FLOAT},
  {std::pair<TypeId, TypeId>(kNumberTypeBool, kNumberTypeUInt8), FROM_BOOL_TO_UINT8},
  {std::pair<TypeId, TypeId>(kNumberTypeBool, kNumberTypeFloat16), FROM_BOOL_TO_FLOAT16}};

void CheckMemSize(const TypeIdArgs &args) {
  auto src_type_size = TypeIdSize(args.host_data_type);
  auto dst_type_size = TypeIdSize(args.device_data_type);
  if (src_type_size < 1 || dst_type_size < 1) {
    MS_LOG(EXCEPTION) << "Invalid src or dst data type.";
  }
  if (args.data_size / src_type_size != args.host_shape_size) {
    MS_LOG(EXCEPTION) << "Invalid src or dst data size.";
  }
}

template <typename SrcT, typename DstT>
void TransDataSrc2Dst(const TypeIdArgs &args, void *dst, const size_t data_size) {
  CheckMemSize(args);
  for (size_t idx = 0; idx != data_size; idx++) {
    SrcT src_data = static_cast<const SrcT *>(args.data)[idx];
    static_cast<DstT *>(dst)[idx] = static_cast<DstT>(src_data);
  }
}

template <typename SrcT>
void TransDataSrc2Fp16(const TypeIdArgs &args, void *dst, const size_t data_size) {
  CheckMemSize(args);
  auto src_data = static_cast<const SrcT *>(args.data);
  auto half_data = static_cast<Eigen::half *>(dst);
  for (size_t i = 0; i < data_size; i++) {
    half_data[i] = Eigen::half(src_data[i]);
  }
}

bool CastKernel(const TypeIdArgs &args, void *dst, const size_t data_size, const DataTypeTransMode mode) {
  using DtypeKernel = std::function<void(const TypeIdArgs &, void *, const size_t)>;
  const std::map<DataTypeTransMode, DtypeKernel> cast_kernel_map{
    {FROM_FLOAT_TO_INT32, TransDataSrc2Dst<float, int32_t>},
    {FROM_FLOAT64_TO_FLOAT32, TransDataSrc2Dst<double, float>},
    {FROM_FLOAT32_TO_FLOAT64, TransDataSrc2Dst<float, double>},
    {FROM_FLOAT16_TO_INT32, TransDataSrc2Dst<float16, int32_t>},
    {FROM_FLOAT16_TO_UINT8, TransDataSrc2Dst<float16, uint8_t>},
    {FROM_INT32_TO_FLOAT, TransDataSrc2Dst<int32_t, float>},
    {FROM_INT32_TO_INT8, TransDataSrc2Dst<int32_t, int8_t>},
    {FROM_INT32_TO_UINT8, TransDataSrc2Dst<int32_t, uint8_t>},
    {FROM_INT32_TO_BOOL, TransDataSrc2Dst<int32_t, int8_t>},
    {FROM_INT32_TO_FLOAT16, TransDataSrc2Fp16<int32_t>},
    {FROM_UINT8_TO_FLOAT, TransDataSrc2Dst<uint8_t, float>},
    {FROM_UINT8_TO_INT32, TransDataSrc2Dst<uint8_t, int32_t>},
    {FROM_UINT8_TO_FLOAT16, TransDataSrc2Fp16<uint8_t>},
    {FROM_INT8_TO_FLOAT, TransDataSrc2Dst<int8_t, float>},
    {FROM_INT8_TO_FLOAT16, TransDataSrc2Fp16<int8_t>},
    {FROM_INT8_TO_INT32, TransDataSrc2Dst<int8_t, int32_t>},
    {FROM_INT64_TO_INT32, TransDataSrc2Dst<int64_t, int32_t>},
    {FROM_UINT16_TO_INT32, TransDataSrc2Dst<uint16_t, int32_t>},
    {FROM_BOOL_TO_INT32, TransDataSrc2Dst<int8_t, int32_t>},
    {FROM_BOOL_TO_FLOAT, TransDataSrc2Dst<int8_t, float>},
    {FROM_BOOL_TO_UINT8, TransDataSrc2Dst<int8_t, uint8_t>},
    {FROM_BOOL_TO_FLOAT16, TransDataSrc2Fp16<int8_t>}};

  if (mode == FROM_FLOAT_TO_FLOAT16) {
    device::FloatToHalf(dst, args.data, data_size);
    return true;
  } else if (mode == FROM_FLOAT16_TO_FLOAT) {
    device::HalfToFloat(dst, args.data, data_size);
    return true;
  }
  auto iter = cast_kernel_map.find(mode);
  if (iter != cast_kernel_map.end()) {
    iter->second(args, dst, data_size);
    return true;
  } else {
    MS_LOG(ERROR) << "Unsupported datatype trans";
    return false;
  }
}

size_t CubeSizeByType(const TypeId data_type) {
  const size_t default_error = 0;
  auto dt_size = TypeIdSize(data_type);
  if (dt_size < 1) {
    MS_LOG(ERROR) << "Illegal dtype.";
    return default_error;
  } else if (dt_size == 1) {
    return kCubeSize * 2;
  }
  return kCubeSize;
}

size_t ShapeSize(const std::vector<size_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), IntToSize(1), std::multiplies<size_t>());
}

size_t TypeIdSize(const TypeId data_type) {
  const size_t unsupported_type_error = 0;
  auto iter = type_map.find(data_type);
  if (iter != type_map.end()) {
    return iter->second;
  }
  return unsupported_type_error;
}

namespace {
bool CheckDims(const std::vector<size_t> &shape) {
  if (shape.size() != kNchwDims) {
    MS_LOG(ERROR) << "Host shape dims shoud be 4";
    return false;
  }
  return true;
}

std::vector<size_t> NchwDeviceShape(const std::vector<size_t> &shape) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  return shape;
}

std::vector<size_t> NhwcDeviceShape(const std::vector<size_t> &shape) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Ccheck dims failed.";
  }
  std::vector<size_t> device_shape;
  device_shape.push_back(shape[kN]);
  device_shape.push_back(shape[kH]);
  device_shape.push_back(shape[kW]);
  device_shape.push_back(shape[kC]);
  return device_shape;
}

std::vector<size_t> HwchDeviceShape(const std::vector<size_t> &shape) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  std::vector<size_t> device_shape;
  device_shape.push_back(shape[kH]);
  device_shape.push_back(shape[kW]);
  device_shape.push_back(shape[kC]);
  device_shape.push_back(shape[kN]);
  return device_shape;
}

std::vector<size_t> FracZDeviceShape(const std::vector<size_t> &shape) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  std::vector<size_t> device_shape;
  const size_t cout16 = ((shape[kN] + kCubeSize - 1) / kCubeSize) * kCubeSize;
  const size_t cin16 = ((shape[kC] + kCubeSize - 1) / kCubeSize) * kCubeSize;
  device_shape.push_back(shape[kH] * shape[kW] * cin16 / kCubeSize);
  device_shape.push_back(cout16 / kCubeSize);
  device_shape.push_back(kCubeSize);
  device_shape.push_back(kCubeSize);
  return device_shape;
}

std::vector<size_t> Nc1hwc0DeviceShape(const std::vector<size_t> &shape) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  std::vector<size_t> device_shape;
  const size_t C1 = (shape[kC] + kCubeSize - 1) / kCubeSize;
  const size_t C0 = kCubeSize;
  device_shape.push_back(shape[kN]);
  device_shape.push_back(C1);
  device_shape.push_back(shape[kH]);
  device_shape.push_back(shape[kW]);
  device_shape.push_back(C0);
  return device_shape;
}

std::vector<size_t> C1hwncoc0DeviceShape(const std::vector<size_t> &shape) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  std::vector<size_t> device_shape;
  device_shape.push_back((shape[kC] - 1) / kCubeSize + 1);
  device_shape.push_back(shape[kH]);
  device_shape.push_back(shape[kW]);
  device_shape.push_back(shape[kN]);
  device_shape.push_back(kCubeSize);
  device_shape.push_back(kCubeSize);
  return device_shape;
}

std::vector<size_t> FracZc04DeviceShape(const std::vector<size_t> &shape) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  std::vector<size_t> device_shape;
  const size_t c0 = 4;
  auto first_dim = DivCeil(c0 * shape[kH] * shape[kW], kCubeSize);
  auto no = DivCeil(shape.at(kN), kCubeSize);
  device_shape.push_back(first_dim);
  device_shape.push_back(no);
  device_shape.push_back(kCubeSize);
  device_shape.push_back(kCubeSize);
  return device_shape;
}

std::vector<size_t> Nc1hwc04DeviceShape(const std::vector<size_t> &shape) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  std::vector<size_t> device_shape;
  const size_t C1 = 1;
  const size_t C0 = 4;
  device_shape.push_back(shape[kN]);
  device_shape.push_back(C1);
  device_shape.push_back(shape[kH]);
  device_shape.push_back(shape[kW]);
  device_shape.push_back(C0);
  return device_shape;
}

std::vector<size_t> NdhwcDeviceShape(const std::vector<size_t> &shape) {
  if (shape.size() < kNdhwc) {
    MS_LOG(EXCEPTION) << "Shape dims must be 5 when format is ndhwc.";
  }
  return shape;
}

std::vector<size_t> PaddingShapeTo4dByDefault(const std::vector<size_t> &shape) {
  std::vector<size_t> shape_4d(kNchwDims, 1);
  switch (shape.size()) {
    case 0:
      return shape_4d;
    case 1:
      shape_4d[kC] = shape[kN];
      break;
    case 2:
      shape_4d[kC] = shape[kN];
      shape_4d[kH] = shape[kC];
      break;
    case 3:
      shape_4d[kC] = shape[kN];
      shape_4d[kH] = shape[kC];
      shape_4d[kW] = shape[kH];
      break;
    case 4:
      std::copy(shape.begin(), shape.end(), shape_4d.begin());
      break;
    default:
      MS_LOG(EXCEPTION) << "Unexpect shape size = " << shape.size();
  }
  return shape_4d;
}
}  // namespace

bool IsNeedPadding(const std::string &format, const size_t shape_size) {
  if (shape_size == 0) {
    return false;
  }
  if (format == kOpFormat_DEFAULT || format == kOpFormat_FRAC_NZ) {
    return false;
  } else if (shape_size < kNchwDims) {
    return true;
  }
  return false;
}

std::vector<int> GetRuntimePaddingShape(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<int> shape;
  std::vector<size_t> host_shape;
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    auto tensor = node_value->cast<tensor::TensorPtr>();
    if (tensor == nullptr) {
      MS_LOG(EXCEPTION) << " The node[ " << node->DebugString() << "]'s cannot convert ";
    }
    auto shape_temp = tensor->shape();
    (void)std::transform(shape_temp.begin(), shape_temp.end(), std::back_inserter(host_shape), IntToSize);
    if (host_shape.empty()) {
      host_shape.push_back(1);
    }
  } else {
    host_shape = AnfAlgo::GetOutputInferShape(node, index);
  }
  if (trans::IsNeedPadding(AnfAlgo::GetOutputFormat(node, 0), host_shape.size())) {
    host_shape = trans::PaddingShapeTo4d(host_shape, AnfAlgo::GetOutputReshapeType(node, 0));
  }
  std::transform(host_shape.begin(), host_shape.end(), std::back_inserter(shape), SizeToInt);
  return shape;
}

std::vector<size_t> PaddingShapeTo4d(const std::vector<size_t> &shape, const std::vector<kernel::Axis> &padding_axis) {
  if (padding_axis.empty() || shape.size() != padding_axis.size()) {
    return PaddingShapeTo4dByDefault(shape);
  }
  std::vector<size_t> shape_4d(kNchwDims, 1);
  for (size_t index = 0; index < padding_axis.size(); index++) {
    shape_4d[padding_axis[index]] = shape[index];
  }
  return shape_4d;
}

std::vector<size_t> TransShapeToDevice(const std::vector<size_t> &shape, const std::string &format) {
  using DeviceShapeTransfer = std::function<std::vector<size_t>(const std::vector<size_t> &)>;
  const std::map<std::string, DeviceShapeTransfer> device_shape_map{{kOpFormat_NCHW, NchwDeviceShape},
                                                                    {kOpFormat_NHWC, NhwcDeviceShape},
                                                                    {kOpFormat_HWCN, HwchDeviceShape},
                                                                    {kOpFormat_FRAC_Z, FracZDeviceShape},
                                                                    {kOpFormat_NC1HWC0, Nc1hwc0DeviceShape},
                                                                    {kOpFormat_C1HWNCoC0, C1hwncoc0DeviceShape},
                                                                    {kOpFormat_FRACTAL_Z_C04, FracZc04DeviceShape},
                                                                    {kOpFormat_NC1HWC0_C04, Nc1hwc04DeviceShape},
                                                                    {kOpFormat_NDHWC, NdhwcDeviceShape}};

  if (format == kOpFormat_ND || format == kOpFormat_DEFAULT) {
    return shape;
  }
  auto temp_shape = shape;
  std::vector<size_t> device_shape;
  if (format == kOpFormat_FRAC_NZ) {
    if (shape.size() == 1 && (shape[0] == 1 || shape[0] % kCubeSize == 0)) {
      // For [1] and [1024] shape we can trait it as NZ shape
      return shape;
    }
    if (shape.size() < 2) {
      MS_LOG(EXCEPTION) << "Format" << format << " is not support shape " << shape.size();
    } else {
      (void)std::copy(shape.begin(), shape.end() - 2, std::back_inserter(device_shape));
    }
    auto h1 = (shape[shape.size() - 2] - 1) / kCubeSize + 1;
    auto w1 = (shape[shape.size() - 1] - 1) / kCubeSize + 1;
    device_shape.push_back(w1);
    device_shape.push_back(h1);
    device_shape.push_back(kCubeSize);
    device_shape.push_back(kCubeSize);
    return device_shape;
  }
  if (shape.size() != kNchwDims) {
    MS_LOG(WARNING) << "Get Device Shape using a shape size is less than 4 ,should be Padding shape by Default firstly";
    temp_shape = PaddingShapeTo4dByDefault(shape);
  }
  auto iter = device_shape_map.find(format);
  if (iter == device_shape_map.end()) {
    MS_LOG(EXCEPTION) << "Unexpected format[" << format << "]";
  }
  return iter->second(temp_shape);
}

bool CheckArgs(const FormatArgs &args, size_t *size, size_t *total_size) {
  if (args.host_shape.size() != kNchwDims) {
    MS_LOG(ERROR) << "Invalid host shape, host shape dims:" << args.host_shape.size() << ", expect dims:" << kNchwDims;
    return false;
  }
  MS_EXCEPTION_IF_NULL(size);
  MS_EXCEPTION_IF_NULL(total_size);
  *size = TypeIdSize(args.src_data_type);
  if (*size < 1) {
    MS_LOG(ERROR) << "Illegal dtype.";
    return false;
  }
  *total_size = ShapeSize(args.device_shape) * (*size);
  if (*total_size != args.device_size) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << *total_size << ", device_size:" << args.device_size;
    return false;
  }
  return true;
}

bool TransDataType(const TypeIdArgs &args, void *result) {
  MS_LOG(DEBUG) << "Begin trans datatype from " << TypeIdLabel(args.host_data_type) << " to "
                << TypeIdLabel(args.device_data_type);
  MS_EXCEPTION_IF_NULL(result);
  std::pair<TypeId, TypeId> type_info(args.host_data_type, args.device_data_type);
  auto iter = mode_map.find(type_info);
  if (iter == mode_map.end()) {
    MS_LOG(ERROR) << "Unsupported datatype trans. src_type :" << TypeIdLabel(args.host_data_type)
                  << ", dst_type:" << TypeIdLabel(args.device_data_type);
    return false;
  }
  auto trans_mode = iter->second;
  if (!CastKernel(args, result, args.host_shape_size, trans_mode)) {
    MS_LOG(ERROR) << "Failed to trans datatype..";
    return false;
  }
  return true;
}

bool TransFormat(const FormatArgs &args, void *result) {
  using FormatTransfer = std::function<bool(const FormatArgs &, void *)>;
  const std::map<std::string, FormatTransfer> format_trans_map{
    {kOpFormat_FRAC_Z, NchwToFracZ},           {kOpFormat_FRAC_NZ, NchwToFracNz},
    {kOpFormat_NC1HWC0, NchwToNc1hwc0},        {kOpFormat_C1HWNCoC0, NchwToC1hwncoc0},
    {kOpFormat_FRACTAL_Z_C04, NchwToFracZc04}, {kOpFormat_NC1HWC0_C04, NchwToNc1hwc04}};
  MS_LOG(DEBUG) << "Start trans format.";
  if (TypeIdSize(args.src_data_type) < 1) {
    MS_LOG(ERROR) << "Invalid datatype..";
    return false;
  }
  if (args.device_format == kOpFormat_HWCN || args.device_format == kOpFormat_NHWC) {
    return NchwTo4D(args, result);
  }
  auto iter = format_trans_map.find(args.device_format);
  if (iter == format_trans_map.end()) {
    MS_LOG(EXCEPTION) << "Unexpected format[" << args.device_format << "]";
  }
  return iter->second(args, result);
}

bool TransFormatFromDeviceToHost(const FormatArgs &args, void *result) {
  using FormatTransfer = std::function<bool(const FormatArgs &, void *)>;
  const std::map<std::string, FormatTransfer> format_trans_map{{kOpFormat_FRAC_Z, FracZToNchw},
                                                               {kOpFormat_FRAC_NZ, FracNzToNchw},
                                                               {kOpFormat_NC1HWC0, Nc1hwc0ToNchw},
                                                               {kOpFormat_C1HWNCoC0, C1hwncoc0ToNchw},
                                                               {kOpFormat_NC1HWC0_C04, Nc1hwc04ToNchw}};
  MS_LOG(DEBUG) << "Start trans format.";
  if (TypeIdSize(args.src_data_type) < 1) {
    MS_LOG(ERROR) << "Invalid datatype..";
    return false;
  }
  if (args.device_format == kOpFormat_HWCN || args.device_format == kOpFormat_NHWC) {
    return ToNchw(args, result);
  }
  auto iter = format_trans_map.find(args.device_format);
  if (iter == format_trans_map.end()) {
    MS_LOG(EXCEPTION) << "Unexpected format[" << args.device_format << "]";
  }
  return iter->second(args, result);
}

bool NchwTo4D(const FormatArgs &args, void *result) {
  // trans nchw to 4d
  MS_LOG(DEBUG) << "Trans format from nchw to 4d.";
  MS_EXCEPTION_IF_NULL(result);
  size_t size = 0;
  size_t total_size = 0;
  if (!CheckArgs(args, &size, &total_size)) {
    MS_LOG(ERROR) << "Check args failed.";
    return false;
  }
  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];
  for (size_t ni = 0; ni < n; ni++) {
    for (size_t ci = 0; ci < c; ci++) {
      for (size_t hi = 0; hi < h; hi++) {
        for (size_t wi = 0; wi < w; wi++) {
          auto src_idx = ni * c * h * w + ci * h * w + hi * w + wi;
          auto dst_idx = 0;
          if (args.device_format == kOpFormat_NHWC) {
            dst_idx = ni * h * w * c + hi * w * c + wi * c + ci;
          } else if (args.device_format == kOpFormat_HWCN) {
            dst_idx = hi * w * c * n + wi * c * n + ci * n + ni;
          }
          SetData(size, false, src_idx, dst_idx, args, result);
        }
      }
    }
  }
  return true;
}

bool ToNchw(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format to nchw from 4d.";
  MS_EXCEPTION_IF_NULL(result);
  size_t size = 0;
  size_t total_size = 0;
  if (!CheckArgs(args, &size, &total_size)) {
    MS_LOG(ERROR) << "Check args failed.";
    return false;
  }
  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];
  for (size_t ni = 0; ni < n; ni++) {
    for (size_t ci = 0; ci < c; ci++) {
      for (size_t hi = 0; hi < h; hi++) {
        for (size_t wi = 0; wi < w; wi++) {
          auto dst_idx = ni * c * h * w + ci * h * w + hi * w + wi;
          auto src_idx = 0;
          if (args.device_format == kOpFormat_NHWC) {
            src_idx = ni * h * w * c + hi * w * c + wi * c + ci;
          } else if (args.device_format == kOpFormat_HWCN) {
            src_idx = hi * w * c * n + wi * c * n + ci * n + ni;
          }
          SetData(size, false, src_idx, dst_idx, args, result);
        }
      }
    }
  }
  return true;
}

bool NchwToFracZ(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from nchw to frac_z";
  MS_EXCEPTION_IF_NULL(result);
  if (args.host_shape.size() != kNchwDims) {
    MS_LOG(ERROR) << "Invalid host shape, host shape dims:" << args.host_shape.size() << ", expect dims:" << kNchwDims;
    return false;
  }
  auto size = TypeIdSize(args.src_data_type);
  if (size < 1) {
    MS_LOG(ERROR) << "Illegal dtype.";
    return false;
  }
  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];

  auto c0 = CubeSizeByType(args.src_data_type);
  if (c0 < 1) {
    MS_LOG(ERROR) << "Illegal dtype.";
    return false;
  }
  auto c1 = DivCeil(c, c0);
  auto hw = h * w;
  auto chw = c * hw;
  auto hwc0 = hw * c0;
  auto nchw = n * chw;

  auto hf_cnt = DivCeil(n, kCubeSize);
  auto vf_cnt = c1 * hw;
  auto fractal_ele_cnt = c0 * kCubeSize;
  auto total_ele_cnt = hf_cnt * vf_cnt * fractal_ele_cnt;
  auto dst_size = total_ele_cnt * size;
  if (dst_size != args.device_size) {
    MS_LOG(ERROR) << "Illegal total data size."
                  << "dst size is :" << dst_size << "device size is :" << args.device_size;
    return false;
  }

  for (size_t vfi = 0; vfi < vf_cnt; vfi++) {
    auto vf_base_i = vfi * hf_cnt;  // vertical fractal matrix base index
    for (size_t hfi = 0; hfi < hf_cnt; hfi++) {
      auto gfi = vf_base_i + hfi;  // global fractal matrix index
      auto src_n_offset = hfi * chw * kCubeSize;
      auto src_f_offset = src_n_offset + vfi % hw + vfi / hw * hwc0;
      for (size_t row = 0; row < c0; row++) {
        auto src_ci = vfi / hw * c0 + row;
        auto src_row_offset = src_f_offset + row * hw;
        for (size_t col = 0; col < kCubeSize; col++) {
          auto src_ni = hfi * kCubeSize + col;
          auto src_idx = src_row_offset + chw * col;
          auto dst_idx = gfi * fractal_ele_cnt + col * c0 + row;
          auto pad_zero = src_ni >= n || src_idx >= nchw || src_ci >= c;
          SetData(size, pad_zero, src_idx, dst_idx, args, result);
        }
      }
    }
  }
  return true;
}

bool FracZToNchw(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from frac_z to nchw";
  MS_EXCEPTION_IF_NULL(result);
  if (args.host_shape.size() != kNchwDims) {
    MS_LOG(ERROR) << "Invalid host shape, host shape dims:" << args.host_shape.size() << ", expect dims:" << kNchwDims;
    return false;
  }
  auto size = TypeIdSize(args.src_data_type);
  if (size < 1) {
    MS_LOG(ERROR) << "Illegal dtype.";
    return false;
  }
  auto total_size = ShapeSize(args.device_shape) * size;
  if (total_size != args.device_size) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << total_size << ", device_size:" << args.device_size;
    return false;
  }

  auto n0 = args.device_shape.at(1);
  auto ni = args.device_shape.at(2);
  auto c0 = args.device_shape.at(3);
  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];
  auto nc = ni * n0;
  auto ncc0 = nc * c0;
  auto wncc0 = w * ncc0;
  auto hwncc0 = h * wncc0;
  auto hw = h * w;
  auto chw = c * hw;

  for (size_t n_idx = 0; n_idx < n; n_idx++) {
    size_t n_head_addr = n_idx * chw;
    for (size_t c_idx = 0; c_idx < c; c_idx++) {
      size_t c_head_addr = n_head_addr + c_idx * hw;
      for (size_t h_idx = 0; h_idx < h; h_idx++) {
        size_t h_head_addr = c_head_addr + h_idx * w;
        for (size_t w_idx = 0; w_idx < w; w_idx++) {
          size_t dst_idx = h_head_addr + w_idx;
          size_t c1_idx = c_idx / c0;
          size_t c0_idx = c_idx % c0;
          size_t nc_idx = n_idx;
          size_t src_idx = c1_idx * hwncc0 + h_idx * wncc0 + w_idx * ncc0 + nc_idx * c0 + c0_idx;
          SetData(size, false, src_idx, dst_idx, args, result);
        }
      }
    }
  }
  return true;
}

bool NchwToFracZc04(const FormatArgs &args, void *result) {
  // trans nchw to FracZc04
  MS_LOG(DEBUG) << "Trans format from nchw to FracZc04.";
  MS_EXCEPTION_IF_NULL(result);
  size_t size = 0;
  size_t total_size = 0;
  if (!CheckArgs(args, &size, &total_size)) {
    MS_LOG(ERROR) << "Check args failed.";
    return false;
  }
  auto cube = kCubeSize;
  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];
  const size_t c0 = 4;
  auto c1 = DivCeil(c, c0);
  auto hwc0 = h * w * c0;
  auto hwc = h * w * c;
  auto nhwc = n * h * w * c;
  auto n_cnt = DivCeil(n, cube);
  auto v_cnt = DivCeil(h * w * c0 * c1, cube);
  size_t dst_idx = 0;

  for (size_t vi = 0; vi < v_cnt; vi++) {
    for (size_t ni = 0; ni < n_cnt; ni++) {
      for (size_t col = 0; col < cube; col++) {
        for (size_t row = 0; row < cube; row++) {
          size_t cur_cube_n = cube * ni + col;
          size_t cur_cube_c1hwc0 = cube * vi + row;
          auto desc_g = cur_cube_n / n;
          auto desc_n = cur_cube_n % n;
          auto desc_c1 = cur_cube_c1hwc0 / hwc0;
          auto desc_c0 = cur_cube_c1hwc0 % c0;
          auto desc_h = (cur_cube_c1hwc0 - hwc0 * desc_c1) / (w * c0);
          auto desc_w = (cur_cube_c1hwc0 - hwc0 * desc_c1 - w * c0 * desc_h) / c0;
          auto c_idx = desc_c1 * c0 + desc_c0;
          auto src_idx = desc_g * nhwc + desc_n * hwc + c_idx * h * w + desc_h * w + desc_w;
          auto pad_zero = desc_g >= 1 || desc_n >= n || c_idx >= c;
          SetData(size, pad_zero, src_idx, dst_idx, args, result);
          dst_idx++;
        }
      }
    }
  }
  return true;
}

bool NchwToNc1hwc04(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from nchw to Nc1hwc04.";
  return NchwToNc1hwc0(args, result);
}
bool Nc1hwc04ToNchw(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from Nc1hwc04 to nchw.";
  return Nc1hwc0ToNchw(args, result);
}

bool TransShapeToNz(const std::vector<size_t> &host_shape, std::vector<size_t> *hw_shape) {
  MS_EXCEPTION_IF_NULL(hw_shape);
  if (host_shape.empty()) {
    MS_LOG(ERROR) << "Size of vector is 0.";
    return false;
  }
  switch (host_shape.size()) {
    case 1:
      hw_shape->push_back(1);
      hw_shape->push_back(1);
      hw_shape->push_back(host_shape[0]);
      return true;
    default:
      auto size = host_shape.size();
      if (size < 2) {
        MS_LOG(ERROR) << "Illegal size.";
        return false;
      }
      size_t times = 1;
      for (size_t i = 0; i != size - 2; i++) {
        times *= host_shape[i];
      }
      hw_shape->push_back(times);
      hw_shape->push_back(host_shape[size - 2]);
      hw_shape->push_back(host_shape[size - 1]);
      return true;
  }
}

bool NchwToFracNz(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from nchw to frac_nz.";
  MS_EXCEPTION_IF_NULL(result);
  std::vector<size_t> hw_shape;
  if (!TransShapeToNz(args.host_shape, &hw_shape)) {
    MS_LOG(ERROR) << "Trans shape failed..";
    return false;
  }
  if (hw_shape.size() < 3 || args.device_shape.size() < 4) {
    MS_LOG(ERROR) << "Invalid shape size.";
    return false;
  }
  auto size = TypeIdSize(args.src_data_type);
  if (size < 1) {
    MS_LOG(ERROR) << "Illegal dtype";
    return false;
  }

  auto dst_size = ShapeSize(args.device_shape) * size;
  if (dst_size != args.device_size) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << dst_size << ", device_size:" << args.device_size;
    return false;
  }
  auto times = hw_shape.at(0);
  auto h = hw_shape.at(1);
  auto w = hw_shape.at(2);
  auto hw = h * w;

  auto shape_size = args.device_shape.size();
  auto w1 = args.device_shape[shape_size - 4];
  auto h1 = args.device_shape[shape_size - 3];
  auto h0 = args.device_shape[shape_size - 2];
  auto w0 = args.device_shape[shape_size - 1];
  auto h1h0w0 = h1 * h0 * w0;
  auto w1h1h0w0 = w1 * h1h0w0;
  auto num_w1 = w / w0;

  for (size_t times_idx = 0; times_idx < times; times_idx++) {
    auto times_head = times_idx * w1h1h0w0;
    auto src_times_head = times_idx * hw;
    for (size_t h1h0_idx = 0; h1h0_idx < h; h1h0_idx++) {
      auto h1h0_head = times_head + h1h0_idx * w0;
      auto src_h_head = src_times_head + h1h0_idx * w;
      for (size_t w1_idx = 0; w1_idx < num_w1; w1_idx++) {
        for (size_t i = 0; i < w0; ++i) {
          size_t src_idx = src_h_head + w1_idx * w0 + i;
          size_t dst_idx = h1h0_head + w1_idx * h1h0w0 + i;
          SetData(size, false, src_idx, dst_idx, args, result);
        }
      }
      auto w1_head = num_w1 * w0;
      for (size_t w0_idx = 0; w1_head + w0_idx < w; w0_idx++) {
        auto src_w_idx = w1_head + w0_idx;
        size_t dst_idx = h1h0_head + num_w1 * h1h0w0 + w0_idx;
        size_t src_idx = src_h_head + src_w_idx;
        SetData(size, false, src_idx, dst_idx, args, result);
      }
    }
  }
  return true;
}

bool FracNzToNchw(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from frac_nz to nchw";
  MS_EXCEPTION_IF_NULL(result);
  std::vector<size_t> hw_shape;
  if (!TransShapeToNz(args.host_shape, &hw_shape)) {
    MS_LOG(ERROR) << "Trans shape failed..";
    return false;
  }
  if (hw_shape.size() < 3 || args.device_shape.size() < 4) {
    MS_LOG(ERROR) << "Invalid shape size.";
    return false;
  }
  auto size = TypeIdSize(args.src_data_type);
  if (size < 1) {
    MS_LOG(ERROR) << "Illegal dtype";
    return false;
  }

  auto dst_size = ShapeSize(args.device_shape) * size;
  if (dst_size != args.device_size) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << dst_size << ", device_size:" << args.device_size;
    return false;
  }
  auto times = hw_shape.at(0);
  auto h = hw_shape.at(1);
  auto w = hw_shape.at(2);
  auto hw = h * w;

  auto shape_size = args.device_shape.size();
  auto w1 = args.device_shape[shape_size - 4];
  auto h1 = args.device_shape[shape_size - 3];
  auto h0 = args.device_shape[shape_size - 2];
  auto w0 = args.device_shape[shape_size - 1];
  auto h1h0w0 = h1 * h0 * w0;
  auto w1h1h0w0 = w1 * h1h0w0;
  auto num_w1 = w / w0;

  for (size_t times_idx = 0; times_idx < times; times_idx++) {
    auto times_head = times_idx * w1h1h0w0;
    auto src_times_head = times_idx * hw;
    for (size_t h1h0_idx = 0; h1h0_idx < h; h1h0_idx++) {
      auto h1h0_head = times_head + h1h0_idx * w0;
      auto src_h_head = src_times_head + h1h0_idx * w;
      for (size_t w1_idx = 0; w1_idx < num_w1; w1_idx++) {
        for (size_t i = 0; i < w0; ++i) {
          size_t src_idx = h1h0_head + w1_idx * h1h0w0 + i;
          size_t dst_idx = src_h_head + w1_idx * w0 + i;
          SetData(size, false, src_idx, dst_idx, args, result);
        }
      }
      auto w1_head = num_w1 * w0;
      for (size_t w0_idx = 0; w1_head + w0_idx < w; w0_idx++) {
        auto src_w_idx = w1_head + w0_idx;
        size_t src_idx = h1h0_head + num_w1 * h1h0w0 + w0_idx;
        size_t dst_idx = src_h_head + src_w_idx;
        SetData(size, false, src_idx, dst_idx, args, result);
      }
    }
  }
  return true;
}

bool NchwToNc1hwc0(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from nchw to Nc1h1wc0";
  MS_EXCEPTION_IF_NULL(result);
  if (args.host_shape.size() != kNchwDims) {
    MS_LOG(ERROR) << "Invalid host shape, host shape dims:" << args.host_shape.size() << ", expect dims:" << kNchwDims;
    return false;
  }
  auto size = TypeIdSize(args.src_data_type);
  if (size < 1) {
    MS_LOG(ERROR) << "Illegal dtype.";
    return false;
  }
  auto total_size = ShapeSize(args.device_shape) * size;
  if (total_size != args.device_size) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << total_size << ", device_size:" << args.device_size;
    return false;
  }

  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];
  auto c0 = CubeSizeByType(args.src_data_type);
  if (c0 < 1) {
    MS_LOG(ERROR) << "Illegal dtype.";
    return false;
  }
  if (args.device_format == kOpFormat_NC1HWC0_C04) {
    c0 = 4;
  }
  auto c1 = DivCeil(c, c0);
  auto hw = h * w;
  auto chw = c * hw;
  auto c1hwc0 = c1 * hw * c0;
  auto wc0 = w * c0;

  for (size_t n_idx = 0; n_idx < n; n_idx++) {
    size_t n_head_addr = n_idx * c1hwc0;
    for (size_t c1_idx = 0; c1_idx < c1; c1_idx++) {
      size_t c1_head_addr = n_head_addr + c1_idx * hw * c0;
      for (size_t h_idx = 0; h_idx < h; h_idx++) {
        size_t h_head_addr = c1_head_addr + h_idx * wc0;
        for (size_t w_idx = 0; w_idx < w; w_idx++) {
          size_t w_head_addr = h_head_addr + w_idx * c0;
          for (size_t c0_idx = 0; c0_idx < c0; c0_idx++) {
            size_t dst_idx = c0_idx + w_head_addr;
            size_t c_idx = c0_idx + c1_idx * c0;
            size_t src_idx = n_idx * chw + c_idx * hw + h_idx * w + w_idx;
            auto pad_zero = c_idx >= c;
            SetData(size, pad_zero, src_idx, dst_idx, args, result);
          }
        }
      }
    }
  }
  return true;
}

bool Nc1hwc0ToNchw(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from nc1h1wc0 to nchw";
  MS_EXCEPTION_IF_NULL(result);
  if (args.host_shape.size() != kNchwDims) {
    MS_LOG(ERROR) << "Invalid host shape, host shape dims:" << args.host_shape.size() << ", expect dims:" << kNchwDims;
    return false;
  }
  auto size = TypeIdSize(args.src_data_type);
  if (size < 1) {
    MS_LOG(ERROR) << "Illegal dtype.";
    return false;
  }
  auto total_size = ShapeSize(args.device_shape) * size;
  if (total_size != args.device_size) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << total_size << ", device_size:" << args.device_size;
    return false;
  }

  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];
  auto c1 = args.device_shape[1];
  auto c0 = args.device_shape[4];

  auto hw = h * w;
  auto chw = c * hw;
  auto wc0 = w * c0;
  auto hwc0 = h * wc0;
  auto c1hwc0 = c1 * hwc0;

  for (size_t n_idx = 0; n_idx < n; n_idx++) {
    size_t n_head_addr = n_idx * chw;
    for (size_t c_idx = 0; c_idx < c; c_idx++) {
      size_t c_head_addr = n_head_addr + c_idx * hw;
      for (size_t h_idx = 0; h_idx < h; h_idx++) {
        size_t h_head_addr = c_head_addr + h_idx * w;
        for (size_t w_idx = 0; w_idx < w; w_idx++) {
          size_t dst_idx = h_head_addr + w_idx;
          size_t c1_idx = c_idx / c0;
          size_t c0_idx = c_idx % c0;
          size_t src_idx = n_idx * c1hwc0 + c1_idx * hwc0 + h_idx * wc0 + w_idx * c0 + c0_idx;
          SetData(size, false, src_idx, dst_idx, args, result);
        }
      }
    }
  }
  return true;
}

bool NchwToC1hwncoc0(const FormatArgs &args, void *result) {
  // trans nchw to c1hwncoc0
  MS_LOG(DEBUG) << "Trans format from nchw to c1hwncoc0.";
  MS_EXCEPTION_IF_NULL(result);
  size_t size = 0;
  size_t total_size = 0;
  if (!CheckArgs(args, &size, &total_size)) {
    MS_LOG(ERROR) << "Check args failed.";
    return false;
  }
  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];
  const int co_idx = 4;
  const int c0_idx = 5;
  auto c1 = args.device_shape[0];
  auto co = args.device_shape[co_idx];
  auto c0 = args.device_shape[c0_idx];

  for (size_t c1_i = 0; c1_i < c1; c1_i++) {
    for (size_t h_i = 0; h_i < h; h_i++) {
      for (size_t w_i = 0; w_i < w; w_i++) {
        for (size_t n_i = 0; n_i < n; n_i++) {
          for (size_t co_i = 0; co_i < co; co_i++) {
            for (size_t c0_i = 0; c0_i < c0; c0_i++) {
              size_t dst_idx = c1_i * h * w * n * co * c0 + h_i * w * n * co * c0 + w_i * n * co * c0 + n_i * co * c0 +
                               co_i * c0 + c0_i;
              size_t c_i = c0_i + c1_i * c0;
              size_t src_idx = n_i * c * h * w + c_i * h * w + h_i * w + w_i;
              auto pad_zero = !(c_i < c && c0_i == co_i);
              SetData(size, pad_zero, src_idx, dst_idx, args, result);
            }
          }
        }
      }
    }
  }
  return true;
}

bool C1hwncoc0ToNchw(const FormatArgs &args, void *result) {
  // trans c1hwncoc0 to nchw
  MS_LOG(DEBUG) << "Trans format from c1hwncoc0 to nchw";
  MS_EXCEPTION_IF_NULL(result);
  size_t size = 0;
  size_t total_size = 0;
  if (!CheckArgs(args, &size, &total_size)) {
    MS_LOG(ERROR) << "Check args failed.";
    return false;
  }
  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];
  const int co_idx = 4;
  const int c0_idx = 5;
  auto co = args.device_shape[co_idx];
  auto c0 = args.device_shape[c0_idx];
  for (size_t n_i = 0; n_i < n; n_i++) {
    for (size_t c_i = 0; c_i < c; c_i++) {
      for (size_t h_i = 0; h_i < h; h_i++) {
        for (size_t w_i = 0; w_i < w; w_i++) {
          size_t dst_idx = n_i * c * h * w + c_i * h * w + h_i * w + w_i;
          size_t c1_i = c_i / kCubeSize;
          size_t c0_i = c_i % kCubeSize;
          size_t co_i = c0_i;
          size_t src_idx =
            c1_i * h * w * n * co * c0 + h_i * w * n * co * c0 + w_i * n * co * c0 + n_i * co * c0 + co_i * c0 + c0_i;
          SetData(size, false, src_idx, dst_idx, args, result);
        }
      }
    }
  }
  return true;
}
}  // namespace trans
}  // namespace mindspore
