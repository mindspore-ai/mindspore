/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "runtime/device/ms_device_shape_transfer.h"
#include <functional>
#include <numeric>
#include <utility>
#include <algorithm>

namespace mindspore {
namespace trans {
static const ShapeValueDType kShapeDimAny = abstract::Shape::kShapeDimAny;

const int b1 = 1;
const int b2 = 2;
const int b4 = 4;
const int b8 = 8;
const int64_t kCubeSize = 16;
const int64_t kCube16 = kCubeSize;
const int64_t kCube32 = 32;
const int64_t kCube64 = 64;
const int64_t kCubeSize_C04 = 4;
const int64_t kNiSize = 16;
constexpr int kDims2 = 2;
constexpr int64_t k4 = 4;
static const std::set<TypeId> C0_64 = {kNumberTypeInt4};
static const std::set<TypeId> C0_32 = {kNumberTypeUInt8, kNumberTypeInt8};
namespace {
const size_t hw_h = 1;
const size_t hw_w = 2;
const size_t fnz_w1 = 4;
const size_t fnz_h1 = 3;
const size_t fnz_h0 = 2;
const size_t fnz_w0 = 1;
const size_t fz_n0 = 1;
const size_t fz_ni = 2;
const size_t fz_c0 = 3;
bool HasShapeDynamic(const ShapeVector &shape_list) {
  return std::any_of(shape_list.begin(), shape_list.end(), [](int64_t v) { return v == kShapeDimAny; });
}

inline int64_t CalMaxShape(int64_t ori_val, int64_t new_val) {
  if (ori_val < 0) {
    return kShapeDimAny;
  }

  return new_val;
}

template <typename T>
T Gcd(T a, T b) {
  if (b == 0) {
    return 0;
  }
  T c = b;
  while (a % b != 0) {
    c = a % b;
    a = b;
    b = c;
  }
  return c;
}

template <typename T>
T Lcm(T a, T b) {
  if (b == 0) {
    return 0;
  }
  T ret = (a * b) / (Gcd(a, b));
  return ret;
}

template <typename T>
T DivCeil(T n1, T n2) {
  if (n2 != 0) {
    return (n1 + n2 - 1) / n2;
  }
  return 0;
}

template <typename T>
bool CheckDims(const std::vector<T> &shape) {
  if (shape.size() != kDim4) {
    MS_LOG(ERROR) << "Host shape dims should be 4";
    return false;
  }
  return true;
}

int64_t GetCubeSizeByType(const TypeId &data_type) {
  if (C0_32.find(data_type) != C0_32.end()) {
    return kCube32;
  }
  if (C0_64.find(data_type) != C0_64.end()) {
    return kCube64;
  }
  return kCube16;
}

RangePair PaddingRangeTo5dDefault(const RangePair &ori_range) {
  RangePair dst_range(kNcdhw, std::pair<int64_t, int64_t>(1, 1));
  switch (ori_range.size()) {
    case N_ncdhw:
      return ori_range;
    case C_ncdhw:
      dst_range[C_ncdhw] = ori_range[N_ncdhw];
      break;
    case D_ncdhw:
      dst_range[C_ncdhw] = ori_range[N_ncdhw];
      dst_range[D_ncdhw] = ori_range[C_ncdhw];
      break;
    case H_ncdhw:
      dst_range[C_ncdhw] = ori_range[N_ncdhw];
      dst_range[D_ncdhw] = ori_range[C_ncdhw];
      dst_range[H_ncdhw] = ori_range[D_ncdhw];
      break;
    case W_ncdhw:
      dst_range[C_ncdhw] = ori_range[N_ncdhw];
      dst_range[D_ncdhw] = ori_range[C_ncdhw];
      dst_range[H_ncdhw] = ori_range[D_ncdhw];
      dst_range[W_ncdhw] = ori_range[H_ncdhw];
      break;
    default:
      MS_LOG(EXCEPTION) << "Unexpected shape size: " << ori_range.size();
  }
  return dst_range;
}

RangePair PaddingRangeTo5D(const RangePair &ori_range, const std::string &padding_str = {""}) {
  std::vector<Axis5D> padding_axis;
  StringToAxisVector5D(padding_str, &padding_axis);
  if (padding_axis.empty() || ori_range.size() > padding_axis.size()) {
    return PaddingRangeTo5dDefault(ori_range);
  }

  RangePair dst_range(kNcdhw, std::pair<int64_t, int64_t>(1, 1));
  for (size_t index = 0; index < ori_range.size(); index++) {
    dst_range[padding_axis[index]] = ori_range[index];
  }
  return dst_range;
}

RangePair PaddingRangeTo4dDefault(const RangePair &ori_range) {
  RangePair dst_range(kNchwDims, std::pair<int64_t, int64_t>(1, 1));
  switch (ori_range.size()) {
    case kN:
      return dst_range;
    case kC:
      dst_range[kC] = ori_range[kN];
      break;
    case kH:
      dst_range[kC] = ori_range[kN];
      dst_range[kH] = ori_range[kC];
      break;
    case kW:
      dst_range[kC] = ori_range[kN];
      dst_range[kH] = ori_range[kC];
      dst_range[kW] = ori_range[kH];
      break;
    case kNchwDims:
      return ori_range;
    default:
      MS_LOG(EXCEPTION) << "Unexpected range size: " << ori_range.size();
  }
  return dst_range;
}

RangePair PaddingRangeTo4D(const RangePair &ori_range, const std::string &padding_str = {""}) {
  std::vector<Axis> padding_axis;
  StringToAxisVector4D(padding_str, &padding_axis);
  if (padding_axis.empty() || ori_range.size() > padding_axis.size()) {
    return PaddingRangeTo4dDefault(ori_range);
  }

  RangePair dst_range(kNchwDims, std::pair<int64_t, int64_t>(1, 1));
  for (size_t index = 0; index < ori_range.size(); index++) {
    dst_range[padding_axis[index]] = ori_range[index];
  }
  return dst_range;
}
}  // namespace

void StringToAxisVector4D(const std::string &reshape_type_str, std::vector<Axis> *reshape_type_vec) {
  MS_EXCEPTION_IF_NULL(reshape_type_vec);
  if (reshape_type_str.empty()) {
    MS_LOG(DEBUG) << "Reshape type str is empty, no need padding.";
    return;
  }
  for (const auto &c : reshape_type_str) {
    switch (c) {
      case 'N':
        reshape_type_vec->push_back(N);
        break;
      case 'C':
        reshape_type_vec->push_back(C);
        break;
      case 'H':
        reshape_type_vec->push_back(H);
        break;
      case 'W':
        reshape_type_vec->push_back(W);
        break;
      default:
        MS_LOG(EXCEPTION) << "Unknown axis " << c << "in reshape type.";
    }
  }
}

void StringToAxisVector5D(const std::string &reshape_type_str, std::vector<Axis5D> *reshape_type_vec) {
  MS_EXCEPTION_IF_NULL(reshape_type_vec);
  if (reshape_type_str.empty()) {
    MS_LOG(DEBUG) << "Reshape type str is empty, no need padding.";
    return;
  }
  for (const auto &c : reshape_type_str) {
    switch (c) {
      case 'N':
        reshape_type_vec->push_back(N_ncdhw);
        break;
      case 'C':
        reshape_type_vec->push_back(C_ncdhw);
        break;
      case 'D':
        reshape_type_vec->push_back(D_ncdhw);
        break;
      case 'H':
        reshape_type_vec->push_back(H_ncdhw);
        break;
      case 'W':
        reshape_type_vec->push_back(W_ncdhw);
        break;
      default:
        MS_LOG(EXCEPTION) << "Unknown axis " << c << "in reshape type.";
    }
  }
}

bool IsNeedPadding(const std::string &format, const ShapeVector &shape) {
  if (shape.size() == 0) {
    return false;
  }
  if (IsDynamicRank(shape) && !IsOneOfDynRankNeedPadShape(format)) {
    return false;
  }
  if (format == kOpFormat_DEFAULT || format == kOpFormat_NCHW || IsOneOfNoPaddingFormat(format)) {
    return false;
  } else if (shape.size() < kDim4) {
    return true;
  }
  return false;
}

ShapeVector GetRuntimePaddingShape(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  ShapeVector host_shape;
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    // Scalar has no shape.
    if (node_value->isa<Scalar>()) {
      return {};
    }
    if (node_value->isa<ValueSequence>()) {
      MS_LOG(INFO) << "GetRuntimePaddingShape does not support the value sequence for value node:"
                   << node->fullname_with_scope() << ", debug name:" << node->DebugString();
      return {0};
    }
    auto tensor = node_value->cast<tensor::TensorPtr>();
    if (tensor == nullptr) {
      MS_LOG(EXCEPTION) << " The node[ " << node->DebugString() << "]'s cannot convert ";
    }
    host_shape = tensor->shape();
    if (host_shape.empty()) {
      host_shape.push_back(1);
    }
  } else {
    host_shape = common::AnfAlgo::GetOutputInferShape(node, index);
  }
  auto format = AnfAlgo::GetOutputFormat(node, index);
  if (IsNeedPadding(format, host_shape)) {
    host_shape = PaddingShape(host_shape, format, AnfAlgo::GetOutputReshapeType(node, index), node);
  }
  return host_shape;
}

bool TransDataType(const TypeIdArgs &args, void *result) {
  DataTypeTransfer dataTypeTransfer;
  return dataTypeTransfer.TransDataType(args, result);
}

bool TransFormat(const FormatArgs &args, void *result, const AnfNodePtr &node, size_t index) {
  FormatTransfer formatTransfer;
  return formatTransfer.TransDataByFormat(args, result, node, index, true);
}

bool TransFormatFromDeviceToHost(const FormatArgs &args, void *result, int64_t groups) {
  FormatTransfer formatTransfer;
  return formatTransfer.TransDataBackwordCore(args, result, groups);
}

bool TransFormatFromDeviceToHost(const FormatArgs &args, void *result, const AnfNodePtr &node, size_t index) {
  FormatTransfer formatTransfer;
  return formatTransfer.TransDataByFormat(args, result, node, index, false);
}

/**###################### DATA TYPE TRANS ################################*/
void CheckMemSize(const TypeIdArgs &args) {
  auto src_type_size = abstract::TypeIdSize(args.src_data_type);
  auto dst_type_size = abstract::TypeIdSize(args.dst_data_type);
  if (src_type_size < 1 || dst_type_size < 1) {
    MS_LOG(EXCEPTION) << "Invalid src or dst data type. Src type: " << TypeIdLabel(args.src_data_type)
                      << ", dst type: " << TypeIdLabel(args.dst_data_type);
  }
  if (SizeToLong(args.data_size / src_type_size) != args.src_shape_size) {
    MS_LOG(EXCEPTION) << "Invalid src or dst data  shape size. Src shape size: " << args.src_shape_size
                      << ", dst shape size: " << args.data_size / src_type_size;
  }
}

template <typename SrcT, typename DstT>
void TransDataSrc2Dst(const TypeIdArgs &args, void *dst, const int64_t data_size) {
  CheckMemSize(args);
  for (int64_t idx = 0; idx != data_size; idx++) {
    SrcT src_data = static_cast<const SrcT *>(args.data)[idx];
    static_cast<DstT *>(dst)[idx] = static_cast<DstT>(src_data);
  }
}
template <typename SrcT>
void TransDataSrc2Fp16(const TypeIdArgs &args, void *dst, const int64_t data_size) {
  CheckMemSize(args);
  auto src_data = static_cast<const SrcT *>(args.data);
  auto half_data = static_cast<float16 *>(dst);
  for (int64_t i = 0; i < data_size; i++) {
    half_data[i] = float16(src_data[i]);
  }
}

bool DataTypeTransfer::CastKernel(const TypeIdArgs &args, void *dst, int64_t data_size, DataTypeTransMode mode) const {
  using DtypeKernel = std::function<void(const TypeIdArgs &, void *, const int64_t)>;
  const std::map<DataTypeTransMode, DtypeKernel> cast_kernel_map{
    {DataTypeTransMode::FROM_BOOL_TO_UINT8, TransDataSrc2Dst<int8_t, uint8_t>},
    {DataTypeTransMode::FROM_BOOL_TO_INT32, TransDataSrc2Dst<int8_t, int32_t>},
    {DataTypeTransMode::FROM_BOOL_TO_FLOAT16, TransDataSrc2Fp16<int8_t>},
    {DataTypeTransMode::FROM_BOOL_TO_FLOAT, TransDataSrc2Dst<int8_t, float>},
    {DataTypeTransMode::FROM_INT8_TO_INT32, TransDataSrc2Dst<int8_t, int32_t>},
    {DataTypeTransMode::FROM_INT8_TO_FLOAT16, TransDataSrc2Fp16<int8_t>},
    {DataTypeTransMode::FROM_INT8_TO_FLOAT, TransDataSrc2Dst<int8_t, float>},
    {DataTypeTransMode::FROM_UINT8_TO_INT32, TransDataSrc2Dst<uint8_t, int32_t>},
    {DataTypeTransMode::FROM_UINT8_TO_FLOAT16, TransDataSrc2Fp16<uint8_t>},
    {DataTypeTransMode::FROM_UINT8_TO_FLOAT, TransDataSrc2Dst<uint8_t, float>},
    {DataTypeTransMode::FROM_UINT16_TO_INT32, TransDataSrc2Dst<uint16_t, int32_t>},
    {DataTypeTransMode::FROM_INT32_TO_BOOL, TransDataSrc2Dst<int32_t, int8_t>},
    {DataTypeTransMode::FROM_INT32_TO_INT8, TransDataSrc2Dst<int32_t, int8_t>},
    {DataTypeTransMode::FROM_INT32_TO_UINT8, TransDataSrc2Dst<int32_t, uint8_t>},
    {DataTypeTransMode::FROM_INT32_TO_FLOAT16, TransDataSrc2Fp16<int32_t>},
    {DataTypeTransMode::FROM_INT32_TO_FLOAT, TransDataSrc2Dst<int32_t, float>},
    {DataTypeTransMode::FROM_INT32_TO_INT64, TransDataSrc2Dst<int32_t, int64_t>},
    {DataTypeTransMode::FROM_INT64_TO_INT32, TransDataSrc2Dst<int64_t, int32_t>},
    {DataTypeTransMode::FROM_FLOAT16_TO_UINT8, TransDataSrc2Dst<float16, uint8_t>},
    {DataTypeTransMode::FROM_FLOAT16_TO_INT32, TransDataSrc2Dst<float16, int32_t>},
    {DataTypeTransMode::FROM_FLOAT_TO_INT32, TransDataSrc2Dst<float, int32_t>},
    {DataTypeTransMode::FROM_FLOAT32_TO_FLOAT64, TransDataSrc2Dst<float, double>},
    {DataTypeTransMode::FROM_FLOAT64_TO_FLOAT32, TransDataSrc2Dst<double, float>}};

  if (mode == DataTypeTransMode::FROM_FLOAT_TO_FLOAT16) {
    device::FloatToHalf(dst, args.data, LongToSize(data_size));
    return true;
  } else if (mode == DataTypeTransMode::FROM_FLOAT16_TO_FLOAT) {
    device::HalfToFloat(dst, args.data, LongToSize(data_size));
    return true;
  }
  auto iter = cast_kernel_map.find(mode);
  if (iter != cast_kernel_map.end()) {
    iter->second(args, dst, data_size);
    return true;
  } else {
    MS_LOG(ERROR) << "Can not find a datatype trans function. Src type :" << TypeIdLabel(args.src_data_type)
                  << ", dst_type:" << TypeIdLabel(args.dst_data_type);
    return false;
  }
}

bool DataTypeTransfer::TransDataType(const TypeIdArgs &args, void *result) const {
  MS_LOG(DEBUG) << "Begin trans datatype from " << TypeIdLabel(args.src_data_type) << " to "
                << TypeIdLabel(args.dst_data_type);
  MS_EXCEPTION_IF_NULL(result);
  std::pair<TypeId, TypeId> type_info(args.src_data_type, args.dst_data_type);
  auto iter = mode_map.find(type_info);
  if (iter == mode_map.end()) {
    MS_LOG(ERROR) << "Can not find a datatype trans type. src_type :" << TypeIdLabel(args.src_data_type)
                  << ", dst_type:" << TypeIdLabel(args.dst_data_type);
    return false;
  }
  auto trans_mode = iter->second;
  if (!CastKernel(args, result, args.src_shape_size, trans_mode)) {
    MS_LOG(ERROR) << "Failed to trans datatype. Src: " << TypeIdLabel(args.src_data_type)
                  << ", dst: " << TypeIdLabel(args.dst_data_type);
    return false;
  }
  return true;
}

/**###################### DATA SHAPE TRANS ################################*/
ShapeVector DeviceShapeTransfer::GetDeviceShapeByFormat(const ShapeVector &shape, const std::string &format,
                                                        const AnfNodePtr &node, size_t index, const TypeId &type,
                                                        bool is_output) const {
  auto dev_shape = GetFixedDeviceShape(shape, node, index, is_output);
  if (dev_shape.has_value()) {
    return dev_shape.value();
  }
  int64_t groups = 1;
  if (format == kOpFormat_FRAC_Z) {
    groups = common::AnfAlgo::GetAttrGroups(node, index);
  }
  ShapeVector input_hidden_size = {kAlign16, kAlign16};
  if (format == kOpFormat_FRACTAL_ZN_RNN || format == kOpFormat_ND_RNN_BIAS) {
    input_hidden_size = GetAttrInputAndHiddenSize(node);
  }
  if (node != nullptr) {
    MS_LOG(DEBUG) << "Start trans infer shape to device shape for node: " << node->DebugString()
                  << ", format: " << format;
  }
  return TransCore(shape, format, type, groups, input_hidden_size);
}

ShapeVector DeviceShapeTransfer::GetDeviceShapeByFormat(const ShapeVector &shape, const std::string &format,
                                                        const TypeId &type, int64_t groups,
                                                        const ShapeVector &input_hidden_size) const {
  return TransCore(shape, format, type, groups, input_hidden_size);
}

std::optional<ShapeVector> DeviceShapeTransfer::GetFixedDeviceShape(const ShapeVector &, const AnfNodePtr &node,
                                                                    size_t index, bool is_output) const {
  if (node == nullptr || !node->isa<CNode>()) {
    return {};
  }
  auto attr_name = is_output ? kAttrFixedOutputDeviceShape : kAttrFixedInputDeviceShape;
  auto cnode = node->cast<CNodePtr>();
  if (!common::AnfAlgo::HasNodeAttr(attr_name, cnode)) {
    return {};
  }

  auto shapes = common::AnfAlgo::GetNodeAttr<std::vector<ShapeVector>>(cnode, attr_name);
  if (index >= shapes.size()) {
    MS_LOG(INFO) << "Index is out of range, got index: " << index << ", shape size: " << shapes.size();
    return {};
  }
  return std::optional<ShapeVector>(std::move(shapes[index]));
}

ShapeVector DeviceShapeTransfer::TransCore(const ShapeVector &shape, const std::string &format, const TypeId &type,
                                           int64_t groups, const ShapeVector &input_hidden_size) const {
  using DeviceShapeTransferFunc = std::function<ShapeVector(const ShapeVector &, const TypeId &)>;
  static const mindspore::HashMap<std::string, DeviceShapeTransferFunc> device_shape_map = {
    {kOpFormat_NCHW, NCHWDeviceShape},
    {kOpFormat_NHWC, NHWCDeviceShape},
    {kOpFormat_HWCN, HWCNDeviceShape},
    {kOpFormat_NCDHW, NCDHWDeviceShape},
    {kOpFormat_FRAC_Z, FRAC_ZDeviceShape},
    {kOpFormat_FRAC_NZ, FRAC_NZDeviceShape},
    {kOpFormat_NC1HWC0, NC1HWC0DeviceShape},
    {kOpFormat_NDC1HWC0, NDC1HWC0DeviceShape},
    {kOpFormat_C1HWNCoC0, C1HWNCOC0DeviceShape},
    {kOpFormat_NC1HWC0_C04, NC1HWC04DeviceShape},
    {kOpFormat_FRACTAL_Z_3D, FRAC_Z3DDeviceShape},
    {kOpFormat_FRACTAL_Z_C04, FRAC_ZC04DeviceShape},
    {kOpFormat_ChannelLast, ChannelLastDeviceShape},
    {kOpFormat_FRACTAL_ZN_LSTM, FRAC_ZN_LSTMDeviceShape}};
  if (format == kOpFormat_ND || format == kOpFormat_DEFAULT || format == kOpFormat_NCHW) {
    return shape;
  }
  if (groups > 1 && format == kOpFormat_FRAC_Z) {
    return FRAC_ZDeviceShapeWithGroups(shape, type, groups);
  }
  if (format == kOpFormat_FRACTAL_ZN_RNN) {
    return FRAC_ZN_RNNDeviceShape(shape, type, input_hidden_size);
  }
  if (format == kOpFormat_ND_RNN_BIAS) {
    return NDRNNBiasDeviceShape(shape, type, input_hidden_size[1]);
  }
  auto temp_shape = shape;
  if (!IsOneOfNoPaddingFormat(format) && format != kOpFormat_FRACTAL_ZN_LSTM && shape.size() < kDim4 &&
      !IsOneOf3DFormat(format)) {
    MS_LOG(INFO) << "Origin shape size is less than 4, should be Padding shape by Default firstly";
    temp_shape = PaddingShapeTo4dDefault(shape);
  }
  if (shape.size() != kDim5 && IsOneOf3DFormat(format)) {
    temp_shape = PaddingShapeTo5dDefault(shape);
  }
  auto iter = device_shape_map.find(format);
  if (iter == device_shape_map.end()) {
    MS_LOG(EXCEPTION) << "Unexpected format[" << format << "]";
  }
  return iter->second(temp_shape, type);
}

ShapeVector DeviceShapeTransfer::NCHWDeviceShape(const ShapeVector &shape, const TypeId &) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  return shape;
}

ShapeVector DeviceShapeTransfer::NHWCDeviceShape(const ShapeVector &shape, const TypeId &) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  ShapeVector device_shape;
  device_shape.push_back(shape[kN]);
  device_shape.push_back(shape[kH]);
  device_shape.push_back(shape[kW]);
  device_shape.push_back(shape[kC]);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::HWCNDeviceShape(const ShapeVector &shape, const TypeId &) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  ShapeVector device_shape;
  device_shape.push_back(shape[kH]);
  device_shape.push_back(shape[kW]);
  device_shape.push_back(shape[kC]);
  device_shape.push_back(shape[kN]);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::FRAC_ZDeviceShape(const ShapeVector &shape, const TypeId &type) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  ShapeVector device_shape;
  auto c0 = GetCubeSizeByType(type);
  if (HasShapeDynamic({shape[kC], shape[kH], shape[kW]})) {
    device_shape.push_back(abstract::Shape::kShapeDimAny);
  } else {
    auto c1 = (shape[kC] + c0 - 1) / c0;
    device_shape.push_back(shape[kH] * shape[kW] * c1);
  }
  if (shape[kN] == abstract::Shape::kShapeDimAny) {
    device_shape.push_back(abstract::Shape::kShapeDimAny);
  } else {
    auto no = (shape[kN] + kNiSize - 1) / kNiSize;
    device_shape.push_back(no);
  }
  device_shape.push_back(kNiSize);
  device_shape.push_back(c0);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::NC1HWC0DeviceShape(const ShapeVector &shape, const TypeId &type) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  ShapeVector device_shape;
  auto c0 = GetCubeSizeByType(type);
  auto c1 = (shape[kC] == abstract::Shape::kShapeDimAny) ? abstract::Shape::kShapeDimAny : (shape[kC] + c0 - 1) / c0;
  device_shape.push_back(shape[kN]);
  device_shape.push_back(c1);
  device_shape.push_back(shape[kH]);
  device_shape.push_back(shape[kW]);
  device_shape.push_back(c0);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::NDC1HWC0DeviceShape(const ShapeVector &shape, const TypeId &type) {
  if (shape.size() == kDim6) {
    return shape;
  }
  if (shape.size() != kDim5) {
    MS_LOG(EXCEPTION) << "Check dims failed, expect shape dim 5, but got shape dim : " << shape.size();
  }
  ShapeVector device_shape;
  auto c0 = GetCubeSizeByType(type);
  auto c1 = (shape[1] == abstract::Shape::kShapeDimAny) ? abstract::Shape::kShapeDimAny : (shape[1] + c0 - 1) / c0;
  device_shape.push_back(shape[N_ncdhw]);
  device_shape.push_back(shape[D_ncdhw]);
  device_shape.push_back(c1);
  device_shape.push_back(shape[H_ncdhw]);
  device_shape.push_back(shape[W_ncdhw]);
  device_shape.push_back(c0);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::FRAC_Z3DDeviceShape(const ShapeVector &shape, const TypeId &type) {
  if (shape.size() != kDim5) {
    MS_LOG(EXCEPTION) << "Check dims failed, expect shape dim 5, but got shape dim : " << shape.size();
  }
  ShapeVector device_shape;
  auto c0 = GetCubeSizeByType(type);
  if (HasShapeDynamic({shape[C_ncdhw], shape[D_ncdhw], shape[H_ncdhw], shape[W_ncdhw]})) {
    device_shape.push_back(abstract::Shape::kShapeDimAny);
  } else {
    auto c1 = (shape[1] + c0 - 1) / c0;
    device_shape.push_back(shape[D_ncdhw] * c1 * shape[H_ncdhw] * shape[W_ncdhw]);
  }
  auto no =
    (shape[0] == abstract::Shape::kShapeDimAny) ? abstract::Shape::kShapeDimAny : (shape[0] + kNiSize - 1) / kNiSize;
  device_shape.push_back(no);
  device_shape.push_back(kNiSize);
  device_shape.push_back(c0);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::C1HWNCOC0DeviceShape(const ShapeVector &shape, const TypeId &type) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  ShapeVector device_shape;
  auto c0 = GetCubeSizeByType(type);
  if (shape[kC] == abstract::Shape::kShapeDimAny) {
    device_shape.push_back(abstract::Shape::kShapeDimAny);
  } else {
    device_shape.push_back((shape[kC] - 1) / c0 + 1);
  }
  device_shape.push_back(shape[kH]);
  device_shape.push_back(shape[kW]);
  device_shape.push_back(shape[kN]);
  device_shape.push_back(c0);
  device_shape.push_back(c0);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::FRAC_ZC04DeviceShape(const ShapeVector &shape, const TypeId &) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  ShapeVector device_shape;
  const int64_t C04 = 4;
  int64_t first_dim;
  if (HasShapeDynamic({shape[kH], shape[kW]})) {
    first_dim = abstract::Shape::kShapeDimAny;
  } else {
    first_dim = DivCeil(C04 * shape[kH] * shape[kW], kCubeSize);
  }
  auto no =
    (shape[kN] == abstract::Shape::kShapeDimAny) ? abstract::Shape::kShapeDimAny : DivCeil(shape.at(kN), kCubeSize);
  device_shape.push_back(first_dim);
  device_shape.push_back(no);
  device_shape.push_back(kCubeSize);
  device_shape.push_back(kCubeSize);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::NC1HWC04DeviceShape(const ShapeVector &shape, const TypeId &) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  ShapeVector device_shape;
  const int64_t C04 = 4;
  const int64_t C1 =
    (shape[kC] == abstract::Shape::kShapeDimAny) ? abstract::Shape::kShapeDimAny : DivCeil(shape.at(kC), C04);
  device_shape.push_back(shape[kN]);
  device_shape.push_back(C1);
  device_shape.push_back(shape[kH]);
  device_shape.push_back(shape[kW]);
  device_shape.push_back(C04);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::NCDHWDeviceShape(const ShapeVector &shape, const TypeId &) {
  if (shape.size() < kDim5) {
    MS_LOG(EXCEPTION) << "Shape dims must be 5 when format is ndhwc.";
  }
  return shape;
}

ShapeVector DeviceShapeTransfer::ChannelLastDeviceShape(const ShapeVector &shape, const TypeId &) {
  auto dim = shape.size();
  ShapeVector axis;
  axis.resize(dim);
  const int step_value = 2;
  std::iota(axis.begin() + 1, axis.end(), step_value);
  axis[dim - 1] = 1;
  ShapeVector device_shape;
  (void)std::transform(axis.begin(), axis.end(), std::back_inserter(device_shape),
                       [&shape](size_t n) { return shape[n]; });
  return device_shape;
}

ShapeVector DeviceShapeTransfer::FRAC_NZDeviceShape(const ShapeVector &shape, const TypeId &type) {
  ShapeVector device_shape;
  auto c0 = GetCubeSizeByType(type);
  if (shape.size() == 1 && (shape[0] == 1 || shape[0] % c0 == 0)) {
    // For [1] and [1024] shape we can trait it as NZ shape
    return shape;
  }
  if (shape.size() < kShape2dDims) {
    MS_LOG(EXCEPTION) << "Format FRACTAL_NZ don't support shape with " << shape.size() << " dims";
  } else {
    const auto remove_dim = 2;
    (void)std::copy(shape.begin(), shape.end() - remove_dim, std::back_inserter(device_shape));
  }
  int64_t h_shape = shape[shape.size() - kH];
  int64_t w_shape = shape[shape.size() - 1];
  int64_t w1 = (w_shape == abstract::Shape::kShapeDimAny) ? abstract::Shape::kShapeDimAny : (w_shape - 1) / c0 + 1;
  int64_t h1 =
    (h_shape == abstract::Shape::kShapeDimAny) ? abstract::Shape::kShapeDimAny : (h_shape - 1) / kCubeSize + 1;
  device_shape.push_back(w1);
  device_shape.push_back(h1);
  device_shape.push_back(kCubeSize);
  device_shape.push_back(c0);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::FRAC_ZN_LSTMDeviceShape(const ShapeVector &shape, const TypeId &) {
  ShapeVector device_shape;
  const int64_t lstm_ni = 4;
  const int64_t ni = 16;
  int64_t first = abstract::Shape::kShapeDimAny;
  int64_t second = abstract::Shape::kShapeDimAny;
  if (!HasShapeDynamic({shape[kN], shape[kC]})) {
    const int64_t h = shape.at(kN) / lstm_ni;
    const int64_t i = shape.at(kC) - h;
    first = DivCeil(i, ni) + DivCeil(h, ni);
    second = lstm_ni * DivCeil(h, ni);
  }
  device_shape.push_back(first);
  device_shape.push_back(second);
  device_shape.push_back(ni);
  device_shape.push_back(ni);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::FRAC_ZDeviceShapeWithGroups(const ShapeVector &shape, const TypeId &type,
                                                             int64_t groups) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  if (groups <= 0) {
    MS_LOG(EXCEPTION) << "The value of groups should be greater than 0, but got " << groups;
  }
  auto cube_size = GetCubeSizeByType(type);
  auto c1_dim = abstract::Shape::kShapeDimAny;
  auto g_dim = abstract::Shape::kShapeDimAny;
  auto n1 = abstract::Shape::kShapeDimAny;
  if (!HasShapeDynamic({shape[kC], shape[kN]})) {
    auto group_size = groups;
    auto cin_ori_tmp = static_cast<int64_t>(shape[kC]);
    auto cout_ori_tmp = static_cast<int64_t>(shape[kN]) / group_size;
    auto e_mult =
      std::min(Lcm(Lcm(cin_ori_tmp, cube_size) / cin_ori_tmp, Lcm(cout_ori_tmp, cube_size) / cout_ori_tmp), group_size);
    auto cin_opt = DivCeil(e_mult * cin_ori_tmp, cube_size) * cube_size;
    c1_dim = cin_opt / cube_size;
    g_dim = DivCeil(group_size, e_mult);
    n1 = DivCeil(cout_ori_tmp * e_mult, cube_size);
  }
  ShapeVector device_shape;
  if (!HasShapeDynamic({shape[kC], shape[kN], shape[kH], shape[kW]})) {
    device_shape.push_back(g_dim * c1_dim * shape[kH] * shape[kW]);
  } else {
    device_shape.push_back(abstract::Shape::kShapeDimAny);
  }
  device_shape.push_back(n1);
  device_shape.push_back(kNiSize);
  device_shape.push_back(cube_size);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::FRAC_ZN_RNNDeviceShape(const ShapeVector &shape, const TypeId &type,
                                                        const ShapeVector &input_hidden_size) {
  if (shape.size() < kShape2dDims) {
    MS_LOG(EXCEPTION) << "Format FRACTAL_NZ_RNN don't support shape with " << shape.size() << " dims";
  }
  auto C0 = GetCubeSizeByType(type);
  auto input_size = input_hidden_size[0];
  auto hidden_size = input_hidden_size[1];
  auto dim_last1 = shape[shape.size() - 1];
  auto dim_last2 = shape[shape.size() - kDim2];
  const int64_t NUM16 = 16;

  ShapeVector device_shape = shape;
  if (dim_last2 == abstract::Shape::kShapeDimAny) {
    device_shape[shape.size() - kDim2] = abstract::Shape::kShapeDimAny;
  } else if (dim_last2 == input_size || dim_last2 == hidden_size) {
    device_shape[shape.size() - kDim2] = DivCeil(dim_last2, NUM16);
  } else if (dim_last2 == input_size + hidden_size) {
    device_shape[shape.size() - kDim2] = DivCeil(input_size, NUM16) + DivCeil(hidden_size, NUM16);
  } else {
    MS_LOG(EXCEPTION) << "The second-last dim value of shape is invalid.";
  }
  if (dim_last1 == abstract::Shape::kShapeDimAny) {
    device_shape[shape.size() - kDim1] = abstract::Shape::kShapeDimAny;
  } else {
    if (dim_last1 % hidden_size != 0) {
      MS_LOG(EXCEPTION) << "Last dim of shape " << shape << " should be multiple of hidden_size " << hidden_size;
    }
    int64_t n_num = shape[shape.size() - 1] / hidden_size;
    device_shape[shape.size() - kDim1] = n_num * DivCeil(hidden_size, C0);
  }
  device_shape.push_back(NUM16);
  device_shape.push_back(C0);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::NDRNNBiasDeviceShape(const ShapeVector &shape, const TypeId &type,
                                                      int64_t hidden_size) {
  if (shape.empty()) {
    MS_LOG(EXCEPTION) << "Format ND_RNN_BIAS don't support empty shape.";
  }
  auto C0 = GetCubeSizeByType(type);
  ShapeVector device_shape = shape;
  // cppcheck-suppress *
  auto dim_last1 = shape[shape.size() - 1];
  if (dim_last1 == abstract::Shape::kShapeDimAny) {
    device_shape[shape.size() - 1] = abstract::Shape::kShapeDimAny;
  } else {
    if (hidden_size <= 0 || dim_last1 % hidden_size != 0) {
      MS_LOG(EXCEPTION) << "Last dim of shape " << shape << " should be multiple of hidden_size " << hidden_size;
    }
    int64_t n_num = shape[shape.size() - 1] / hidden_size;
    device_shape[shape.size() - 1] = n_num * DivCeil(hidden_size, C0) * C0;
  }
  return device_shape;
}

ShapeVector DeviceShapeTransfer::GetAttrInputAndHiddenSize(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<int64_t> input_hidden_size = {kAlign16, kAlign16};
  if (!node->isa<CNode>() && !node->isa<Parameter>()) {
    return input_hidden_size;
  }

  if (node->isa<Parameter>()) {
    auto param = node->cast<ParameterPtr>();
    input_hidden_size[0] = param->input_size();
    input_hidden_size[1] = param->hidden_size();
  } else {
    CNodePtr cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || !common::AnfAlgo::HasNodeAttr(kAttrHiddenSize, cnode) ||
        !common::AnfAlgo::HasNodeAttr(kAttrInputSize, cnode)) {
      MS_LOG(EXCEPTION)
        << "Node with format FRACTAL_ZN_RNN or ND_RNN_BIAS should have hidden_size or input_size attr. Node info:"
        << node->DebugString();
    }
    input_hidden_size[0] = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrInputSize);
    input_hidden_size[1] = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrHiddenSize);
  }
  return input_hidden_size;
}

/**###################### DATA FORMAT TRANS ################################*/
inline void SetData(int64_t size, bool pad_zero, int64_t src_idx, int64_t dst_idx, const FormatArgs &args,
                    void *result) {
  switch (size) {
    case b1:
      static_cast<uint8_t *>(result)[dst_idx] = pad_zero ? 0 : static_cast<const uint8_t *>(args.data)[src_idx];
      break;
    case b2:
      static_cast<uint16_t *>(result)[dst_idx] = pad_zero ? 0 : static_cast<const uint16_t *>(args.data)[src_idx];
      break;
    case b4:
      static_cast<uint32_t *>(result)[dst_idx] = pad_zero ? 0 : static_cast<const uint32_t *>(args.data)[src_idx];
      break;
    case b8:
      static_cast<uint64_t *>(result)[dst_idx] = pad_zero ? 0 : static_cast<const uint64_t *>(args.data)[src_idx];
      break;
    default:
      MS_LOG(EXCEPTION) << "Trans data not support size " << size;
  }
}

bool FormatTransfer::TransDataByFormat(const FormatArgs &args, void *result, const AnfNodePtr &node, size_t index,
                                       bool is_forward) {
  int64_t groups = 1;
  if (args.device_format == kOpFormat_FRAC_Z && node != nullptr) {
    groups = common::AnfAlgo::GetAttrGroups(node, index);
  }
  if (is_forward) {
    return TransDataForwardCore(args, result, groups);
  }
  return TransDataBackwordCore(args, result, groups);
}

bool FormatTransfer::TransDataForwardCore(const FormatArgs &args, void *result, int64_t groups) {
  MS_LOG(DEBUG) << "Start trans format.";
  if (abstract::TypeIdSize(args.src_data_type) < 1) {
    MS_LOG(ERROR) << "Invalid datatype: " << args.src_data_type;
    return false;
  }
  if (groups > 1 && args.device_format == kOpFormat_FRAC_Z) {
    return NCHW_TO_FRAC_Z_WITH_GROUPS(args, result, true, groups);
  }
  auto iter = format_trans_fp_map.find(args.device_format);
  if (iter == format_trans_fp_map.end()) {
    MS_LOG(EXCEPTION) << "Unexpected format[" << args.device_format << "]";
  }
  return iter->second(args, result);
}

bool FormatTransfer::TransDataBackwordCore(const FormatArgs &args, void *result, int64_t groups) {
  MS_LOG(DEBUG) << "Start trans format.";
  if (abstract::TypeIdSize(args.src_data_type) < 1) {
    MS_LOG(ERROR) << "Invalid datatype, type: " << args.src_data_type;
    return false;
  }
  if (groups > 1 && args.device_format == kOpFormat_FRAC_Z) {
    return FRAC_Z_TO_NCHW_WITH_GROUPS(args, result, groups);
  }
  auto iter = format_trans_bp_map.find(args.device_format);
  if (iter == format_trans_bp_map.end()) {
    MS_LOG(EXCEPTION) << "Unexpected format[" << args.device_format << "]";
  }
  return iter->second(args, result);
}

bool FormatTransfer::CheckArgs(const FormatArgs &args, int64_t *size) {
  if (args.host_shape.size() != kDim4) {
    MS_LOG(ERROR) << "Invalid host shape, host shape dims:" << args.host_shape.size() << ", expect dims:" << kNchwDims;
    return false;
  }
  MS_EXCEPTION_IF_NULL(size);
  *size = SizeToLong(abstract::TypeIdSize(args.src_data_type));
  if (*size < 1) {
    MS_LOG(ERROR) << "Illegal dtype: " << args.src_data_type;
    return false;
  }
  auto total_size = abstract::ShapeSize(args.device_shape) * (*size);
  if (total_size != SizeToLong(args.device_size)) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << total_size << ", device_size:" << args.device_size;
    return false;
  }
  return true;
}

bool FormatTransfer::TransShapeToHW_NZ(const ShapeVector &host_shape, ShapeVector *hw_shape) {
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
      if (size < kDim2) {
        MS_LOG(ERROR) << "Illegal size: " << size;
        return false;
      }
      int64_t times = 1;
      for (size_t i = 0; i != size - kDim2; i++) {
        times *= host_shape[i];
      }
      hw_shape->push_back(times);
      hw_shape->push_back(host_shape[size - kDim2]);
      hw_shape->push_back(host_shape[size - kDim1]);
      return true;
  }
}

bool FormatTransfer::NCHW_TO_4D(const FormatArgs &args, void *result) {
  // trans nchw to NHWC or HWCN
  MS_LOG(DEBUG) << "Trans format from nchw to " << args.device_format;
  MS_EXCEPTION_IF_NULL(result);
  int64_t size = 0;
  if (!CheckArgs(args, &size)) {
    MS_LOG(ERROR) << "Check args failed.";
    return false;
  }
  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];
  for (int64_t ni = 0; ni < n; ni++) {
    for (int64_t ci = 0; ci < c; ci++) {
      for (int64_t hi = 0; hi < h; hi++) {
        for (int64_t wi = 0; wi < w; wi++) {
          auto src_idx = ni * c * h * w + ci * h * w + hi * w + wi;
          int64_t dst_idx = 0;
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

bool FormatTransfer::TO_NCHW(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format to nchw from " << args.device_format;
  MS_EXCEPTION_IF_NULL(result);
  int64_t size = 0;
  if (!CheckArgs(args, &size)) {
    MS_LOG(ERROR) << "Check args failed.";
    return false;
  }
  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];
  for (int64_t ni = 0; ni < n; ni++) {
    for (int64_t ci = 0; ci < c; ci++) {
      for (int64_t hi = 0; hi < h; hi++) {
        for (int64_t wi = 0; wi < w; wi++) {
          auto dst_idx = ni * c * h * w + ci * h * w + hi * w + wi;
          int64_t src_idx = 0;
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

bool FormatTransfer::NCHW_TO_FRAC_Z(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from nchw to frac_z";
  MS_EXCEPTION_IF_NULL(result);
  auto size = Common4DCheck(args);
  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];
  auto c0 = GetCubeSizeByType(args.src_data_type);
  auto c1 = DivCeil(c, c0);
  auto hw = h * w;
  auto chw = c * hw;
  auto hwc0 = hw * c0;
  auto nchw = n * chw;

  auto hf_cnt = DivCeil(n, kNiSize);
  auto vf_cnt = c1 * hw;
  auto fractal_ele_cnt = c0 * kNiSize;
  auto total_ele_cnt = hf_cnt * vf_cnt * fractal_ele_cnt;
  auto dst_size = total_ele_cnt * size;
  if (dst_size != SizeToLong(args.device_size)) {
    MS_LOG(ERROR) << "Illegal total data size, "
                  << "dst size is :" << dst_size << ", device size is :" << args.device_size;
    return false;
  }

  for (int64_t vfi = 0; vfi < vf_cnt; vfi++) {
    auto vf_base_i = vfi * hf_cnt;  // vertical fractal matrix base index
    for (int64_t hfi = 0; hfi < hf_cnt; hfi++) {
      auto gfi = vf_base_i + hfi;  // global fractal matrix index
      auto src_n_offset = hfi * chw * kNiSize;
      auto src_f_offset = src_n_offset + vfi % hw + vfi / hw * hwc0;
      for (int64_t row = 0; row < c0; row++) {
        auto src_ci = vfi / hw * c0 + row;
        auto src_row_offset = src_f_offset + row * hw;
        for (int64_t col = 0; col < kNiSize; col++) {
          auto src_ni = hfi * kNiSize + col;
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

bool FormatTransfer::NCHW_TO_FRAC_NZ(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from nchw to frac_nz.";
  MS_EXCEPTION_IF_NULL(result);
  ShapeVector hw_shape;
  if (!TransShapeToHW_NZ(args.host_shape, &hw_shape)) {
    MS_LOG(ERROR) << "Trans shape failed..";
    return false;
  }
  if (hw_shape.size() < kDim3 || args.device_shape.size() < kDim4) {
    MS_LOG(ERROR) << "Invalid shape size.";
    return false;
  }
  auto size = SizeToLong(abstract::TypeIdSize(args.src_data_type));
  if (size < 1) {
    MS_LOG(ERROR) << "Illegal dtype: " << args.src_data_type;
    return false;
  }

  auto dst_size = abstract::ShapeSize(args.device_shape) * size;
  if (dst_size != SizeToLong(args.device_size)) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << dst_size << ", device_size:" << args.device_size;
    return false;
  }
  auto times = hw_shape.at(0);
  auto h = hw_shape.at(hw_h);
  auto w = hw_shape.at(hw_w);
  auto hw = h * w;

  auto shape_size = args.device_shape.size();
  auto w1 = args.device_shape[shape_size - fnz_w1];
  auto h1 = args.device_shape[shape_size - fnz_h1];
  auto h0 = args.device_shape[shape_size - fnz_h0];
  auto w0 = args.device_shape[shape_size - fnz_w0];
  auto h1h0w0 = h1 * h0 * w0;
  auto w1h1h0w0 = w1 * h1h0w0;
  auto num_w1 = w / w0;

  for (int64_t times_idx = 0; times_idx < times; times_idx++) {
    auto times_head = times_idx * w1h1h0w0;
    auto src_times_head = times_idx * hw;
    for (int64_t h1h0_idx = 0; h1h0_idx < h; h1h0_idx++) {
      auto h1h0_head = times_head + h1h0_idx * w0;
      auto src_h_head = src_times_head + h1h0_idx * w;
      for (int64_t w1_idx = 0; w1_idx < num_w1; w1_idx++) {
        for (int64_t i = 0; i < w0; ++i) {
          int64_t src_idx = src_h_head + w1_idx * w0 + i;
          int64_t dst_idx = h1h0_head + w1_idx * h1h0w0 + i;
          SetData(size, false, src_idx, dst_idx, args, result);
        }
      }
      auto w1_head = num_w1 * w0;
      for (int64_t w0_idx = 0; w1_head + w0_idx < w; w0_idx++) {
        auto src_w_idx = w1_head + w0_idx;
        int64_t dst_idx = h1h0_head + num_w1 * h1h0w0 + w0_idx;
        int64_t src_idx = src_h_head + src_w_idx;
        SetData(size, false, src_idx, dst_idx, args, result);
      }
    }
  }
  return true;
}

bool FormatTransfer::NCHW_TO_FRAC_ZC04(const FormatArgs &args, void *result) {
  // trans nchw to FracZc04
  MS_LOG(DEBUG) << "Trans format from nchw to FracZc04.";
  MS_EXCEPTION_IF_NULL(result);
  int64_t size = 0;
  if (!CheckArgs(args, &size)) {
    MS_LOG(ERROR) << "Check args failed.";
    return false;
  }
  auto cube = GetCubeSizeByType(args.src_data_type);
  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];
  const int64_t c0 = 4;
  auto c1 = DivCeil(c, c0);
  auto hwc0 = h * w * c0;
  auto hwc = h * w * c;
  auto nhwc = n * h * w * c;
  auto n_cnt = DivCeil(n, kNiSize);
  auto v_cnt = DivCeil(h * w * c0 * c1, cube);
  int64_t dst_idx = 0;

  for (int64_t vi = 0; vi < v_cnt; vi++) {
    for (int64_t ni = 0; ni < n_cnt; ni++) {
      for (int64_t col = 0; col < kNiSize; col++) {
        for (int64_t row = 0; row < kNiSize; row++) {
          int64_t cur_cube_n = kNiSize * ni + col;
          int64_t cur_cube_c1hwc0 = kNiSize * vi + row;
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

bool FormatTransfer::NCHW_TO_NC1HWC0(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from nchw to Nc1h1wc0";
  MS_EXCEPTION_IF_NULL(result);
  auto size = Common4DCheck(args);
  auto total_size = abstract::ShapeSize(args.device_shape) * size;
  if (total_size != SizeToLong(args.device_size)) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << total_size << ", device_size:" << args.device_size;
    return false;
  }

  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];
  auto c0 = GetCubeSizeByType(args.src_data_type);
  if (args.device_format == kOpFormat_NC1HWC0_C04) {
    c0 = kCubeSize_C04;
  }
  auto c1 = DivCeil(c, c0);
  auto hw = h * w;
  auto chw = c * hw;
  auto c1hwc0 = c1 * hw * c0;
  auto wc0 = w * c0;

  for (int64_t n_idx = 0; n_idx < n; n_idx++) {
    int64_t n_head_addr = n_idx * c1hwc0;
    for (int64_t c1_idx = 0; c1_idx < c1; c1_idx++) {
      int64_t c1_head_addr = n_head_addr + c1_idx * hw * c0;
      for (int64_t h_idx = 0; h_idx < h; h_idx++) {
        int64_t h_head_addr = c1_head_addr + h_idx * wc0;
        for (int64_t w_idx = 0; w_idx < w; w_idx++) {
          int64_t w_head_addr = h_head_addr + w_idx * c0;
          for (int64_t c0_idx = 0; c0_idx < c0; c0_idx++) {
            int64_t dst_idx = c0_idx + w_head_addr;
            int64_t c_idx = c0_idx + c1_idx * c0;
            int64_t src_idx = n_idx * chw + c_idx * hw + h_idx * w + w_idx;
            auto pad_zero = c_idx >= c;
            SetData(size, pad_zero, src_idx, dst_idx, args, result);
          }
        }
      }
    }
  }
  return true;
}

bool FormatTransfer::NCHW_TO_NC1HWC04(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from nchw to Nc1hwc04.";
  return NCHW_TO_NC1HWC0(args, result);
}

bool FormatTransfer::NCHW_TO_C1HWNCOC0(const FormatArgs &args, void *result) {
  // trans nchw to c1hwncoc0
  MS_LOG(DEBUG) << "Trans format from nchw to c1hwncoc0.";
  MS_EXCEPTION_IF_NULL(result);
  int64_t size = 0;
  if (!CheckArgs(args, &size)) {
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

  for (int64_t c1_i = 0; c1_i < c1; c1_i++) {
    for (int64_t h_i = 0; h_i < h; h_i++) {
      for (int64_t w_i = 0; w_i < w; w_i++) {
        for (int64_t n_i = 0; n_i < n; n_i++) {
          for (int64_t co_i = 0; co_i < co; co_i++) {
            for (int64_t c0_i = 0; c0_i < c0; c0_i++) {
              int64_t dst_idx = c1_i * h * w * n * co * c0 + h_i * w * n * co * c0 + w_i * n * co * c0 + n_i * co * c0 +
                                co_i * c0 + c0_i;
              int64_t c_i = c0_i + c1_i * c0;
              int64_t src_idx = n_i * c * h * w + c_i * h * w + h_i * w + w_i;
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

bool FormatTransfer::NCDHW_TO_NDC1HWC0(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans from ncdhw to ndc1hwc0";
  MS_EXCEPTION_IF_NULL(result);

  if (args.host_shape.size() != kDim5) {
    MS_LOG(ERROR) << "Illegal host shape dim, expect dim: 5, but got " << args.host_shape.size();
    return false;
  }
  auto size = SizeToLong(abstract::TypeIdSize(args.src_data_type));
  if (size < 1) {
    MS_LOG(ERROR) << "Illegal dtype: " << args.src_data_type;
    return false;
  }
  auto total_size = abstract::ShapeSize(args.device_shape) * size;
  if (total_size != SizeToLong(args.device_size)) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << total_size << ", device_size:" << args.device_size;
    return false;
  }

  auto n = args.host_shape[N_ncdhw];
  auto c = args.host_shape[C_ncdhw];
  auto d = args.host_shape[D_ncdhw];
  auto h = args.host_shape[H_ncdhw];
  auto w = args.host_shape[W_ncdhw];
  auto c0 = GetCubeSizeByType(args.src_data_type);
  auto c1 = DivCeil(c, c0);
  const int64_t cdhw = c * d * h * w;
  const int64_t dhw = d * h * w;
  const int64_t hw = h * w;
  const int64_t dc1hwc0 = d * c1 * h * w * c0;
  const int64_t c1hwc0 = c1 * h * w * c0;
  const int64_t hwc0 = h * w * c0;
  const int64_t wc0 = w * c0;

  for (int64_t n_i = 0; n_i < n; n_i++) {
    int64_t n_head = n_i * dc1hwc0;
    for (int64_t d_i = 0; d_i < d; d_i++) {
      int64_t d_head = n_head + d_i * c1hwc0;
      for (int64_t c1_i = 0; c1_i < c1; c1_i++) {
        int64_t c1_head = d_head + c1_i * hwc0;
        for (int64_t h_i = 0; h_i < h; h_i++) {
          int64_t h_head = c1_head + h_i * wc0;
          for (int64_t w_i = 0; w_i < w; w_i++) {
            int64_t w_head = h_head + w_i * c0;
            for (int64_t c0_i = 0; c0_i < c0; c0_i++) {
              int64_t dst_i = c0_i + w_head;
              int64_t c_i = c0_i + c1_i * c0;
              int64_t src_i = n_i * cdhw + c_i * dhw + d_i * hw + h_i * w + w_i;
              auto pad_zero = c_i >= c;
              SetData(size, pad_zero, src_i, dst_i, args, result);
            }
          }
        }
      }
    }
  }
  return true;
}

bool FormatTransfer::NCDHW_TO_FRAC_Z3D(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans from ncdhw to frac_z_3d";
  MS_EXCEPTION_IF_NULL(result);

  if (args.host_shape.size() != kDim5) {
    MS_LOG(ERROR) << "Illegal host shape dim, expect dim: 5, but got " << args.host_shape.size();
    return false;
  }
  auto size = SizeToLong(abstract::TypeIdSize(args.src_data_type));
  if (size < 1) {
    MS_LOG(ERROR) << "Illegal dtype: " << args.src_data_type;
    return false;
  }
  auto total_size = abstract::ShapeSize(args.device_shape) * size;
  if (total_size != SizeToLong(args.device_size)) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << total_size << ", device_size:" << args.device_size;
    return false;
  }

  auto n = args.host_shape[N_ncdhw];
  auto c = args.host_shape[C_ncdhw];
  auto d = args.host_shape[D_ncdhw];
  auto h = args.host_shape[H_ncdhw];
  auto w = args.host_shape[W_ncdhw];

  auto n1n0 = DivCeil(n, kNiSize) * kNiSize;
  auto c0 = GetCubeSizeByType(args.src_data_type);
  auto c1 = DivCeil(c, c0);
  auto hw = h * w;
  auto dhw = d * hw;
  auto cdhw = c * dhw;
  auto n1n0c0 = n1n0 * c0;
  auto wn1n0c0 = w * n1n0c0;
  auto hwn1n0c0 = h * wn1n0c0;
  auto c1hwn1n0c0 = c1 * hwn1n0c0;

  for (int64_t d_i = 0; d_i < d; d_i++) {
    for (int64_t c1_i = 0; c1_i < c1; c1_i++) {
      for (int64_t h_i = 0; h_i < h; h_i++) {
        for (int64_t w_i = 0; w_i < w; w_i++) {
          for (int64_t n1n0_i = 0; n1n0_i < n1n0; n1n0_i++) {
            for (int64_t c0_i = 0; c0_i < c0; c0_i++) {
              auto dst_i = d_i * c1hwn1n0c0 + c1_i * hwn1n0c0 + h_i * wn1n0c0 + w_i * n1n0c0 + n1n0_i * c0 + c0_i;
              // ncdhw
              int64_t src_i = n1n0_i * cdhw + (c1_i * c0 + c0_i) * dhw + d_i * hw + h_i * w + w_i;
              auto pad_zero = ((c1_i * c0 + c0_i) >= c) || (n1n0_i >= n);
              SetData(size, pad_zero, src_i, dst_i, args, result);
            }
          }
        }
      }
    }
  }
  return true;
}

bool FormatTransfer::NCHW_TO_FRAC_Z_WITH_GROUPS(const FormatArgs &args, void *result, bool to_device, int64_t groups) {
  MS_EXCEPTION_IF_NULL(result);
  auto size = Common4DCheck(args);
  auto n_dim = args.host_shape[kN];
  auto c_dim = args.host_shape[kC];
  auto h_dim = args.host_shape[kH];
  auto w_dim = args.host_shape[kW];
  auto d_dim = 1;
  auto cin_ori = c_dim;
  if (groups <= 0) {
    MS_LOG(EXCEPTION) << "The value of groups should be greater than 0, but got " << groups;
  }
  // cppcheck-suppress *
  auto cout_ori = n_dim / groups;
  if (cin_ori == 0 || cout_ori == 0) {
    MS_LOG(ERROR) << "cin_ori, cout_ori must not equal to 0";
    return false;
  }
  auto cube_k = GetCubeSizeByType(args.src_data_type);
  auto e_mult = std::min(Lcm(Lcm(cin_ori, cube_k) / cin_ori, Lcm(cout_ori, kCubeSize) / cout_ori), groups);
  if (e_mult == 0) {
    MS_LOG(EXCEPTION) << "The value of e_mult should be greater than 0, but got " << e_mult;
  }
  auto cin_opt = DivCeil(e_mult * cin_ori, cube_k) * cube_k;
  auto cout_opt = DivCeil(e_mult * cout_ori, kCubeSize) * kCubeSize;
  // cppcheck-suppress *
  auto c1_dim = cin_opt / cube_k;
  auto dst_size =
    to_device ? abstract::ShapeSize(args.device_shape) * size : abstract::ShapeSize(args.host_shape) * size;
  if (dst_size == 0) {
    return true;
  }
  auto ret = memset_s(result, LongToSize(dst_size), 0, LongToSize(dst_size));
  if (ret != EOK) {
    MS_LOG(ERROR) << "memset failed";
    return false;
  }
  for (int64_t g = 0; g < groups; ++g) {
    for (int64_t d = 0; d < d_dim; ++d) {
      for (int64_t c = 0; c < c_dim; ++c) {
        for (int64_t h = 0; h < h_dim; ++h) {
          for (int64_t w = 0; w < w_dim; ++w) {
            for (int64_t n = 0; n < cout_ori; ++n) {
              int64_t e_val = g % e_mult;
              int64_t dst_ci = e_val * cin_ori + c;
              int64_t dst_co = e_val * cout_ori + n;
              int64_t src_co = g * cout_ori + n;
              int64_t temporary = dst_ci % cube_k;
              int64_t dev_idx = (g / e_mult) * d_dim * c1_dim * h_dim * w_dim * cout_opt * cube_k +
                                d * c1_dim * h_dim * w_dim * cout_opt * cube_k +
                                (dst_ci / cube_k) * h_dim * w_dim * cout_opt * cube_k + h * w_dim * cout_opt * cube_k +
                                w * cout_opt * cube_k + dst_co * cube_k + temporary;
              int64_t hst_idx =
                src_co * c_dim * d_dim * h_dim * w_dim + c * d_dim * h_dim * w_dim + d * h_dim * w_dim + h * w_dim + w;
              if (to_device) {
                SetData(size, false, hst_idx, dev_idx, args, result);
              } else {
                SetData(size, false, dev_idx, hst_idx, args, result);
              }
            }
          }
        }
      }
    }
  }
  return true;
}

bool FormatTransfer::NC1HWC0_TO_NCHW(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from nc1h1wc0 to nchw";
  MS_EXCEPTION_IF_NULL(result);
  auto size = Common4DCheck(args);
  auto total_size = abstract::ShapeSize(args.device_shape) * size;
  if (total_size != SizeToLong(args.device_size)) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << total_size << ", device_size:" << args.device_size;
    return false;
  }

  auto n = args.host_shape[kN];
  auto c = args.host_shape[kC];
  auto h = args.host_shape[kH];
  auto w = args.host_shape[kW];
  auto c1 = args.device_shape[kDim1];
  auto c0 = args.device_shape[kDim4];

  auto hw = h * w;
  auto chw = c * hw;
  auto wc0 = w * c0;
  auto hwc0 = h * wc0;
  auto c1hwc0 = c1 * hwc0;

  for (int64_t n_idx = 0; n_idx < n; n_idx++) {
    int64_t n_head_addr = n_idx * chw;
    for (int64_t c_idx = 0; c_idx < c; c_idx++) {
      int64_t c_head_addr = n_head_addr + c_idx * hw;
      for (int64_t h_idx = 0; h_idx < h; h_idx++) {
        int64_t h_head_addr = c_head_addr + h_idx * w;
        for (int64_t w_idx = 0; w_idx < w; w_idx++) {
          int64_t dst_idx = h_head_addr + w_idx;
          int64_t c1_idx = c_idx / c0;
          int64_t c0_idx = c_idx % c0;
          int64_t src_idx = n_idx * c1hwc0 + c1_idx * hwc0 + h_idx * wc0 + w_idx * c0 + c0_idx;
          SetData(size, false, src_idx, dst_idx, args, result);
        }
      }
    }
  }
  return true;
}

bool FormatTransfer::NC1HWC04_TO_NCHW(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from Nc1hwc04 to nchw.";
  return NC1HWC0_TO_NCHW(args, result);
}

bool FormatTransfer::C1HWNCOC0_TO_NCHW(const FormatArgs &args, void *result) {
  // trans c1hwncoc0 to nchw
  MS_LOG(DEBUG) << "Trans format from c1hwncoc0 to nchw";
  MS_EXCEPTION_IF_NULL(result);
  int64_t size = 0;
  if (!CheckArgs(args, &size)) {
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
  auto cube_k = GetCubeSizeByType(args.src_data_type);
  for (int64_t n_i = 0; n_i < n; n_i++) {
    for (int64_t c_i = 0; c_i < c; c_i++) {
      for (int64_t h_i = 0; h_i < h; h_i++) {
        for (int64_t w_i = 0; w_i < w; w_i++) {
          int64_t dst_idx = n_i * c * h * w + c_i * h * w + h_i * w + w_i;
          int64_t c1_i = c_i / cube_k;
          int64_t c0_i = c_i % cube_k;
          int64_t co_i = c0_i;
          int64_t src_idx =
            c1_i * h * w * n * co * c0 + h_i * w * n * co * c0 + w_i * n * co * c0 + n_i * co * c0 + co_i * c0 + c0_i;
          SetData(size, false, src_idx, dst_idx, args, result);
        }
      }
    }
  }
  return true;
}

bool FormatTransfer::FRAC_Z_TO_NCHW(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from frac_z to nchw";
  MS_EXCEPTION_IF_NULL(result);
  auto size = Common4DCheck(args);
  auto total_size = abstract::ShapeSize(args.device_shape) * size;
  if (total_size != SizeToLong(args.device_size)) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << total_size << ", device_size:" << args.device_size;
    return false;
  }

  auto n0 = args.device_shape.at(fz_n0);
  auto ni = args.device_shape.at(fz_ni);
  auto c0 = args.device_shape.at(fz_c0);
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

  for (int64_t n_idx = 0; n_idx < n; n_idx++) {
    int64_t n_head_addr = n_idx * chw;
    for (int64_t c_idx = 0; c_idx < c; c_idx++) {
      int64_t c_head_addr = n_head_addr + c_idx * hw;
      for (int64_t h_idx = 0; h_idx < h; h_idx++) {
        int64_t h_head_addr = c_head_addr + h_idx * w;
        for (int64_t w_idx = 0; w_idx < w; w_idx++) {
          auto dst_idx = h_head_addr + w_idx;
          auto c1_idx = c_idx / c0;
          auto c0_idx = c_idx % c0;
          auto nc_idx = n_idx;
          auto src_idx = c1_idx * hwncc0 + h_idx * wncc0 + w_idx * ncc0 + nc_idx * c0 + c0_idx;
          SetData(size, false, src_idx, dst_idx, args, result);
        }
      }
    }
  }
  return true;
}

bool FormatTransfer::FRAC_NZ_TO_NCHW(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans format from frac_nz to nchw";
  MS_EXCEPTION_IF_NULL(result);
  ShapeVector hw_shape;
  if (!TransShapeToHW_NZ(args.host_shape, &hw_shape)) {
    MS_LOG(ERROR) << "Trans shape failed..";
    return false;
  }
  if (hw_shape.size() < kDim3 || args.device_shape.size() < kDim4) {
    MS_LOG(ERROR) << "Invalid shape size.";
    return false;
  }
  auto size = SizeToLong(abstract::TypeIdSize(args.src_data_type));
  if (size < 1) {
    MS_LOG(ERROR) << "Illegal dtype: " << args.src_data_type;
    return false;
  }

  auto dst_size = abstract::ShapeSize(args.device_shape) * size;
  if (dst_size != SizeToLong(args.device_size)) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << dst_size << ", device_size:" << args.device_size;
    return false;
  }
  auto times = hw_shape.at(0);
  auto h = hw_shape.at(hw_h);
  auto w = hw_shape.at(hw_w);
  auto hw = h * w;

  auto shape_size = args.device_shape.size();
  auto w1 = args.device_shape[shape_size - fnz_w1];
  auto h1 = args.device_shape[shape_size - fnz_h1];
  auto h0 = args.device_shape[shape_size - fnz_h0];
  auto w0 = args.device_shape[shape_size - fnz_w0];
  auto h1h0w0 = h1 * h0 * w0;
  auto w1h1h0w0 = w1 * h1h0w0;
  auto num_w1 = w / w0;

  for (int64_t times_idx = 0; times_idx < times; times_idx++) {
    auto times_head = times_idx * w1h1h0w0;
    auto src_times_head = times_idx * hw;
    for (int64_t h1h0_idx = 0; h1h0_idx < h; h1h0_idx++) {
      auto h1h0_head = times_head + h1h0_idx * w0;
      auto src_h_head = src_times_head + h1h0_idx * w;
      for (int64_t w1_idx = 0; w1_idx < num_w1; w1_idx++) {
        for (int64_t i = 0; i < w0; ++i) {
          int64_t src_idx = h1h0_head + w1_idx * h1h0w0 + i;
          int64_t dst_idx = src_h_head + w1_idx * w0 + i;
          SetData(size, false, src_idx, dst_idx, args, result);
        }
      }
      auto w1_head = num_w1 * w0;
      for (int64_t w0_idx = 0; w1_head + w0_idx < w; w0_idx++) {
        auto src_w_idx = w1_head + w0_idx;
        int64_t src_idx = h1h0_head + num_w1 * h1h0w0 + w0_idx;
        int64_t dst_idx = src_h_head + src_w_idx;
        SetData(size, false, src_idx, dst_idx, args, result);
      }
    }
  }
  return true;
}

bool FormatTransfer::FRAC_Z3D_TO_NCDHW(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans from frac_z_3d to ncdhw";
  MS_EXCEPTION_IF_NULL(result);

  if (args.host_shape.size() != kDim5) {
    MS_LOG(ERROR) << "Illegal host shape dim, expect dim: 5, but got " << args.host_shape.size();
    return false;
  }
  auto size = SizeToLong(abstract::TypeIdSize(args.src_data_type));
  if (size < 1) {
    MS_LOG(ERROR) << "Illegal dtype: " << args.src_data_type;
    return false;
  }
  auto total_size = abstract::ShapeSize(args.device_shape) * size;
  if (total_size != SizeToLong(args.device_size)) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << total_size << ", device_size:" << args.device_size;
    return false;
  }
  auto n = args.host_shape[N_ncdhw];
  auto c = args.host_shape[C_ncdhw];
  auto d = args.host_shape[D_ncdhw];
  auto h = args.host_shape[H_ncdhw];
  auto w = args.host_shape[W_ncdhw];
  const int kFZ3D_C0 = 3;
  auto c0 = args.device_shape[kFZ3D_C0];
  auto cube_k = GetCubeSizeByType(args.src_data_type);
  auto c1 = DivCeil(c, cube_k);
  auto n1n0 = DivCeil(n, kNiSize) * kNiSize;
  auto n1n0c0 = n1n0 * c0;
  auto wn1n0c0 = w * n1n0c0;
  auto hwn1n0c0 = h * wn1n0c0;
  auto c1hwn1n0c0 = c1 * hwn1n0c0;
  auto hw = h * w;
  auto dhw = d * hw;
  auto cdhw = c * dhw;

  for (int64_t n_i = 0; n_i < n; n_i++) {
    int64_t n_head = n_i * cdhw;
    for (int64_t c_i = 0; c_i < c; c_i++) {
      int64_t c_head = n_head + c_i * dhw;
      for (int64_t d_i = 0; d_i < d; d_i++) {
        int64_t d_head = c_head + d_i * hw;
        for (int64_t h_i = 0; h_i < h; h_i++) {
          int64_t h_head = d_head + h_i * w;
          for (int64_t w_i = 0; w_i < w; w_i++) {
            int64_t dst_i = h_head + w_i;
            int64_t c1_i = c_i / c0;
            int64_t c0_i = c_i % c0;
            int64_t nc_i = n_i;
            int64_t src_i = d_i * c1hwn1n0c0 + c1_i * hwn1n0c0 + h_i * wn1n0c0 + w_i * n1n0c0 + nc_i * c0 + c0_i;
            SetData(size, false, src_i, dst_i, args, result);
          }
        }
      }
    }
  }
  return true;
}

bool FormatTransfer::NDC1HWC0_TO_NCDHW(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "Trans from ndc1hwc0 to ncdhw";
  MS_EXCEPTION_IF_NULL(result);

  if (args.host_shape.size() != kDim5) {
    MS_LOG(ERROR) << "Illegal host shape dim, expect dim: 5, but got " << args.host_shape.size();
    return false;
  }
  auto size = SizeToLong(abstract::TypeIdSize(args.src_data_type));
  if (size < 1) {
    MS_LOG(ERROR) << "Illegal dtype: " << args.src_data_type;
    return false;
  }
  auto total_size = abstract::ShapeSize(args.device_shape) * size;
  if (total_size != SizeToLong(args.device_size)) {
    MS_LOG(ERROR) << "Illegal total data size, total_size:" << total_size << ", device_size:" << args.device_size;
    return false;
  }
  auto n = args.host_shape[N_ncdhw];
  auto c = args.host_shape[C_ncdhw];
  auto d = args.host_shape[D_ncdhw];
  auto h = args.host_shape[H_ncdhw];
  auto w = args.host_shape[W_ncdhw];
  auto c1 = args.device_shape[C1_ndc1hwc0];
  auto c0 = args.device_shape[C0_ndc1hwc0];
  const int64_t cdhw = c * d * h * w;
  const int64_t dhw = d * h * w;
  const int64_t hw = h * w;
  const int64_t dc1hwc0 = d * c1 * h * w * c0;
  const int64_t c1hwc0 = c1 * h * w * c0;
  const int64_t hwc0 = h * w * c0;
  const int64_t wc0 = w * c0;

  for (int64_t n_i = 0; n_i < n; n_i++) {
    int64_t n_head = n_i * cdhw;
    for (int64_t c_i = 0; c_i < c; c_i++) {
      int64_t c_head = n_head + c_i * dhw;
      for (int64_t d_i = 0; d_i < d; d_i++) {
        int64_t d_head = c_head + d_i * hw;
        for (int64_t h_i = 0; h_i < h; h_i++) {
          int64_t h_head = d_head + h_i * w;
          for (int64_t w_i = 0; w_i < w; w_i++) {
            int64_t dst_i = h_head + w_i;
            int64_t c1_i = c_i / c0;
            int64_t c0_i = c_i % c0;
            auto src_idx = n_i * dc1hwc0 + d_i * c1hwc0 + c1_i * hwc0 + h_i * wc0 + w_i * c0 + c0_i;
            SetData(size, false, src_idx, dst_i, args, result);
          }
        }
      }
    }
  }
  return true;
}

bool FormatTransfer::FRAC_Z_TO_NCHW_WITH_GROUPS(const FormatArgs &args, void *result, int64_t groups) {
  MS_LOG(DEBUG) << "Trans format from frac_z to nchw with groups=" << groups;
  return NCHW_TO_FRAC_Z_WITH_GROUPS(args, result, false, groups);
}

int64_t FormatTransfer::Common4DCheck(const FormatArgs &args) {
  if (args.host_shape.size() != kDim4) {
    MS_LOG(EXCEPTION) << "Invalid host shape, host shape dims:" << args.host_shape.size()
                      << ", expect dims:" << kNchwDims;
  }
  auto size = SizeToLong(abstract::TypeIdSize(args.src_data_type));
  if (size < 1) {
    MS_LOG(EXCEPTION) << "Illegal dtype: " << args.src_data_type;
  }
  return size;
}

// ########################  RANGE TRANS ########################
RangePair ShapeRangeTransfer::GetRealRange(const RangePair &ori_range, const std::string &format, const TypeId &type,
                                           const std::string &padding_str) const {
  const std::set<std::string> no_need_change = {kOpFormat_ND, kOpFormat_DEFAULT, kOpFormat_NCHW, kOpFormat_NCDHW};
  using RangeTransfer = std::function<RangePair(const RangePair &, const TypeId &)>;
  const std::map<std::string, RangeTransfer> format_range_map = {{kOpFormat_NHWC, NHWCRange},
                                                                 {kOpFormat_HWCN, HWCNRange},
                                                                 {kOpFormat_FRAC_Z, FRAC_ZRange},
                                                                 {kOpFormat_NC1HWC0, NC1HWC0Range},
                                                                 {kOpFormat_NDC1HWC0, NDC1HWC0Range},
                                                                 {kOpFormat_C1HWNCoC0, C1HWNCOC0Range},
                                                                 {kOpFormat_NC1HWC0_C04, NC1HWC04Range},
                                                                 {kOpFormat_FRACTAL_Z_3D, FRAC_Z_3DRange},
                                                                 {kOpFormat_FRACTAL_Z_C04, FRAC_ZC04Range}};
  if (no_need_change.find(format) != no_need_change.end()) {
    return ori_range;
  }
  // kOpFormat_FRACTAL_ZN_LSTM, kOpFormat_FRAC_NZ no need pad range
  if (format == kOpFormat_FRACTAL_ZN_LSTM) {
    return FRAC_ZN_LSTMRange(ori_range, type);
  }
  if (format == kOpFormat_FRAC_NZ) {
    return FRAC_NZRange(ori_range, type);
  }
  auto temp_range = ori_range;
  if (ori_range.size() < kDim4 && !IsOneOf3DFormat(format)) {
    MS_LOG(DEBUG) << "A special format:" << format << " with a range size less than 4, so padding the range firstly";
    temp_range = PaddingRangeTo4D(ori_range, padding_str);
  }
  if (ori_range.size() < kDim5 && IsOneOf3DFormat(format)) {
    MS_LOG(DEBUG) << "A special format:" << format << " with a range size less than 5, so padding the range firstly";
    temp_range = PaddingRangeTo5D(ori_range, padding_str);
  }
  auto iter = format_range_map.find(format);
  if (iter == format_range_map.end()) {
    MS_LOG(INFO) << "Can not find a supported format: " << format << ", using default range";
    return ori_range;
  }
  return iter->second(temp_range, type);
}

RangePair ShapeRangeTransfer::NHWCRange(const RangePair &ori_range, const TypeId &) {
  RangePair dst_range;
  dst_range.push_back(ori_range[kN]);
  dst_range.push_back(ori_range[kH]);
  dst_range.push_back(ori_range[kW]);
  dst_range.push_back(ori_range[kC]);
  return dst_range;
}

RangePair ShapeRangeTransfer::HWCNRange(const RangePair &ori_range, const TypeId &) {
  RangePair dst_range;
  dst_range.push_back(ori_range[kH]);
  dst_range.push_back(ori_range[kW]);
  dst_range.push_back(ori_range[kC]);
  dst_range.push_back(ori_range[kN]);
  return dst_range;
}

RangePair ShapeRangeTransfer::NC1HWC04Range(const RangePair &ori_range, const TypeId &) {
  RangePair dst_range;
  const std::pair<int64_t, int64_t> c0 = {k4, k4};
  auto tmp_max = CalMaxShape(ori_range[kC].second, (ori_range[kC].second + k4 - 1) / k4);
  const std::pair<int64_t, int64_t> c1 = {(ori_range[kC].first + k4 - 1) / k4, tmp_max};
  dst_range.push_back(ori_range[kN]);
  dst_range.push_back(c1);
  dst_range.push_back(ori_range[kH]);
  dst_range.push_back(ori_range[kW]);
  dst_range.push_back(c0);
  return dst_range;
}

RangePair ShapeRangeTransfer::FRAC_ZC04Range(const RangePair &ori_range, const TypeId &) {
  RangePair dst_range;
  const std::pair<int64_t, int64_t> c0 = {k4, k4};
  const std::pair<int64_t, int64_t> c16 = {kNiSize, kNiSize};

  auto tmp_max = CalMaxShape(c0.second * ori_range[kH].second * ori_range[kW].second,
                             (c0.second * ori_range[kH].second * ori_range[kW].second + kNiSize - 1) / kNiSize);
  const std::pair<int64_t, int64_t> first_dim = {
    (c0.first * ori_range[kH].first * ori_range[kW].first + kNiSize - 1) / kNiSize, tmp_max};

  tmp_max = CalMaxShape(ori_range[kN].second, (ori_range[kN].second + kNiSize - 1) / kNiSize);
  const std::pair<int64_t, int64_t> no = {(ori_range[kN].first + kNiSize - 1) / kNiSize, tmp_max};
  dst_range.push_back(first_dim);
  dst_range.push_back(no);
  dst_range.push_back(c16);
  dst_range.push_back(c16);
  return dst_range;
}

RangePair ShapeRangeTransfer::FRAC_ZRange(const RangePair &ori_range, const TypeId &type) {
  RangePair dst_range;
  auto cube = GetCubeSizeByType(type);
  const std::pair<int64_t, int64_t> c0 = {cube, cube};

  auto tmp_max = CalMaxShape(ori_range[kN].second, ((ori_range[kN].second + kNiSize - 1) / kNiSize) * kNiSize);

  const std::pair<int64_t, int64_t> cout16 = {((ori_range[kN].first + kNiSize - 1) / kNiSize) * kNiSize, tmp_max};

  tmp_max = CalMaxShape(ori_range[kC].second, ((ori_range[kC].second + cube - 1) / cube) * cube);
  const std::pair<int64_t, int64_t> cin16 = {((ori_range[kC].first + cube - 1) / cube) * cube, tmp_max};

  tmp_max = CalMaxShape(ori_range[kH].second * ori_range[kW].second * cin16.second,
                        ori_range[kH].second * ori_range[kW].second * cin16.second / cube);
  const std::pair<int64_t, int64_t> r0 = {ori_range[kH].first * ori_range[kW].first * cin16.first / cube, tmp_max};

  tmp_max = CalMaxShape(cin16.second, cout16.second / kNiSize);
  const std::pair<int64_t, int64_t> r1 = {cout16.first / kNiSize, tmp_max};
  const std::pair<int64_t, int64_t> co = {kNiSize, kNiSize};
  dst_range.push_back(r0);
  dst_range.push_back(r1);
  dst_range.push_back(co);
  dst_range.push_back(c0);
  return dst_range;
}

RangePair ShapeRangeTransfer::FRAC_NZRange(const RangePair &ori_range, const TypeId &type) {
  RangePair dst_range;
  auto cube = GetCubeSizeByType(type);
  auto ori_size = ori_range.size();
  if (ori_size < kDims2) {
    return ori_range;
  } else {
    (void)std::copy(ori_range.begin(), ori_range.end() - kDims2, std::back_inserter(dst_range));
  }
  const std::pair<int64_t, int64_t> c0 = {cube, cube};
  auto tmp_max = CalMaxShape(ori_range[ori_size - 1].second, (ori_range[ori_size - 1].second - 1) / cube + 1);
  const std::pair<int64_t, int64_t> w1 = {(ori_range[ori_size - 1].first - 1) / cube + 1, tmp_max};
  tmp_max = CalMaxShape(ori_range[ori_size - kDims2].second, (ori_range[ori_size - kDims2].second - 1) / kNiSize + 1);
  const std::pair<int64_t, int64_t> h1 = {(ori_range[ori_size - kDims2].first - 1) / kNiSize + 1, tmp_max};
  const std::pair<int64_t, int64_t> co = {kNiSize, kNiSize};
  dst_range.push_back(w1);
  dst_range.push_back(h1);
  dst_range.push_back(co);
  dst_range.push_back(c0);
  return dst_range;
}

RangePair ShapeRangeTransfer::NC1HWC0Range(const RangePair &ori_range, const TypeId &type) {
  RangePair dst_range;
  auto cube = GetCubeSizeByType(type);
  const std::pair<int64_t, int64_t> c0 = {cube, cube};
  auto tmp_max = CalMaxShape(ori_range[kC].second, (ori_range[kC].second + cube - 1) / cube);
  const std::pair<int64_t, int64_t> c1 = {(ori_range[kC].first + cube - 1) / cube, tmp_max};
  dst_range.push_back(ori_range[kN]);
  dst_range.push_back(c1);
  dst_range.push_back(ori_range[kH]);
  dst_range.push_back(ori_range[kW]);
  dst_range.push_back(c0);
  return dst_range;
}

RangePair ShapeRangeTransfer::FRAC_ZN_LSTMRange(const RangePair &ori_range, const TypeId &) {
  RangePair dst_range;
  const std::pair<int64_t, int64_t> c0 = {k4, k4};
  const std::pair<int64_t, int64_t> c16 = {k4, k4};

  auto tmp_max = CalMaxShape(ori_range[kN].second, ori_range[kN].second / c0.second);
  const std::pair<int64_t, int64_t> h = {ori_range[kN].first / c0.first, tmp_max};

  tmp_max = CalMaxShape(ori_range[kC].second * h.second, ori_range[kC].second - h.second);
  const std::pair<int64_t, int64_t> i = {ori_range[kC].first - h.first, tmp_max};

  tmp_max = CalMaxShape(i.second * h.second, (i.second + kCube16 - 1) / kCube16 + (h.second + kCube16 - 1) / kCube16);
  const std::pair<int64_t, int64_t> first_dim = {(i.first + kCube16 - 1) / kCube16 + (h.first + kCube16 - 1) / kCube16,
                                                 tmp_max};

  tmp_max = CalMaxShape(h.second, c0.second * ((h.second + kCube16 - 1) / kCube16));
  const std::pair<int64_t, int64_t> second = {c0.first * ((h.first + kCube16 - 1) / kCube16), tmp_max};
  dst_range.push_back(first_dim);
  dst_range.push_back(second);
  dst_range.push_back(c16);
  dst_range.push_back(c16);
  return dst_range;
}

RangePair ShapeRangeTransfer::NDC1HWC0Range(const RangePair &ori_range, const TypeId &type) {
  RangePair dst_range;
  auto cube = GetCubeSizeByType(type);
  const std::pair<int64_t, int64_t> c0 = {cube, cube};
  auto tmp_max = CalMaxShape(ori_range[C_ncdhw].second, (ori_range[C_ncdhw].second + cube - 1) / cube);
  const std::pair<int64_t, int64_t> c1 = {(ori_range[C_ncdhw].first + cube - 1) / cube, tmp_max};
  dst_range.push_back(ori_range[N_ncdhw]);
  dst_range.push_back(ori_range[D_ncdhw]);
  dst_range.push_back(c1);
  dst_range.push_back(ori_range[H_ncdhw]);
  dst_range.push_back(ori_range[W_ncdhw]);
  dst_range.push_back(c0);
  return dst_range;
}

RangePair ShapeRangeTransfer::C1HWNCOC0Range(const RangePair &ori_range, const TypeId &type) {
  RangePair dst_range;
  auto cube = GetCubeSizeByType(type);
  const std::pair<int64_t, int64_t> c0 = {cube, cube};
  auto tmp_max = CalMaxShape(ori_range[kC].second, (ori_range[kC].second - 1) / cube + 1);
  const std::pair<int64_t, int64_t> r1 = {(ori_range[kC].first - 1) / cube + 1, tmp_max};
  dst_range.push_back(r1);
  dst_range.push_back(ori_range[kH]);
  dst_range.push_back(ori_range[kW]);
  dst_range.push_back(ori_range[kN]);
  dst_range.push_back(c0);
  dst_range.push_back(c0);
  return dst_range;
}

RangePair ShapeRangeTransfer::FRAC_Z_3DRange(const RangePair &ori_range, const TypeId &type) {
  RangePair dst_range;
  auto cube = GetCubeSizeByType(type);
  const std::pair<int64_t, int64_t> c0 = {cube, cube};
  auto tmp_max = CalMaxShape(ori_range[C_ncdhw].second, (ori_range[C_ncdhw].second + cube - 1) / cube);
  const std::pair<int64_t, int64_t> c1 = {(ori_range[C_ncdhw].first + cube - 1) / cube, tmp_max};

  tmp_max = CalMaxShape(ori_range[N_ncdhw].second, (ori_range[N_ncdhw].second + kNiSize - 1) / kNiSize);
  const std::pair<int64_t, int64_t> n1 = {(ori_range[N_ncdhw].first + kNiSize - 1) / kNiSize, tmp_max};

  const int64_t r1_0 = ori_range[D_ncdhw].first * c1.first * ori_range[H_ncdhw].first * ori_range[W_ncdhw].first;
  const int64_t r1_1 =
    CalMaxShape(ori_range[D_ncdhw].second * c1.second * ori_range[H_ncdhw].second * ori_range[W_ncdhw].second,
                ori_range[D_ncdhw].second * c1.second * ori_range[H_ncdhw].second * ori_range[W_ncdhw].second);
  const std::pair<int64_t, int64_t> r1 = {r1_0, r1_1};
  dst_range.push_back(r1);
  dst_range.push_back(n1);
  dst_range.push_back(c1);
  dst_range.push_back(c0);
  return dst_range;
}
}  // namespace trans
}  // namespace mindspore
