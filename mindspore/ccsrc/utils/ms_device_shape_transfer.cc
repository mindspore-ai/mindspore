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
#include "utils/ms_device_shape_transfer.h"
#include <functional>
#include <numeric>
#include <utility>
#include <algorithm>
namespace mindspore {
namespace TEMP {
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
bool HasShapeDynamic(const ShapeVector &shape_list) {
  return std::any_of(shape_list.begin(), shape_list.end(), [](int64_t v) { return v == abstract::Shape::SHP_ANY; });
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
  if (shape.size() != kNchwDims) {
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
}  // namespace

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

bool DataTypeTransfer::CastKernel(const TypeIdArgs &args, void *dst, int64_t data_size, DataTypeTransMode mode) {
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
    device::FloatToHalf(dst, args.data, data_size);
    return true;
  } else if (mode == DataTypeTransMode::FROM_FLOAT16_TO_FLOAT) {
    device::HalfToFloat(dst, args.data, data_size);
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

bool DataTypeTransfer::TransDataType(const TypeIdArgs &args, void *result) {
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
                                                        bool is_output) {
  auto dev_shape = GetFixedDeviceShape(shape, node, index, is_output);
  if (dev_shape.has_value()) {
    return dev_shape.value();
  }
  int64_t groups = 1;
  if (format == kOpFormat_FRAC_Z) {
    groups = AnfAlgo::GetAttrGroups(node, index);
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
                                                        const ShapeVector &input_hidden_size) {
  return TransCore(shape, format, type, groups, input_hidden_size);
}

std::optional<ShapeVector> DeviceShapeTransfer::GetFixedDeviceShape(const ShapeVector &, const AnfNodePtr &node,
                                                                    size_t index, bool is_output) {
  if (node == nullptr || !node->isa<CNode>()) {
    return {};
  }
  auto attr_name = is_output ? kAttrFixedOutputDeviceShape : kAttrFixedInputDeviceShape;
  auto cnode = node->cast<CNodePtr>();
  if (!AnfAlgo::HasNodeAttr(attr_name, cnode)) {
    return {};
  }

  auto shapes = AnfAlgo::GetNodeAttr<std::vector<ShapeVector>>(cnode, attr_name);
  if (index >= shapes.size()) {
    MS_LOG(INFO) << "Index is out of range, got index: " << index << ", shape size: " << shapes.size();
    return {};
  }
  return std::optional<ShapeVector>(std::move(shapes[index]));
}

ShapeVector DeviceShapeTransfer::TransCore(const ShapeVector &shape, const std::string &format, const TypeId &type,
                                           int64_t groups, const ShapeVector &input_hidden_size) {
  using DeviceShapeTransfer = std::function<ShapeVector(const ShapeVector &, const TypeId &)>;
  const std::map<std::string, DeviceShapeTransfer> device_shape_map = {
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
  if (kNoPaddingFormatSet.find(format) == kNoPaddingFormatSet.end() && format != kOpFormat_FRACTAL_ZN_LSTM &&
      shape.size() < kNchwDims && k3DFormatSet.find(format) == k3DFormatSet.end()) {
    MS_LOG(WARNING) << "Origin shape size is less than 4, should be Padding shape by Default firstly";
    temp_shape = PaddingShapeTo4dDefault(shape);
  }
  if (shape.size() != kNcdhw && k3DFormatSet.find(format) != k3DFormatSet.end()) {
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
    device_shape.push_back(abstract::Shape::SHP_ANY);
  } else {
    auto c1 = (shape[kC] + c0 - 1) / c0;
    device_shape.push_back(shape[kH] * shape[kW] * c1);
  }
  if (shape[kN] == abstract::Shape::SHP_ANY) {
    device_shape.push_back(abstract::Shape::SHP_ANY);
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
  auto c1 = (shape[kC] == abstract::Shape::SHP_ANY) ? abstract::Shape::SHP_ANY : (shape[kC] + c0 - 1) / c0;
  device_shape.push_back(shape[kN]);
  device_shape.push_back(c1);
  device_shape.push_back(shape[kH]);
  device_shape.push_back(shape[kW]);
  device_shape.push_back(c0);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::NDC1HWC0DeviceShape(const ShapeVector &shape, const TypeId &type) {
  if (shape.size() != kNcdhw) {
    MS_LOG(EXCEPTION) << "Check dims failed, expect shape dim 5, but got shape dim : " << shape.size();
  }
  ShapeVector device_shape;
  auto c0 = GetCubeSizeByType(type);
  auto c1 = (shape[1] == abstract::Shape::SHP_ANY) ? abstract::Shape::SHP_ANY : (shape[1] + c0 - 1) / c0;
  device_shape.push_back(shape[N_ncdhw]);
  device_shape.push_back(shape[D_ncdhw]);
  device_shape.push_back(c1);
  device_shape.push_back(shape[H_ncdhw]);
  device_shape.push_back(shape[W_ncdhw]);
  device_shape.push_back(c0);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::FRAC_Z3DDeviceShape(const ShapeVector &shape, const TypeId &type) {
  if (shape.size() != kNcdhw) {
    MS_LOG(EXCEPTION) << "Check dims failed, expect shape dim 5, but got shape dim : " << shape.size();
  }
  ShapeVector device_shape;
  auto c0 = GetCubeSizeByType(type);
  if (HasShapeDynamic({shape[C_ncdhw], shape[D_ncdhw], shape[H_ncdhw], shape[W_ncdhw]})) {
    device_shape.push_back(abstract::Shape::SHP_ANY);
  } else {
    auto c1 = (shape[1] + c0 - 1) / c0;
    device_shape.push_back(shape[D_ncdhw] * c1 * shape[H_ncdhw] * shape[W_ncdhw]);
  }
  auto no = (shape[0] == abstract::Shape::SHP_ANY) ? abstract::Shape::SHP_ANY : (shape[0] + kNiSize - 1) / kNiSize;
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
  if (shape[kC] == abstract::Shape::SHP_ANY) {
    device_shape.push_back(abstract::Shape::SHP_ANY);
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

ShapeVector DeviceShapeTransfer::FRAC_ZC04DeviceShape(const ShapeVector &shape, const TypeId &type) {
  if (!CheckDims(shape)) {
    MS_LOG(EXCEPTION) << "Check dims failed.";
  }
  ShapeVector device_shape;
  const int64_t C04 = 4;
  int64_t first_dim;
  if (HasShapeDynamic({shape[kH], shape[kW]})) {
    first_dim = abstract::Shape::SHP_ANY;
  } else {
    first_dim = DivCeil(C04 * shape[kH] * shape[kW], kCubeSize);
  }
  auto no = (shape[kN] == abstract::Shape::SHP_ANY) ? abstract::Shape::SHP_ANY : DivCeil(shape.at(kN), kCubeSize);
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
  const int64_t C1 = (shape[kC] == abstract::Shape::SHP_ANY) ? abstract::Shape::SHP_ANY : DivCeil(shape.at(kC), C04);
  device_shape.push_back(shape[kN]);
  device_shape.push_back(C1);
  device_shape.push_back(shape[kH]);
  device_shape.push_back(shape[kW]);
  device_shape.push_back(C04);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::NCDHWDeviceShape(const ShapeVector &shape, const TypeId &) {
  if (shape.size() < kNcdhw) {
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
                       [&shape](int64_t n) { return shape[n]; });
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
  int64_t w1 = (w_shape == abstract::Shape::SHP_ANY) ? abstract::Shape::SHP_ANY : (w_shape - 1) / c0 + 1;
  int64_t h1 = (h_shape == abstract::Shape::SHP_ANY) ? abstract::Shape::SHP_ANY : (h_shape - 1) / kCubeSize + 1;
  device_shape.push_back(w1);
  device_shape.push_back(h1);
  device_shape.push_back(kCubeSize);
  device_shape.push_back(c0);
  return device_shape;
}

ShapeVector DeviceShapeTransfer::FRAC_ZN_LSTMDeviceShape(const ShapeVector &shape, const TypeId &type) {
  ShapeVector device_shape;
  const int64_t lstm_ni = 4;
  const int64_t ni = 16;
  int64_t first = abstract::Shape::SHP_ANY;
  int64_t second = abstract::Shape::SHP_ANY;
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
  auto c1_dim = abstract::Shape::SHP_ANY;
  auto g_dim = abstract::Shape::SHP_ANY;
  auto n1 = abstract::Shape::SHP_ANY;
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
    device_shape.push_back(abstract::Shape::SHP_ANY);
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
  if (dim_last2 == abstract::Shape::SHP_ANY) {
    device_shape[shape.size() - kDim2] = abstract::Shape::SHP_ANY;
  } else if (dim_last2 == input_size || dim_last2 == hidden_size) {
    device_shape[shape.size() - kDim2] = DivCeil(dim_last2, NUM16);
  } else if (dim_last2 == input_size + hidden_size) {
    device_shape[shape.size() - kDim2] = DivCeil(input_size, NUM16) + DivCeil(hidden_size, NUM16);
  } else {
    MS_LOG(EXCEPTION) << "The second-last dim value of shape is invalid.";
  }
  if (dim_last1 == abstract::Shape::SHP_ANY) {
    device_shape[shape.size() - kDim1] = abstract::Shape::SHP_ANY;
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
  if (dim_last1 == abstract::Shape::SHP_ANY) {
    device_shape[shape.size() - 1] = abstract::Shape::SHP_ANY;
  } else {
    if (hidden_size <= 0 || dim_last1 % hidden_size != 0) {
      MS_LOG(EXCEPTION) << "Last dim of shape " << shape << " should be multiple of hidden_size " << hidden_size;
    }
    int64_t n_num = shape[shape.size() - 1] / hidden_size;
    device_shape[shape.size() - 1] = n_num * DivCeil(hidden_size, C0) * C0;
  }
  return device_shape;
}

ShapeVector DeviceShapeTransfer::GetAttrInputAndHiddenSize(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  ShapeVector input_hidden_size = {kAlign16, kAlign16};
  if (!node->isa<CNode>()) {
    return input_hidden_size;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!AnfAlgo::HasNodeAttr(kAttrHiddenSize, cnode) || !AnfAlgo::HasNodeAttr(kAttrInputSize, cnode)) {
    MS_LOG(EXCEPTION)
      << "Node with format FRACTAL_ZN_RNN or ND_RNN_BIAS should have hidden_size or input_size attr. Node info:"
      << cnode->DebugString();
  }
  input_hidden_size[0] = AnfAlgo::GetNodeAttr<int64_t>(node, kAttrInputSize);
  input_hidden_size[1] = AnfAlgo::GetNodeAttr<int64_t>(node, kAttrHiddenSize);
  return input_hidden_size;
}
}  // namespace TEMP
}  // namespace mindspore
