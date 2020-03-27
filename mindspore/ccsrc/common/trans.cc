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
#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include "./securec.h"
#include "common/utils.h"
#include "device/convert_tensor_utils.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/utils.h"

namespace mindspore {
namespace trans {
const size_t kNchwDims = 4;
const std::map<TypeId, size_t> type_map = {{kNumberTypeBool, 1},    {kNumberTypeInt, 4},     {kNumberTypeInt8, 1},
                                           {kNumberTypeInt16, 2},   {kNumberTypeInt32, 4},   {kNumberTypeInt64, 8},
                                           {kNumberTypeUInt, 4},    {kNumberTypeUInt8, 1},   {kNumberTypeUInt16, 2},
                                           {kNumberTypeUInt32, 4},  {kNumberTypeUInt64, 8},  {kNumberTypeFloat, 4},
                                           {kNumberTypeFloat16, 2}, {kNumberTypeFloat32, 4}, {kNumberTypeFloat64, 8}};

template <typename T>
T Ceil(T n1, T n2) {
  return (n2 != 0) ? (n1 - 1) / n2 + 1 : 0;
}

enum DataTypeTransMode {
  FROM_FLOAT_TO_FLOAT16,
  FROM_FLOAT_TO_INT32,
  FROM_FLOAT16_TO_FLOAT,
  FROM_FLOAT16_TO_INT32,
  FROM_INT32_TO_FLOAT,
  FROM_INT32_TO_FLOAT16,
  FROM_INT32_TO_UINT8,
  FROM_INT32_TO_INT8,
  FROM_UINT8_TO_FLOAT,
  FROM_UINT8_TO_INT32,
  FROM_INT8_TO_FLOAT,
  FROM_INT8_TO_INT32,
  FROM_INT64_TO_INT32,
};

const std::map<std::pair<TypeId, TypeId>, DataTypeTransMode> mode_map{
  {std::pair<TypeId, TypeId>(kNumberTypeFloat32, kNumberTypeFloat16), FROM_FLOAT_TO_FLOAT16},
  {std::pair<TypeId, TypeId>(kNumberTypeFloat32, kNumberTypeInt32), FROM_FLOAT_TO_INT32},
  {std::pair<TypeId, TypeId>(kNumberTypeFloat16, kNumberTypeFloat32), FROM_FLOAT16_TO_FLOAT},
  {std::pair<TypeId, TypeId>(kNumberTypeFloat16, kNumberTypeInt32), FROM_FLOAT16_TO_INT32},
  {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeFloat32), FROM_INT32_TO_FLOAT},
  {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeFloat16), FROM_INT32_TO_FLOAT16},
  {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeUInt8), FROM_INT32_TO_UINT8},
  {std::pair<TypeId, TypeId>(kNumberTypeInt32, kNumberTypeInt8), FROM_INT32_TO_INT8},
  {std::pair<TypeId, TypeId>(kNumberTypeUInt8, kNumberTypeFloat32), FROM_UINT8_TO_FLOAT},
  {std::pair<TypeId, TypeId>(kNumberTypeUInt8, kNumberTypeInt32), FROM_UINT8_TO_INT32},
  {std::pair<TypeId, TypeId>(kNumberTypeInt8, kNumberTypeFloat32), FROM_INT8_TO_FLOAT},
  {std::pair<TypeId, TypeId>(kNumberTypeInt8, kNumberTypeInt32), FROM_INT8_TO_INT32},
  {std::pair<TypeId, TypeId>(kNumberTypeInt64, kNumberTypeInt32), FROM_INT64_TO_INT32}};

template <typename SrcT, typename DstT>
void TransDataSrc2Dst(const TypeIdArgs &args, void *dst, const size_t data_size) {
  for (size_t idx = 0; idx != data_size; idx++) {
    SrcT src_data = static_cast<const SrcT *>(args.data)[idx];
    static_cast<DstT *>(dst)[idx] = static_cast<DstT>(src_data);
  }
}

bool CastKernel(const TypeIdArgs &args, void *dst, const size_t data_size, const DataTypeTransMode mode) {
  switch (mode) {
    case FROM_FLOAT_TO_FLOAT16:
      device::FloatToHalf(dst, args.data, data_size);
      break;
    case FROM_FLOAT16_TO_FLOAT:
      device::HalfToFloat(dst, args.data, data_size);
      break;
    case FROM_FLOAT_TO_INT32:
      TransDataSrc2Dst<float, int32_t>(args, dst, data_size);
      break;
    case FROM_FLOAT16_TO_INT32:
      TransDataSrc2Dst<float16, int32_t>(args, dst, data_size);
      break;
    case FROM_INT32_TO_FLOAT:
      TransDataSrc2Dst<int32_t, float>(args, dst, data_size);
      break;
    case FROM_INT32_TO_INT8:
      TransDataSrc2Dst<int32_t, int8_t>(args, dst, data_size);
      break;
    case FROM_INT32_TO_UINT8:
      TransDataSrc2Dst<int32_t, uint8_t>(args, dst, data_size);
      break;
    case FROM_UINT8_TO_INT32:
      TransDataSrc2Dst<uint8_t, int32_t>(args, dst, data_size);
      break;
    case FROM_UINT8_TO_FLOAT:
      TransDataSrc2Dst<uint8_t, float>(args, dst, data_size);
      break;
    case FROM_INT8_TO_FLOAT:
      TransDataSrc2Dst<int8_t, float>(args, dst, data_size);
      break;
    case FROM_INT8_TO_INT32:
      TransDataSrc2Dst<int8_t, int32_t>(args, dst, data_size);
      break;
    case FROM_INT64_TO_INT32:
      TransDataSrc2Dst<int64_t, int32_t>(args, dst, data_size);
      break;
    default:
      MS_LOG(ERROR) << "unsupported datatype trans";
      return false;
  }
  return true;
}

size_t CubeSizeByType(const TypeId data_type) {
  const size_t default_error = 0;
  auto dt_size = TypeIdSize(data_type);
  if (dt_size < 1) {
    MS_LOG(ERROR) << "illegal dtype.";
    return default_error;
  } else if (dt_size == 1) {
    return kCubeSize * 2;
  }
  return kCubeSize;
}

size_t ShapeSize(const std::vector<size_t> &shape) {
  size_t product = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  return product;
}

size_t TypeIdSize(const TypeId data_type) {
  const size_t unsupport_type_error = 0;
  auto iter = type_map.find(data_type);
  if (iter != type_map.end()) {
    return iter->second;
  }
  return unsupport_type_error;
}

std::vector<size_t> TransShapeTo4d(const std::vector<size_t> &shape) {
  std::vector<size_t> shape_4d(4, 1);
  switch (shape.size()) {
    case 0:
      break;
    case 1:
      shape_4d[1] = shape[0];
      break;
    case 2:
      shape_4d[0] = shape[0];
      shape_4d[1] = shape[1];
      break;
    case 3:
      MS_LOG(EXCEPTION) << "Unexpected shape size = 3,it should has a default format";
    case 4:
      for (size_t i = 0; i < 4; ++i) {
        shape_4d[i] = shape[i];
      }
      break;
    default:
      MS_LOG(EXCEPTION) << "Unexpeted shape size = " << shape.size();
  }
  return shape_4d;
}

std::vector<size_t> TransShapeToDevice(const std::vector<size_t> &shape, const std::string &format) {
  std::vector<size_t> device_shape;
  if (format == kOpFormat_FRAC_NZ) {
    if (shape.size() < 2) {
      MS_EXCEPTION(NotSupportError) << "format " << format << " is not support shape " << shape.size();
    }
    if (shape.size() > 2) {
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
  if (shape.size() != 4) {
    MS_LOG(EXCEPTION) << "shape_4d size should be 4";
  }
  if (format == kOpFormat_NC1HWC0) {
    size_t C1 = (shape[1] + kCubeSize - 1) / kCubeSize;
    size_t C0 = kCubeSize;
    device_shape.push_back(shape[0]);
    device_shape.push_back(C1);
    device_shape.push_back(shape[2]);
    device_shape.push_back(shape[3]);
    device_shape.push_back(C0);
    return device_shape;
  } else if (format == kOpFormat_FRAC_Z) {
    size_t cout16 = ((shape[0] + kCubeSize - 1) / kCubeSize) * kCubeSize;
    size_t cin16 = ((shape[1] + kCubeSize - 1) / kCubeSize) * kCubeSize;
    device_shape.push_back(shape[2] * shape[3] * cin16 / kCubeSize);
    device_shape.push_back(cout16 / kCubeSize);
    device_shape.push_back(kCubeSize);
    device_shape.push_back(kCubeSize);
    return device_shape;
  } else if (format == kOpFormat_NHWC) {
    device_shape.push_back(shape[0]);
    device_shape.push_back(shape[2]);
    device_shape.push_back(shape[3]);
    device_shape.push_back(shape[1]);
    return device_shape;
  } else if (format == kOpFormat_NCHW) {
    return shape;
  } else if (format == kOpFormat_HWCN) {
    return {shape[2], shape[3], shape[1], shape[0]};
  }
  MS_LOG(EXCEPTION) << "Unexpected format[" << format << "]";
}

bool TransDataType(const TypeIdArgs &args, void *result) {
  MS_LOG(DEBUG) << "begin trans datatype from " << TypeIdLabel(args.host_data_type) << " to "
                << TypeIdLabel(args.device_data_type);
  MS_EXCEPTION_IF_NULL(result);
  std::pair<TypeId, TypeId> type_info(args.host_data_type, args.device_data_type);
  auto iter = mode_map.find(type_info);
  if (iter == mode_map.end()) {
    MS_LOG(ERROR) << "unsupported datatype trans. src_type :" << TypeIdLabel(args.host_data_type)
                  << ", dst_type:" << TypeIdLabel(args.device_data_type);
    return false;
  }
  auto trans_mode = iter->second;
  auto type_size = TypeIdSize(args.device_data_type);
  if (type_size < 1) {
    MS_LOG(ERROR) << "invalid host data type.";
    return false;
  }
  if (args.host_shape_size < 1) {
    MS_LOG(ERROR) << "invalid host data size.";
    return false;
  }
  if (!CastKernel(args, result, args.host_shape_size, trans_mode)) {
    MS_LOG(ERROR) << "failed to trans datatype..";
    return false;
  }
  return true;
}

bool TransFormat(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "start trans format.";
  if (TypeIdSize(args.src_data_type) < 1) {
    MS_LOG(ERROR) << "invalid datatype..";
    return false;
  }
  if ((args.host_format == kOpFormat_NCHW || args.host_format == kOpFormat_ND) &&
      args.device_format == kOpFormat_FRAC_Z) {
    return NchwToFracZ(args, result);
  } else if (args.device_format == kOpFormat_FRAC_NZ) {
    return NchwToFracNz(args, result);
  } else if (args.device_format == kOpFormat_NC1HWC0) {
    return NchwToNc1hwc0(args, result);
  }
  return true;
}

bool TransFormatFromDeviceToHost(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "start trans format.";
  if (TypeIdSize(args.src_data_type) < 1) {
    MS_LOG(ERROR) << "invalid datatype..";
    return false;
  }
  if ((args.host_format == kOpFormat_NCHW || args.host_format == kOpFormat_ND) &&
      args.device_format == kOpFormat_FRAC_Z) {
    return FracZToNchw(args, result);
  } else if (args.device_format == kOpFormat_FRAC_NZ) {
    return FracNzToNchw(args, result);
  } else if (args.device_format == kOpFormat_NC1HWC0) {
    return Nc1hwc0ToNchw(args, result);
  }
  return true;
}

bool NchwToFracZ(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "trans format from nchw to frac_z";
  MS_EXCEPTION_IF_NULL(result);
  if (args.host_shape.size() != kNchwDims) {
    MS_LOG(ERROR) << "invalid host shape, host shape dims:" << args.host_shape.size() << ", expect dims:" << kNchwDims;
    return false;
  }
  size_t size = TypeIdSize(args.src_data_type);
  if (size < 1) {
    MS_LOG(ERROR) << "illegal dtype.";
    return false;
  }
  auto n = args.host_shape[0];
  auto c = args.host_shape[1];
  auto h = args.host_shape[2];
  auto w = args.host_shape[3];

  size_t c0 = CubeSizeByType(args.src_data_type);
  if (c0 < 1) {
    MS_LOG(ERROR) << "illegal dtype.";
    return false;
  }
  size_t c1 = Ceil(c, c0);
  size_t hw = h * w;
  size_t chw = c * hw;
  size_t hwc0 = hw * c0;
  size_t nchw = n * chw;

  size_t hf_cnt = Ceil(n, kCubeSize);
  size_t vf_cnt = c1 * hw;
  size_t fractal_ele_cnt = c0 * kCubeSize;
  size_t total_ele_cnt = hf_cnt * vf_cnt * fractal_ele_cnt;

  size_t dst_size = total_ele_cnt * size;
  if (dst_size != args.device_size) {
    MS_LOG(ERROR) << "illegal total data size."
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
          auto src_offset = src_row_offset + chw * col;

          auto need_pad_zero = src_ni >= n || src_offset >= nchw || src_ci >= c;
          auto idx = gfi * fractal_ele_cnt + col * c0 + row;
          auto offset = idx * size;
          auto protected_size = dst_size - offset < static_cast<size_t>(SECUREC_MEM_MAX_LEN)
                                  ? dst_size - offset
                                  : static_cast<size_t>(SECUREC_MEM_MAX_LEN);
          errno_t ret;
          if (need_pad_zero) {
            ret = memset_s(static_cast<uint8_t *>(result) + offset, protected_size, 0, size);
          } else {
            ret = memcpy_s(static_cast<uint8_t *>(result) + offset, protected_size,
                           static_cast<uint8_t const *>(args.data) + src_offset * size, size);
          }
          if (ret != 0) {
            MS_LOG(ERROR) << "Failed to operate the dst memory error-code " << ret;
            return false;
          }
        }
      }
    }
  }
  return true;
}

bool FracZToNchw(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "trans format from frac_z to nchw";
  MS_EXCEPTION_IF_NULL(result);
  if (args.host_shape.size() != kNchwDims) {
    MS_LOG(ERROR) << "invalid host shape, host shape dims:" << args.host_shape.size() << ", expect dims:" << kNchwDims;
    return false;
  }
  size_t size = TypeIdSize(args.src_data_type);
  if (size < 1) {
    MS_LOG(ERROR) << "illegal dtype.";
    return false;
  }
  size_t total_size = ShapeSize(args.device_shape) * size;
  if (total_size != args.device_size) {
    MS_LOG(ERROR) << "illegal total data size, total_size:" << total_size << ", device_size:" << args.device_size;
    return false;
  }

  auto n0 = args.device_shape.at(1);
  auto ni = args.device_shape.at(2);
  auto c0 = args.device_shape.at(3);

  auto n = args.host_shape[0];
  auto c = args.host_shape[1];
  auto h = args.host_shape[2];
  auto w = args.host_shape[3];

  size_t nc = ni * n0;
  size_t ncc0 = nc * c0;
  size_t wncc0 = w * ncc0;
  size_t hwncc0 = h * wncc0;
  size_t hw = h * w;
  size_t chw = c * hw;

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
          auto src_offset = src_idx * size;
          auto dst_offset = dst_idx * size;
          auto protected_size = total_size - dst_offset < static_cast<size_t>(SECUREC_MEM_MAX_LEN)
                                  ? total_size - dst_offset
                                  : static_cast<size_t>(SECUREC_MEM_MAX_LEN);
          auto ret = memcpy_s(static_cast<uint8_t *>(result) + dst_offset, protected_size,
                              static_cast<uint8_t const *>(args.data) + src_offset, size);
          if (ret != EOK) {
            MS_LOG(ERROR) << "Failed to operate the dst memory error-code " << ret;
            return false;
          }
        }
      }
    }
  }
  return true;
}

bool TransShapeToNz(const std::vector<size_t> &host_shape, std::vector<size_t> *hw_shape) {
  MS_EXCEPTION_IF_NULL(hw_shape);
  if (host_shape.empty()) {
    MS_LOG(ERROR) << "size of vector is 0.";
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
        MS_LOG(ERROR) << "illegal size.";
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
  MS_LOG(DEBUG) << "trans format from nchw to frac_nz.";
  MS_EXCEPTION_IF_NULL(result);
  std::vector<size_t> hw_shape;
  if (!TransShapeToNz(args.host_shape, &hw_shape)) {
    MS_LOG(ERROR) << "trans shape failed..";
    return false;
  }
  if (hw_shape.size() < 3 || args.device_shape.size() < 4) {
    MS_LOG(ERROR) << "invalid shape size.";
    return false;
  }
  auto size = TypeIdSize(args.src_data_type);
  if (size < 1) {
    MS_LOG(ERROR) << "illegal dtype";
    return false;
  }

  auto dst_size = ShapeSize(args.device_shape) * size;
  if (dst_size != args.device_size) {
    MS_LOG(ERROR) << "illegal total data size, total_size:" << dst_size << ", device_size:" << args.device_size;
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
        size_t dst_offset = (h1h0_head + w1_idx * h1h0w0) * size;
        size_t src_offset = (src_h_head + w1_idx * w0) * size;
        auto protected_size = dst_size - dst_offset < static_cast<size_t>(SECUREC_MEM_MAX_LEN)
                                ? dst_size - dst_offset
                                : static_cast<size_t>(SECUREC_MEM_MAX_LEN);
        auto cp_ret = memcpy_s(static_cast<uint8_t *>(result) + dst_offset, protected_size,
                               static_cast<uint8_t const *>(args.data) + src_offset, size * w0);
        if (cp_ret != EOK) {
          MS_LOG(ERROR) << "Failed to operate the dst memory, error-code " << cp_ret;
          return false;
        }
      }
      auto w1_head = num_w1 * w0;
      for (size_t w0_idx = 0; w1_head + w0_idx < w; w0_idx++) {
        auto src_w_idx = w1_head + w0_idx;
        size_t dst_offset = (h1h0_head + num_w1 * h1h0w0 + w0_idx) * size;
        size_t src_offset = (src_h_head + src_w_idx) * size;
        auto protected_size = dst_size - dst_offset < static_cast<size_t>(SECUREC_MEM_MAX_LEN)
                                ? dst_size - dst_offset
                                : static_cast<size_t>(SECUREC_MEM_MAX_LEN);
        auto cp_ret = memcpy_s(static_cast<uint8_t *>(result) + dst_offset, protected_size,
                               static_cast<uint8_t const *>(args.data) + src_offset, size);
        if (cp_ret != EOK) {
          MS_LOG(ERROR) << "Failed to operate the dst memory error-code " << cp_ret;
          return false;
        }
      }
    }
  }
  return true;
}

bool FracNzToNchw(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "trans format from frac_nz to nchw";
  MS_EXCEPTION_IF_NULL(result);
  std::vector<size_t> hw_shape;
  if (!TransShapeToNz(args.host_shape, &hw_shape)) {
    MS_LOG(ERROR) << "trans shape failed..";
    return false;
  }
  if (hw_shape.size() < 3 || args.device_shape.size() < 4) {
    MS_LOG(ERROR) << "invalid shape size.";
    return false;
  }
  auto size = TypeIdSize(args.src_data_type);
  if (size < 1) {
    MS_LOG(ERROR) << "illegal dtype";
    return false;
  }

  auto dst_size = ShapeSize(args.device_shape) * size;
  if (dst_size != args.device_size) {
    MS_LOG(ERROR) << "illegal total data size, total_size:" << dst_size << ", device_size:" << args.device_size;
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
        size_t src_offset = (h1h0_head + w1_idx * h1h0w0) * size;
        size_t dst_offset = (src_h_head + w1_idx * w0) * size;
        auto protected_size = dst_size - dst_offset < static_cast<size_t>(SECUREC_MEM_MAX_LEN)
                                ? dst_size - dst_offset
                                : static_cast<size_t>(SECUREC_MEM_MAX_LEN);
        auto cp_ret = memcpy_s(static_cast<uint8_t *>(result) + dst_offset, protected_size,
                               static_cast<uint8_t const *>(args.data) + src_offset, size * w0);
        if (cp_ret != EOK) {
          MS_LOG(ERROR) << "Failed to operate the dst memory, error-code " << cp_ret;
          return false;
        }
      }
      auto w1_head = num_w1 * w0;
      for (size_t w0_idx = 0; w1_head + w0_idx < w; w0_idx++) {
        auto src_w_idx = w1_head + w0_idx;
        size_t src_offset = (h1h0_head + num_w1 * h1h0w0 + w0_idx) * size;
        size_t dst_offset = (src_h_head + src_w_idx) * size;
        auto protected_size = dst_size - dst_offset < static_cast<size_t>(SECUREC_MEM_MAX_LEN)
                                ? dst_size - dst_offset
                                : static_cast<size_t>(SECUREC_MEM_MAX_LEN);
        auto cp_ret = memcpy_s(static_cast<uint8_t *>(result) + dst_offset, protected_size,
                               static_cast<uint8_t const *>(args.data) + src_offset, size);
        if (cp_ret != EOK) {
          MS_LOG(ERROR) << "Failed to operate the dst memory error-code " << cp_ret;
          return false;
        }
      }
    }
  }
  return true;
}

bool NchwToNc1hwc0(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "trans format from nchw to Nc1h1wc0";
  MS_EXCEPTION_IF_NULL(result);
  if (args.host_shape.size() != kNchwDims) {
    MS_LOG(ERROR) << "invalid host shape, host shape dims:" << args.host_shape.size() << ", expect dims:" << kNchwDims;
    return false;
  }
  size_t size = TypeIdSize(args.src_data_type);
  if (size < 1) {
    MS_LOG(ERROR) << "illegal dtype.";
    return false;
  }
  auto total_size = ShapeSize(args.device_shape) * size;
  if (total_size != args.device_size) {
    MS_LOG(ERROR) << "illegal total data size, total_size:" << total_size << ", device_size:" << args.device_size;
    return false;
  }

  auto n = args.host_shape[0];
  auto c = args.host_shape[1];
  auto h = args.host_shape[2];
  auto w = args.host_shape[3];
  size_t c0 = CubeSizeByType(args.src_data_type);
  if (c0 < 1) {
    MS_LOG(ERROR) << "illegal dtype.";
    return false;
  }
  size_t c1 = Ceil(c, c0);
  size_t hw = h * w;
  size_t chw = c * hw;
  size_t c1hwc0 = c1 * hw * c0;
  size_t wc0 = w * c0;

  for (size_t n_idx = 0; n_idx < n; n_idx++) {
    size_t n_head_addr = n_idx * c1hwc0;
    for (size_t c1_idx = 0; c1_idx < c1; c1_idx++) {
      size_t c1_head_addr = n_head_addr + c1_idx * hw * c0;
      for (size_t h_idx = 0; h_idx < h; h_idx++) {
        size_t h_head_addr = c1_head_addr + h_idx * wc0;
        for (size_t w_idx = 0; w_idx < w; w_idx++) {
          size_t w_head_addr = h_head_addr + w_idx * c0;
          for (size_t c0_idx = 0; c0_idx < c0; c0_idx++) {
            size_t dst_index = c0_idx + w_head_addr;
            size_t dst_offset = dst_index * size;
            auto protected_size = total_size - dst_offset < static_cast<size_t>(SECUREC_MEM_MAX_LEN)
                                    ? total_size - dst_offset
                                    : static_cast<size_t>(SECUREC_MEM_MAX_LEN);
            size_t c_idx = c0_idx + c1_idx * c0;
            size_t src_idx = n_idx * chw + c_idx * hw + h_idx * w + w_idx;
            auto src_offset = src_idx * size;

            if (c_idx < c) {
              auto ret = memcpy_s(static_cast<uint8_t *>(result) + dst_offset, protected_size,
                                  static_cast<uint8_t const *>(args.data) + src_offset, size);
              if (ret != EOK) {
                MS_LOG(ERROR) << "Failed to operate the dst memory error-code " << ret;
                return false;
              }
            } else {
              auto ret = memset_s(static_cast<uint8_t *>(result) + dst_offset, protected_size, 0, size);
              if (ret != EOK) {
                MS_LOG(ERROR) << "Failed to operate the dst memory error-code " << ret;
                return false;
              }
            }
          }
        }
      }
    }
  }
  return true;
}

bool Nc1hwc0ToNchw(const FormatArgs &args, void *result) {
  MS_LOG(DEBUG) << "trans format from nc1h1wc0 to nchw";
  MS_EXCEPTION_IF_NULL(result);
  if (args.host_shape.size() != kNchwDims) {
    MS_LOG(ERROR) << "invalid host shape, host shape dims:" << args.host_shape.size() << ", expect dims:" << kNchwDims;
    return false;
  }
  size_t size = TypeIdSize(args.src_data_type);
  if (size < 1) {
    MS_LOG(ERROR) << "illegal dtype.";
    return false;
  }
  size_t total_size = ShapeSize(args.device_shape) * size;
  if (total_size != args.device_size) {
    MS_LOG(ERROR) << "illegal total data size, total_size:" << total_size << ", device_size:" << args.device_size;
    return false;
  }

  auto n = args.host_shape[0];
  auto c = args.host_shape[1];
  auto h = args.host_shape[2];
  auto w = args.host_shape[3];
  auto c1 = args.device_shape[1];
  auto c0 = args.device_shape[4];

  size_t hw = h * w;
  size_t chw = c * hw;
  size_t wc0 = w * c0;
  size_t hwc0 = h * wc0;
  size_t c1hwc0 = c1 * hwc0;

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
          auto src_offset = src_idx * size;
          auto dst_offset = dst_idx * size;
          auto protected_size = total_size - dst_offset < static_cast<size_t>(SECUREC_MEM_MAX_LEN)
                                  ? total_size - dst_offset
                                  : static_cast<size_t>(SECUREC_MEM_MAX_LEN);
          auto ret = memcpy_s(static_cast<uint8_t *>(result) + dst_offset, protected_size,
                              static_cast<uint8_t const *>(args.data) + src_offset, size);
          if (ret != EOK) {
            MS_LOG(ERROR) << "Failed to operate the dst memory error-code " << ret;
            return false;
          }
        }
      }
    }
  }
  return true;
}
}  // namespace trans
}  // namespace mindspore
