/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

#include "cpu_kernel/format_transfer/format_transfer_utils.h"

#include <functional>
#include <memory>
#include <numeric>

#include "cpu_kernel/format_transfer/formats_definitions.h"
#include "utils/kernel_util.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"

namespace aicpu {
namespace formats {
bool IsShapeValid(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return false;
  }
  int64_t num = 1;
  for (auto dim : shape) {
    if (dim < 0) {
      std::string error = "Invalid negative dims in the shape " + FmtToStr(VectorToString(shape));
      KERNEL_LOG_ERROR("%s", error.c_str());
      return false;
    }
    if (dim != 0 && kShapeItemNumMAX / dim < num) {
      std::string error = "Shape overflow, the total count should be less than " + FmtToStr(kShapeItemNumMAX);
      KERNEL_LOG_ERROR("%s", error.c_str());
      return false;
    }
    num *= dim;
  }
  return true;
}

bool CheckShapeValid(const std::vector<int64_t> &shape, const int64_t expect_dims) {
  if (expect_dims <= 0 || shape.size() != static_cast<size_t>(expect_dims)) {
    std::string error = "Invalid shape, dims num " + FmtToStr(shape.size()) + ", expect " + FmtToStr(expect_dims);
    KERNEL_LOG_ERROR("%s", error.c_str());
    return false;
  }
  return IsShapeValid(shape);
}

int64_t GetCubeSizeByDataType(DataType data_type) {
  // Current cube does not support 4 bytes and longer data
  auto size = GetSizeByDataType(data_type);
  if (size <= 0) {
    std::string error = "Failed to get cube size, the data type " + FmtToStr(DTypeStr(data_type)) + " is invalid";
    KERNEL_LOG_ERROR("%s", error.c_str());
    return -1;
  } else if (size == 1) {
    return kCubeSize * 2;  // 32 bytes cube size
  } else {
    return kCubeSize;
  }
}

bool IsTransShapeSrcCorrect(const TransArgs &args, std::vector<int64_t> &expect_shape) {
  if (args.src_shape != expect_shape) {
    std::string error = "Failed to trans format from" + FmtToStr(FormatToSerialString(args.src_format)) + " to " +
                        FmtToStr(FormatToSerialString(args.dst_format)) + ", invalid relationship between src shape " +
                        FmtToStr(VectorToString(args.src_shape)) + " and dst " +
                        FmtToStr(VectorToString(args.dst_shape));
    KERNEL_LOG_ERROR("%s", error.c_str());
    return false;
  }
  return true;
}

bool IsTransShapeDstCorrect(const TransArgs &args, std::vector<int64_t> &expect_shape) {
  if (!args.dst_shape.empty() && args.dst_shape != expect_shape) {
    std::string error = "Failed to trans format from " + FmtToStr(FormatToSerialString(args.src_format)) + " to " +
                        FmtToStr(FormatToSerialString(args.dst_format)) + ", the dst shape" +
                        FmtToStr(VectorToString(args.dst_shape)) + " is invalid, expect" +
                        FmtToStr(VectorToString(expect_shape));
    KERNEL_LOG_ERROR("%s", error.c_str());
    return false;
  }
  return true;
}

int64_t GetItemNumByShape(const std::vector<int64_t> &shape) {
  // shape will not be greater than INT_MAX
  int64_t num = 1;
  for (auto dim : shape) {
    num *= dim;
  }
  return num;
}

uint32_t TransFormat(const TransArgs &args, TransResult &result) {
  auto transfer = BuildFormatTransfer(args);
  if (transfer == nullptr) {
    std::string error = "Failed to trans data from format " + FmtToStr(FormatToSerialString(args.src_format)) + " to " +
                        FmtToStr(FormatToSerialString(args.dst_format));
    KERNEL_LOG_WARN("%s", error.c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto src_shape_size = GetItemNumByShape(args.src_shape);
  if (args.data == nullptr && src_shape_size != 0) {
    KERNEL_LOG_WARN("Invalid input null data");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return transfer->TransFormat(args, result);
}

int64_t Measure(int64_t x, int64_t y) {
  int64_t z = y;
  while (x % y != 0) {
    z = x % y;
    x = y;
    y = z;
  }
  return z;
}
// least common multiple
int64_t Lcm(int64_t a, int64_t b) {
  if (b == 0) {
    return -1;
  }
  int64_t temp = (a * b) / (Measure(a, b));
  return temp;
}

void copy_data(const uint8_t *input_data, std::shared_ptr<uint8_t> dst, int64_t src_index, int64_t dst_index,
               int64_t data_size) {
  char *dst_data = reinterpret_cast<char *>(dst.get() + dst_index * data_size);
  const char *src_data = reinterpret_cast<const char *>(input_data + src_index * data_size);
  for (int64_t index = 0; index < data_size; index++) {
    *dst_data++ = *src_data++;
  }
}

KernelStatus CheckDimOri(int64_t cin_ori, int64_t cout_ori) {
  if (cin_ori == 0 || cout_ori == 0) {
    KERNEL_LOG_ERROR(
      "Cin_ori, cout_ori must not be equal 0, and current cin_ori is [%ld], "
      "cout_ori is [%ld]",
      cin_ori, cout_ori);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

KernelStatus GetFormatDim(int64_t &d_dim, int64_t &h_dim, int64_t &w_dim, int64_t &c_dim, int64_t &n_dim,
                          const Format &input_format, const std::vector<int64_t> &dims) {
  if (input_format == FORMAT_NCDHW) {
    n_dim = dims[kNcdhwN];
    c_dim = dims[kNcdhwC];
    d_dim = dims[kNcdhwD];
    h_dim = dims[kNcdhwH];
    w_dim = dims[kNcdhwW];
  } else if (input_format == FORMAT_DHWCN) {
    d_dim = dims[kDhwcnD];
    h_dim = dims[kDhwcnH];
    w_dim = dims[kDhwcnW];
    c_dim = dims[kDhwcnC];
    n_dim = dims[kDhwcnN];
  } else if (input_format == FORMAT_NDHWC) {
    n_dim = dims[kNdhwcN];
    d_dim = dims[kNdhwcD];
    h_dim = dims[kNdhwcH];
    w_dim = dims[kNdhwcW];
    c_dim = dims[kNdhwcC];
  } else if (input_format == FORMAT_NHWC) {
    n_dim = dims[kNhwcN];
    h_dim = dims[kNhwcH];
    d_dim = 1;
    w_dim = dims[kNhwcW];
    c_dim = dims[kNhwcC];
  } else if (input_format == FORMAT_NCHW) {
    n_dim = dims[kNchwN];
    c_dim = dims[kNchwC];
    h_dim = dims[kNchwH];
    w_dim = dims[kNchwW];
    d_dim = 1;
  } else if (input_format == FORMAT_HWCN) {
    h_dim = dims[kHwcnH];
    w_dim = dims[kHwcnW];
    c_dim = dims[kHwcnC];
    n_dim = dims[kHwcnN];
    d_dim = 1;
  } else {
    KERNEL_LOG_WARN(
      "Format is not FORMAT_DHWCN or FORMAT_NDHWC or FORMAT_NCDHW or "
      "FORMAT_NHWC or FORMAT_NCHW or FORMAT_HWCN, current input "
      "format is [%d]",
      static_cast<int32_t>(input_format));
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
}  // namespace formats
}  // namespace aicpu
