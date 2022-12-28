/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include <algorithm>
#include <memory>
#include <vector>

#include "cpu_kernel/format_transfer/format_transfer_fractalz_3d.h"

#include "cpu_kernel/format_transfer/format_transfer_utils.h"
#include "cpu_kernel/format_transfer/formats_definitions.h"
#include "utils/kernel_util.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"
#include "securec/include/securec.h"
#include "cpu_kernel/common/status.h"

namespace aicpu {
namespace formats {
namespace {
KernelStatus CheckDataTypeSupport(DataType data_type) {
  return GetSizeByDataType(data_type) > 0 ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

/**
 * FZ represents the weight of convolution,.
 * After the conversion to two-dimensional matrix, the memory arrangement is
 * small n and large Z. If 4D(eg.NCHW) is used to represent convolution kernel,
 * N is width, HWC is height.
 *
 * frac_z_3d axes: (C1 * H* W * D, N1, Ni, C0), which Ni = 16, C0 = 16 / 32, No =
 * Ceil(N / Ni), C1 = Ceil(C / C0)
 * @return
 */

uint32_t TransShapeToFz3DWithGroups(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w, DataType data_type,
                                    std::vector<int64_t> &dst_shape, int64_t groups) {
  auto c0 = GetCubeSizeByDataType(data_type);
  if (c0 < 0) {
    KERNEL_LOG_ERROR("Cube size must greater than or equal to 0");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t cin_ori = c;
  // For this place , groups is not equal to 0, which had been checked in [Transdata] entrance.
  int64_t cout_ori = n / groups;
  if (cin_ori == 0 || cout_ori == 0) {
    KERNEL_LOG_ERROR(
      "Check param Failed, cin_ori, cout_ori must not be equal 0, "
      "and current cin_ori, cout_ori, groups are [%ld] [%ld] [%ld]",
      cin_ori, cout_ori, groups);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t cube_k = GetCubeSizeByDataType(data_type);
  int64_t e_mult = std::min(
    Lcm(Lcm(cin_ori, cube_k) / (cin_ori), Lcm(cout_ori, static_cast<int64_t>(kCubeSize)) / (cout_ori)), groups);
  int64_t cin_opt = Ceil(e_mult * cin_ori, cube_k) * cube_k;
  int64_t c1_dim = cin_opt / cube_k;
  int64_t dim_g = Ceil(groups, e_mult);
  auto n1 = Ceil(cout_ori * e_mult, static_cast<int64_t>(kCubeSize));
  dst_shape.clear();
  dst_shape.push_back(dim_g * c1_dim * d * h * w);
  dst_shape.push_back(n1);
  dst_shape.push_back(kNiSize);
  dst_shape.push_back(cube_k);
  if (!IsShapeValid(dst_shape)) {
    KERNEL_LOG_ERROR("Check shape failed, dst shape [%s]", VectorToString(dst_shape).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t TransShapeNcdhwToFzWithGroups(const std::vector<int64_t> &src_shape, DataType data_type,
                                       std::vector<int64_t> &dst_shape, int64_t groups) {
  if (!CheckShapeValid(src_shape, static_cast<int64_t>(kNcdhwDimsNum))) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto n = src_shape.at(kNcdhwN);
  auto c = src_shape.at(kNcdhwC);
  auto d = src_shape.at(kNcdhwD);
  auto h = src_shape.at(kNcdhwH);
  auto w = src_shape.at(kNcdhwW);
  return TransShapeToFz3DWithGroups(n, c, d, h, w, data_type, dst_shape, groups);
}

uint32_t TransShapeDhwcnToFzWithGroups(const std::vector<int64_t> &src_shape, DataType data_type,
                                       std::vector<int64_t> &dst_shape, int64_t groups) {
  if (!CheckShapeValid(src_shape, static_cast<int64_t>(kDhwcnDimsNum))) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto d = src_shape.at(kDhwcnD);
  auto h = src_shape.at(kDhwcnH);
  auto w = src_shape.at(kDhwcnW);
  auto c = src_shape.at(kDhwcnC);
  auto n = src_shape.at(kDhwcnN);

  return TransShapeToFz3DWithGroups(n, c, d, h, w, data_type, dst_shape, groups);
}

uint32_t TransShapeNdhwcToFzWithGroups(const std::vector<int64_t> &src_shape, DataType data_type,
                                       std::vector<int64_t> &dst_shape, int64_t groups) {
  if (!CheckShapeValid(src_shape, kNdhwcDimsNum)) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto n = src_shape.at(kNdhwcN);
  auto d = src_shape.at(kNdhwcD);
  auto h = src_shape.at(kNdhwcH);
  auto w = src_shape.at(kNdhwcW);
  auto c = src_shape.at(kNdhwcC);

  return TransShapeToFz3DWithGroups(n, c, d, h, w, data_type, dst_shape, groups);
}

// Supporting NCDHW, DHWCN, NDHWC converte to FORMAT_FRACTAL_Z_3D (GDC1HWN1N0C0),
// the final effect achieved is for the data to be distributed diagonally.
// For example: When the input filter format is NCDHW, calculated the Correspondence of
// index between NCDHW and FORMAT_FRACTAL_Z_3D , then Convert the old filter to the new
// filter, and finally added 0 to the position where there is no data.
uint32_t TransFormatWithGroups(const Format &format_5d, const std::vector<int64_t> &shape_5d, const TransArgs &args,
                               TransResult &result, bool reverse) {
  int64_t h_dim = 0;
  int64_t w_dim = 0;
  int64_t c_dim = 0;
  int64_t n_dim = 0;
  int64_t d_dim = 0;
  if (GetFormatDim(d_dim, h_dim, w_dim, c_dim, n_dim, format_5d, shape_5d) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t cin_ori = c_dim;
  // For this place , groups is not equal to 0, which had been checked in [Transdata] entrance.
  int64_t cout_ori = n_dim / args.groups;
  if (CheckDimOri(cin_ori, cout_ori) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  const int64_t cube_k = GetCubeSizeByDataType(args.src_data_type);
  int64_t e_mult = std::min(
    Lcm(Lcm(cin_ori, cube_k) / (cin_ori), Lcm(cout_ori, static_cast<int64_t>(kCubeSize)) / (cout_ori)), args.groups);
  int64_t cin_opt = Ceil(e_mult * cin_ori, cube_k) * cube_k;
  int64_t cout_opt = Ceil(e_mult * cout_ori, static_cast<int64_t>(kCubeSize)) * static_cast<int64_t>(kCubeSize);
  int64_t c1_dim = cin_opt / cube_k;
  int64_t data_size = GetSizeByDataType(args.src_data_type);
  int64_t dst_size = GetItemNumByShape(args.dst_shape) * data_size;
  // The input is empty tensor, we should return success directly.
  if (dst_size == 0) {
    result.length = static_cast<size_t>(dst_size);
    return KERNEL_STATUS_OK;
  }
  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  KERNEL_CHECK_NULLPTR(dst, KERNEL_STATUS_PARAM_INVALID,
                       "Failed to allcoate memory for dst buf [%lld] when trans "
                       "format from [%s] to [%s]",
                       dst_size, FormatToSerialString(args.src_format).c_str(),
                       FormatToSerialString(args.dst_format).c_str())
  (void)memset_s(dst.get(), static_cast<size_t>(dst_size), 0, static_cast<size_t>(dst_size));
  for (int64_t g = 0; g < args.groups; g++) {
    for (int64_t d = 0; d < d_dim; d++) {
      for (int64_t c = 0; c < c_dim; c++) {
        for (int64_t h = 0; h < h_dim; h++) {
          for (int64_t w = 0; w < w_dim; w++) {
            for (int64_t n = 0; n < cout_ori; n++) {
              int64_t e_val = g % e_mult;
              int64_t dst_ci = e_val * cin_ori + c;
              int64_t dst_co = e_val * cout_ori + n;
              int64_t src_co = g * cout_ori + n;
              int64_t temporary = dst_ci % cube_k;
              int64_t index_5d = 0;
              int64_t index_fz = (g / e_mult) * d_dim * c1_dim * h_dim * w_dim * cout_opt * cube_k +
                                 d * c1_dim * h_dim * w_dim * cout_opt * cube_k +
                                 (dst_ci / cube_k) * h_dim * w_dim * cout_opt * cube_k + h * w_dim * cout_opt * cube_k +
                                 w * cout_opt * cube_k + dst_co * cube_k + temporary;
              if (format_5d == FORMAT_DHWCN) {
                index_5d = d * h_dim * w_dim * c_dim * n_dim + h * w_dim * c_dim * n_dim + w * c_dim * n_dim +
                           c * n_dim + src_co;
              } else if (format_5d == FORMAT_NCDHW) {
                index_5d = src_co * c_dim * d_dim * h_dim * w_dim + c * d_dim * h_dim * w_dim + d * h_dim * w_dim +
                           h * w_dim + w;
              } else if (format_5d == FORMAT_NDHWC) {
                index_5d = src_co * d_dim * h_dim * w_dim * c_dim + d * h_dim * w_dim * c_dim + h * w_dim * c_dim +
                           w * c_dim + c;
              }
              if (!reverse) {
                copy_data(args.data, dst, index_5d, index_fz, data_size);
              } else {
                copy_data(args.data, dst, index_fz, index_5d, data_size);
              }
            }
          }
        }
      }
    }
  }
  result.data = dst;
  result.length = static_cast<size_t>(dst_size);
  return KERNEL_STATUS_OK;
}

}  // namespace

uint32_t FormatTransferFractalz3D::TransFormat(const TransArgs &args, TransResult &result) {
  KERNEL_LOG_DEBUG(
    "Begin to trans format from [%s] to [%s], src shape [%s], data type "
    "[%s], dst "
    "shape [%s]",
    FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str(),
    VectorToString(args.src_shape).c_str(), DTypeStr(args.src_data_type).c_str(),
    VectorToString(args.dst_shape).c_str());

  if ((args.groups) == 0) {
    KERNEL_LOG_ERROR("Attr[groups] must not be equal 0");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (((args.src_format == FORMAT_NDHWC) || (args.src_format == FORMAT_DHWCN) || (args.src_format == FORMAT_NCDHW)) &&
      args.dst_format == FORMAT_FRACTAL_Z_3D) {
    std::vector<int64_t> expect_shape;
    auto ret =
      TransShape(args.src_format, args.src_shape, args.src_data_type, args.dst_format, expect_shape, args.groups);
    if (ret != KERNEL_STATUS_OK) {
      return ret;
    }
    if (!IsTransShapeDstCorrect(args, expect_shape)) {
      return KERNEL_STATUS_PARAM_INVALID;
    }
    return TransFormatWithGroups(args.src_format, args.src_shape, args, result, false);
  } else if (((args.dst_format == FORMAT_NDHWC) || (args.dst_format == FORMAT_DHWCN) ||
              (args.dst_format == FORMAT_NCDHW)) &&
             args.src_format == FORMAT_FRACTAL_Z_3D) {
    std::vector<int64_t> expect_input_shape;
    auto ret =
      TransShape(args.dst_format, args.dst_shape, args.src_data_type, args.src_format, expect_input_shape, args.groups);
    if (ret != KERNEL_STATUS_OK) {
      KERNEL_LOG_ERROR("Check dst shape failed, dst shape [%s]", VectorToString(args.dst_shape).c_str());
      return ret;
    }

    if ((!args.src_shape.empty()) && (args.src_shape != expect_input_shape)) {
      KERNEL_LOG_ERROR("Check dst shape failed, dst shape [%s]", VectorToString(args.dst_shape).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }

    return TransFormatWithGroups(args.dst_format, args.dst_shape, args, result, true);
  }
  return KERNEL_STATUS_PARAM_INVALID;
}

uint32_t FormatTransferFractalz3D::TransShape(Format src_format, const std::vector<int64_t> &src_shape,
                                              DataType data_type, Format dst_format, std::vector<int64_t> &dst_shape,
                                              int64_t groups) {
  if (CheckDataTypeSupport(data_type) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (src_format == FORMAT_NDHWC &&
      GetPrimaryFormat(static_cast<int32_t>(dst_format)) == static_cast<int32_t>(FORMAT_FRACTAL_Z_3D)) {
    return TransShapeNdhwcToFzWithGroups(src_shape, data_type, dst_shape, groups);
  }
  if ((src_format == FORMAT_DHWCN) &&
      GetPrimaryFormat(static_cast<int32_t>(dst_format)) == static_cast<int32_t>(FORMAT_FRACTAL_Z_3D)) {
    return TransShapeDhwcnToFzWithGroups(src_shape, data_type, dst_shape, groups);
  }
  if (src_format == FORMAT_NCDHW &&
      GetPrimaryFormat(static_cast<int32_t>(dst_format)) == static_cast<int32_t>(FORMAT_FRACTAL_Z_3D)) {
    return TransShapeNcdhwToFzWithGroups(src_shape, data_type, dst_shape, groups);
  }

  return KERNEL_STATUS_PARAM_INVALID;
}
REGISTER_FORMAT_TRANSFER(FormatTransferFractalz3D, FORMAT_NCDHW, FORMAT_FRACTAL_Z_3D)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalz3D, FORMAT_DHWCN, FORMAT_FRACTAL_Z_3D)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalz3D, FORMAT_NDHWC, FORMAT_FRACTAL_Z_3D)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalz3D, FORMAT_FRACTAL_Z_3D, FORMAT_NCDHW)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalz3D, FORMAT_FRACTAL_Z_3D, FORMAT_DHWCN)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalz3D, FORMAT_FRACTAL_Z_3D, FORMAT_NDHWC)
}  // namespace formats
}  // namespace  aicpu