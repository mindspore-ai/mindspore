/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "cpu_kernel/format_transfer/format_transfer_ndc1hwc0.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpu_kernel/format_transfer/format_transfer_utils.h"
#include "cpu_kernel/format_transfer/formats_definitions.h"
#include "utils/kernel_util.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"
#include "securec/include/securec.h"
#include "cpu_kernel/common/status.h"

namespace aicpu {
namespace formats {
namespace {
std::map<Format, std::string> kFormatTable = {
  {FORMAT_NCDHW, "NCDHW"},
  {FORMAT_NDHWC, "NDHWC"},
};

KernelStatus CheckDataTypeSupport(DataType data_type) {
  return GetSizeByDataType(data_type) > 0 ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

void TransSrcDataToDstData(const TransArgs &args, const std::vector<int64_t> &shape_ndhwc,
                           std::shared_ptr<uint8_t> &dst, int64_t c0, int32_t data_size) {
  const int64_t n = shape_ndhwc[0];
  const int64_t d = shape_ndhwc[1];
  const int64_t h = shape_ndhwc[2];
  const int64_t w = shape_ndhwc[3];
  const int64_t c = shape_ndhwc[4];
  // c0 is definitely a number greater than 0
  const int64_t c1 = ((c - 1) / c0) + 1;
  const int64_t hw = h * w;
  const int64_t dhw = d * hw;
  const int64_t dhwc = dhw * c;
  const int64_t hwc0 = hw * c0;
  const int64_t c1hwc0 = c1 * hwc0;
  const int64_t dc1hwc0 = d * c1hwc0;
  const int64_t ndhwc = n * dhwc;
  int64_t src_index = 0;

  for (int64_t ndhwc_idx = 0; ndhwc_idx < ndhwc; ++ndhwc_idx) {
    const int64_t n_idx = ndhwc_idx / dhwc;
    const int64_t dhw_idx = ndhwc_idx % dhwc / c;
    const int64_t c_idx = ndhwc_idx % c;
    const int64_t dst_index =
      n_idx * dc1hwc0 + (dhw_idx / hw) * c1hwc0 + (c_idx / c0) * hwc0 + (dhw_idx % hw) * c0 + c_idx % c0;
    src_index = n_idx * dhwc + c_idx * dhw + dhw_idx;
    if (args.src_format == FORMAT_NDHWC) {
      src_index = n_idx * dhwc + dhw_idx * c + c_idx;
    }
    uint8_t *dst_data = dst.get() + dst_index * data_size;
    const uint8_t *src_data = args.data + src_index * data_size;
    for (int64_t index = 0; index < data_size; ++index) {
      *dst_data++ = *src_data++;
    }
  }
}

uint32_t TransDstDataToNdc1hwc0(const TransArgs &args, TransResult &result) {
  const int32_t data_size = GetSizeByDataType(args.src_data_type);
  const auto dst_size = GetItemNumByShape(args.dst_shape) * data_size;
  // The input is empty tensor, we should return success directly
  if (dst_size == 0) {
    result.length = 0;
    return KERNEL_STATUS_OK;
  }
  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    KERNEL_LOG_ERROR("Failed to allocate memory for dst buf [%ld] when trans format from [%s] to [%s]", dst_size,
                     FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  errno_t ret = memset_s(dst.get(), static_cast<size_t>(dst_size), 0, static_cast<size_t>(dst_size));
  if (ret != EOK) {
    KERNEL_LOG_ERROR("memset failed, ret is [%d]", ret);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto iter = kFormatTable.find(args.src_format);
  if (iter == kFormatTable.end()) {
    KERNEL_LOG_ERROR("src_format is wrong, now format is [%d]", static_cast<int32_t>(args.src_format));
    return KERNEL_STATUS_PARAM_INVALID;
  }

  std::string cur_format = iter->second;
  size_t n_index = cur_format.find('N');
  size_t d_index = cur_format.find('D');
  size_t h_index = cur_format.find('H');
  size_t w_index = cur_format.find('W');
  size_t c_index = cur_format.find('C');
  std::vector<int64_t> shape_ndhwc;
  shape_ndhwc.push_back(args.src_shape.at(n_index));
  shape_ndhwc.push_back(args.src_shape.at(d_index));
  shape_ndhwc.push_back(args.src_shape.at(h_index));
  shape_ndhwc.push_back(args.src_shape.at(w_index));
  shape_ndhwc.push_back(args.src_shape.at(c_index));
  const int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  if (c0 <= 0) {
    KERNEL_LOG_ERROR("Failed to get c0, c0 is [%ld]", c0);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  TransSrcDataToDstData(args, shape_ndhwc, dst, c0, data_size);

  result.data = dst;
  result.length = static_cast<size_t>(dst_size);

  return KERNEL_STATUS_OK;
}

uint32_t TransShapeToNdc1hwc0(const std::vector<int64_t> &src_shape, const Format &src_format,
                              const DataType &data_type, std::vector<int64_t> &dst_shape) {
  auto iter = kFormatTable.find(src_format);
  if (iter == kFormatTable.end()) {
    KERNEL_LOG_ERROR("src_format is wrong, now format is [%d]", static_cast<int32_t>(src_format));
    return KERNEL_STATUS_PARAM_INVALID;
  }

  std::string cur_format = iter->second;
  size_t n_index = cur_format.find('N');
  size_t d_index = cur_format.find('D');
  size_t h_index = cur_format.find('H');
  size_t w_index = cur_format.find('W');
  size_t c_index = cur_format.find('C');
  const int64_t c0 = GetCubeSizeByDataType(data_type);
  if (c0 <= 0) {
    KERNEL_LOG_ERROR("Failed to get c0, c0 is [%ld]", c0);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (!CheckShapeValid(src_shape, static_cast<int64_t>(cur_format.length()))) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  dst_shape.clear();
  dst_shape.push_back(src_shape.at(n_index));
  dst_shape.push_back(src_shape.at(d_index));
  dst_shape.push_back(Ceil(src_shape.at(c_index), c0));
  dst_shape.push_back(src_shape.at(h_index));
  dst_shape.push_back(src_shape.at(w_index));
  dst_shape.push_back(c0);
  if (!IsShapeValid(dst_shape)) {
    KERNEL_LOG_ERROR("Check shape failed, dst shape [%s]", VectorToString(dst_shape).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}
}  // namespace

uint32_t FormatTransferNdc1hwc0::TransFormat(const TransArgs &args, TransResult &result) {
  KERNEL_LOG_INFO(
    "Begin to trans format from [%s] to [%s], src shape [%s], data type [%s], dst "
    "shape [%s]",
    FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str(),
    VectorToString(args.src_shape).c_str(), DTypeStr(args.src_data_type).c_str(),
    VectorToString(args.dst_shape).c_str());

  std::vector<int64_t> expect_shape;
  auto ret =
    TransShape(args.src_format, args.src_shape, args.src_data_type, args.dst_format, expect_shape, args.groups);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  if (!IsTransShapeDstCorrect(args, expect_shape)) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return TransDstDataToNdc1hwc0(args, result);
}

uint32_t FormatTransferNdc1hwc0::TransShape(Format src_format, const std::vector<int64_t> &src_shape,
                                            DataType data_type, Format dst_format, std::vector<int64_t> &dst_shape,
                                            int64_t groups) {
  (void)dst_format;
  (void)groups;
  if (CheckDataTypeSupport(data_type) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (src_format != FORMAT_NCDHW && src_format != FORMAT_NDHWC) {
    KERNEL_LOG_ERROR("The current format is not supported, src_format is [%s]",
                     FormatToSerialString(src_format).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return TransShapeToNdc1hwc0(src_shape, src_format, data_type, dst_shape);
}
REGISTER_FORMAT_TRANSFER(FormatTransferNdc1hwc0, FORMAT_NCDHW, FORMAT_NDC1HWC0)
REGISTER_FORMAT_TRANSFER(FormatTransferNdc1hwc0, FORMAT_NDHWC, FORMAT_NDC1HWC0)
}  // namespace formats
}  // namespace  aicpu