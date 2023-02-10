/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#ifndef AICPU_UTILS_KERNEL_UTIL_H_
#define AICPU_UTILS_KERNEL_UTIL_H_

#include <climits>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "cpu_kernel/inc/cpu_context.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"
#include "cpu_kernel/common/status.h"

namespace aicpu {
constexpr uint32_t kResvCpuNum = 2;
constexpr uint32_t kThreadNum = 32;
constexpr uint32_t kFirstInputIndex = 0;
constexpr uint32_t kSecondInputIndex = 1;
constexpr uint32_t kThirdInputIndex = 2;
constexpr uint32_t kFirstOutputIndex = 0;
constexpr uint32_t kSecondOutputIndex = 1;
constexpr uint32_t kDynamicInput = -1;
constexpr uint32_t kDynamicOutput = -2;
constexpr uint64_t kEigenAlignmentBytes = 16;

constexpr uint64_t kFormatNCHWIndexN = 0;
constexpr uint64_t kFormatNCHWIndexC = 1;
constexpr uint64_t kFormatNCHWIndexH = 2;
constexpr uint64_t kFormatNCHWIndexW = 3;

constexpr uint64_t kFormatCHWIndexC = 0;
constexpr uint64_t kFormatCHWIndexH = 1;
constexpr uint64_t kFormatCHWIndexW = 2;

constexpr uint64_t kFormatNHWCIndexN = 0;
constexpr uint64_t kFormatNHWCIndexH = 1;
constexpr uint64_t kFormatNHWCIndexW = 2;
constexpr uint64_t kFormatNHWCIndexC = 3;

constexpr uint64_t kFormatHWCIndexH = 0;
constexpr uint64_t kFormatHWCIndexW = 1;
constexpr uint64_t kFormatHWCIndexC = 2;

const size_t INPUT_NUM0 = 0;
const size_t INPUT_NUM1 = 1;
const size_t INPUT_NUM2 = 2;
const size_t INPUT_NUM3 = 3;
const size_t INPUT_NUM4 = 4;
const size_t INPUT_NUM5 = 5;
const size_t INPUT_NUM6 = 6;
const size_t INPUT_NUM7 = 7;
const size_t INPUT_NUM8 = 8;
const size_t INPUT_NUM9 = 9;
const size_t INPUT_NUM32 = 32;
/*
 * str cat util function
 * param[in] params need concat to string
 * return concatted string
 */
template <typename T>
std::string ConcatString(T arg) {
  std::ostringstream oss;
  oss << arg;
  return oss.str();
}

template <typename T, typename... Ts>
std::string ConcatString(T arg, Ts... arg_left) {
  std::ostringstream oss;
  oss << arg;
  oss << ConcatString(arg_left...);
  return oss.str();
}

/**
 * @brief get debug string of vector
 * @param values values in vector
 * @return string of values
 */
template <typename T>
inline std::string VectorToString(const std::vector<T> &values) {
  std::stringstream ss;
  for (auto iter = values.begin(); iter != values.end(); ++iter) {
    ss << *iter;
    if (iter != values.end() - 1) {
      ss << ", ";
    }
  }
  return ss.str();
}

template <typename T>
std::string FmtToStr(const T &t) {
  std::string fmt;
  std::stringstream st;
  st << "[" << t << "]";
  fmt = st.str();
  return fmt;
}

std::string FormatToSerialString(Format format);

/**
 * Get primary-format from format,
 * in bits field:
 * ------------------------------------------
 * |  1 byte  |   2 bytes  |     1 byt      |
 * |----------|------------|----------------|
 * | reserved | sub-format | primary-format |
 * ------------------------------------------
 * @param format
 * @return
 */
inline int32_t GetPrimaryFormat(int32_t format) { return static_cast<int32_t>(static_cast<uint32_t>(format) & 0xff); }

inline int32_t GetSubFormat(int32_t format) {
  return static_cast<int32_t>((static_cast<uint32_t>(format) & 0xffff00) >> 8);
}

inline bool HasSubFormat(int32_t format) { return GetSubFormat(format) > 0; }

/**
 * @brief Judge whether tensor is empty
 * @param tensor need judged tensor
 * @return true: is empty tensor, false: isn't empty tensor
 */
bool IsEmptyTensor(Tensor *tensor);

/**
 * @brief multiply two nonnegative int64's
 * @param x mul value x
 * @param y mul value y
 * @param xy product of x and y
 * @return true: normal, false: overflow
 */
inline bool MulWithoutOverflow(const int64_t x, const int64_t y, int64_t &xy) {
  // Multiply in uint64 rather than int64 since signed overflow is undefined.
  // Negative values will wrap around to large unsigned values in the casts
  // (see section 4.7 [conv.integral] of the C++14 standard).
  const uint64_t ux = static_cast<uint64_t>(x);
  const uint64_t uy = static_cast<uint64_t>(y);
  const uint64_t uxy = ux * uy;

  // Check if we overflow uint64, using a cheap check if both inputs are small
  if ((ux | uy) >> 32 != 0) {
    // Ensure nonnegativity.  Note that negative numbers will appear "large"
    // to the unsigned comparisons above.
    if (x < 0 || y < 0) {
      KERNEL_LOG_ERROR("Can't multiply negative numbers.");
      return false;
    }

    // Otherwise, detect overflow using a division
    if (ux != 0 && uxy / ux != uy) {
      return false;
    }
  }

  // Cast back to signed.  Any negative value will signal an error.
  xy = static_cast<int64_t>(uxy);
  return true;
}

/**
 * @brief add two int64's
 * @param x add value x
 * @param y add value y
 * @param sum sum of x and y
 * @return true: normal, false: overflow
 */
inline bool AddWithoutOverflow(const int64_t x, const int64_t y, int64_t &sum) {
  const uint64_t ux = static_cast<uint64_t>(x);
  const uint64_t uy = static_cast<uint64_t>(y);
  const uint64_t usum = ux + uy;
  sum = static_cast<int64_t>(usum);

  return !(((x >= 0) == (y >= 0)) && ((sum >= 0) != (x >= 0)));
}

/**
 * @brief normal check for calculation
 * @param ctx context
 * @return status code
 */
uint32_t NormalMathCheck(CpuKernelContext &ctx);

/**
 * @brief normal check for kernel
 * @param ctx context
 * @param inputs_num num of inputs
 * @param outputs_num num of outputs
 * @return status code
 */
uint32_t NormalCheck(CpuKernelContext &ctx, const uint32_t inputs_num, const uint32_t outputs_num);

/**
 * @brief normal check for kernel
 * @param ctx context
 * @param inputs_num num of inputs
 * @param outputs_num num of outputs
 * @param attr_names names of attrs
 * @return status code
 */
uint32_t NormalCheck(CpuKernelContext &ctx, const uint32_t inputs_num, const uint32_t outputs_num,
                     const std::vector<std::string> &attr_names);

bool IsScalar(const std::vector<int64_t> &shape);

bool IsMatrix(const std::vector<int64_t> &shape);

bool IsVector(const std::vector<int64_t> &shape);

bool IsSquareMatrix(const std::vector<int64_t> &shape);
/**
 * @brief check if addr is aligned
 * @param addr address for check
 * @return true: aligned, false: not aligned
 */
bool AddrAlignedCheck(const void *addr, uint64_t alignment = kEigenAlignmentBytes);

bool IsVectorOrHigher(const std::vector<int64_t> &shape);

/**
 * @brief get data type from string
 * @param dtype_str string of data type
 * @return DataType
 */
DataType DType(std::string dtype_str);

/**
 * @brief get string from data type
 * @param dtype data type
 * @return string of data type
 */
std::string DTypeStr(DataType dtype);

}  // namespace aicpu
#endif
