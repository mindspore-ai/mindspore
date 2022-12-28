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
#ifndef AICPU_KERNELS_HOST_FORMAT_TRANSFER_REGISTER_FORMAT_TRANSFER_H
#define AICPU_KERNELS_HOST_FORMAT_TRANSFER_REGISTER_FORMAT_TRANSFER_H

#include <functional>
#include <memory>
#include <vector>

#include "cpu_kernel/inc/cpu_types.h"

namespace aicpu {
namespace formats {
struct TransArgs {
  const uint8_t *data;
  // primary format
  Format src_format;
  Format dst_format;
  // For scenes that need to supplement the shape, for example, 5D to 4D
  // It is not possible to convert the format normally if you only get the
  // src_shape, and must get the shape before you mend the shape. So the
  // parameters here need to be passed in both src_shape and dst_shape
  std::vector<int64_t> src_shape;
  std::vector<int64_t> dst_shape;
  DataType src_data_type;
  int64_t groups;
};

struct TransResult {
  std::shared_ptr<uint8_t> data;
  // data length in bytes
  size_t length;
};

class FormatTransfer {
 public:
  virtual ~FormatTransfer() = default;
  virtual uint32_t TransFormat(const TransArgs &args, TransResult &result) = 0;
  virtual uint32_t TransShape(Format src_format, const std::vector<int64_t> &src_shape, DataType data_type,
                              Format dst_format, std::vector<int64_t> &dst_shape, int64_t groups) = 0;
};

using FormatTransferBuilder = std::function<std::shared_ptr<FormatTransfer>()>;

class FormatTransferRegister {
 public:
  FormatTransferRegister(FormatTransferBuilder builder, Format src, Format dst);
  ~FormatTransferRegister() = default;
};

#define REGISTER_FORMAT_TRANSFER(TransferClass, format1, format2)                    \
  namespace {                                                                        \
  FormatTransferRegister format_transfer_register_##TransferClass##format1##format2( \
    []() { return std::make_shared<TransferClass>(); }, format1, format2);           \
  }

/**
 * Build a FormatTransfer according to 'args'
 * @param args
 * @param result
 * @return
 */
std::shared_ptr<FormatTransfer> BuildFormatTransfer(const TransArgs &args);

bool FormatTransferExists(const TransArgs &args);
}  // namespace formats
}  // namespace aicpu
#endif