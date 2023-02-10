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
#ifndef AICPU_KERNELS_HOST_FORMAT_TRANSFER_FORMAT_TRANSFER_FRACTAL_Z_3D_H
#define AICPU_KERNELS_HOST_FORMAT_TRANSFER_FORMAT_TRANSFER_FRACTAL_Z_3D_H

#include <vector>
#include "cpu_kernel/format_transfer/register_format_transfer.h"

namespace aicpu {
namespace formats {
class FormatTransferFractalz3D : public FormatTransfer {
 public:
  uint32_t TransFormat(const TransArgs &args, TransResult &result) override;
  uint32_t TransShape(Format src_format, const std::vector<int64_t> &src_shape, DataType data_type, Format dst_format,
                      std::vector<int64_t> &dst_shape, int64_t groups) override;
};
}  // namespace formats
}  // namespace aicpu

#endif  // AICPU_KERNELS_HOST_FORMAT_TRANSFERS_FORMAT_TRANSFER_FRACTAL_NZ_H_
