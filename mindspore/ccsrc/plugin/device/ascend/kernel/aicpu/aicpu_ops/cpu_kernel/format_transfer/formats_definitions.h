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

#ifndef AICPU_KERNELS_HOST_FORMAT_TRANSFER_FORMAT_TRANSFERS_FORMAT_TRANSFER_DEFINITIONS_H
#define AICPU_KERNELS_HOST_FORMAT_TRANSFER_FORMAT_TRANSFERS_FORMAT_TRANSFER_DEFINITIONS_H

namespace aicpu {
namespace formats {
enum NchwDimIndex { kNchwN, kNchwC, kNchwH, kNchwW, kNchwDimsNum };

enum NhwcDimIndex { kNhwcN, kNhwcH, kNhwcW, kNhwcC, kNhwcDimsNum };

enum HwcnDimIndex { kHwcnH, kHwcnW, kHwcnC, kHwcnN, kHwcnDimsNum };

enum ChwnDimIndex { kChwnC, kChwnH, kChwnW, kChwnN, kChwnDimsNum };

enum Nc1hwc0DimIndex { kNc1hwc0N, kNc1hwc0C1, kNc1hwc0H, kNc1hwc0W, kNc1hwc0C0, kNc1hwc0DimsNum };

enum C1hwncoc0DimIndex {
  kC1hwncoc0C1,
  kC1hwncoc0H,
  kC1hwncoc0W,
  kC1hwncoc0N,
  kC1hwncoc0Co,
  kC1hwncoc0C0,
  kC1hwncoc0DimsNum
};

enum FracZDimIndex { kFracZHWC1, kFracZN0, kFracZNi, kFracZC0, kFracZDimsNum };

enum DhwcnDimIndex { kDhwcnD, kDhwcnH, kDhwcnW, kDhwcnC, kDhwcnN, kDhwcnDimsNum };

enum NcdhwDimIndex { kNcdhwN, kNcdhwC, kNcdhwD, kNcdhwH, kNcdhwW, kNcdhwDimsNum };

enum NdhwcDimIndex { kNdhwcN, kNdhwcD, kNdhwcH, kNdhwcW, kNdhwcC, kNdhwcDimsNum };
}  // namespace formats
}  // namespace aicpu
#endif  // AICPU_KERNELS_HOST_FORMAT_TRANSFER_FORMAT_TRANSFERS_FORMAT_TRANSFER_DEFINITIONS_H_
