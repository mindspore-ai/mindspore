/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_QUANT_CONSTANT_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_QUANT_CONSTANT_H_

namespace mindspore {
namespace kernel {
constexpr size_t kDimSizeOne = 1;
constexpr size_t kDimSizeTwo = 2;
constexpr size_t kDimSizeThree = 3;
constexpr size_t kDimSizeFour = 4;
constexpr size_t kDimIndexZeroth = 0;
constexpr size_t kDimIndexFirst = 1;
constexpr size_t kDimIndexSecond = 2;
constexpr size_t kDimIndexThird = 3;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_QUANT_CONSTANT_H_
