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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_LAYOUT_TRANSFORM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_LAYOUT_TRANSFORM_H_

#ifdef ENABLE_FP16
#include <arm_neon.h>
#endif
#include "nnacl/pack.h"
#include "src/tensor.h"

namespace mindspore::kernel {
typedef void (*LayoutConvertor)(const void *src, void *dst, int batch, int plane, int channel);
#ifdef ENABLE_FP16
LayoutConvertor LayoutTransformFp16(mindspore::Format src_format, mindspore::Format dst_format);
#endif

LayoutConvertor LayoutTransformFp32(mindspore::Format src_format, mindspore::Format dst_format);

LayoutConvertor LayoutTransformInt8(mindspore::Format src_format, mindspore::Format dst_format);

LayoutConvertor LayoutTransform(TypeId data_type, mindspore::Format src_format, mindspore::Format dst_format);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_LAYOUT_TRANSFORM_H_
