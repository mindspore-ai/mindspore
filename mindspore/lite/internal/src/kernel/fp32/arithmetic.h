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
#ifndef INTERNAL_SRC_RUNTIME_KERNEL_MUL_H_
#define INTERNAL_SRC_RUNTIME_KERNEL_MUL_H_

#include "internal/include/model.h"
#include "internal/include/lite_utils.h"
#include "internal/src/allocator.h"
#include "nnacl/arithmetic_common.h"

int DoArithmeticInferShape(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, OpParameter *param);

int DoArithmetic(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, Node *node,
                 mindspore::lite::Allocator *allocator);

#endif  // INTERNAL_SRC_RUNTIME_KERNEL_MUL_H_
