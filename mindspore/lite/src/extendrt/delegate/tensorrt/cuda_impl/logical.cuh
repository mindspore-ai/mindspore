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

#ifndef MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_CDUA_IMPL_LOGICAL_H_
#define MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_CDUA_IMPL_LOGICAL_H_

template <typename T>
void LogicalAnd(const T *input1, const T *input2, T *output, int element_cnt, cudaStream_t stream);

template <typename T>
void LogicalOr(const T *input1, const T *input2, T *output, int element_cnt, cudaStream_t stream);

template <typename T>
void LogicalNot(const T *input1, T *output, int element_cnt, cudaStream_t stream);

template <typename T>
void GreaterOrEqual(const T *input1, const T *input2, T *output, int element_cnt, cudaStream_t stream);

template <typename T>
void LessOrEqual(const T *input1, const T *input2, T *output, int element_cnt, cudaStream_t stream);

#endif  // MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_CDUA_IMPL_LOGICAL_H_
