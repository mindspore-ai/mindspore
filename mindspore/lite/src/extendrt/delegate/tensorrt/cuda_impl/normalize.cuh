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

#ifndef MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_CDUA_IMPL_NORMALIZE_H_
#define MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_CDUA_IMPL_NORMALIZE_H_

template <typename T>
void Normalize(const T *input, const T *gamma, const T *beta, T *output, size_t dim_at_axis, float epsilion,
               int element_cnt, cudaStream_t stream);

#endif  // MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_CDUA_IMPL_NORMALIZE_H_
