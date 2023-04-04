/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_CUDA_IMPL_USE_PAST_EMBEDDING_CUH_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_CUDA_IMPL_USE_PAST_EMBEDDING_CUH_

void InvokeUsePastEmbedding(const int *input_position, int *out_position, int *init_reset,
                            int *valid_length, int size, cudaStream_t cuda_stream);

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_CUDA_IMPL_USE_PAST_EMBEDDING_CUH_
