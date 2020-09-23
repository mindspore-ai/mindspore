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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_IMPL_CUH
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_IMPL_CUH

template <typename T>
void CalculateFwdVar(T *log_alpha_b, int *label_value_with_blank, T *softmax_probs, const int *sequence_length,
                     bool ctc_merge_repeated, int batch, int SOffSet, int maxtime, int blank, int *label_squence_length,
                     int *cum_labels_length, bool ignore_longer_outputs_than_inputs, cudaStream_t stream);

template <typename T>
void CalculateBwdVar(T *log_beta_b, int *label_value_with_blank, T *softmax_probs, const int *sequence_length,
                     bool ctc_merge_repeated, int batch, int SOffSet, int maxtime, int blank, int *label_squence_length,
                     int *cum_labels_length, bool ignore_longer_outputs_than_inputs, cudaStream_t stream);

template <typename T>
void InnerSoftMax(const T *probs, T *softmax_cost, const int *sequence_length, int max_time, int batch, int numclass,
                  cudaStream_t stream);

void GenLabelValuePCR(int *label_value_sp, int *label_value_pcr, int *label_squence_length, int *cum_labels_length,
                      int *max_labels_length, int batch, cudaStream_t stream);

void GenLabelWithBlank(int *label_value, int *label_value_with_blank, int *label_squence_length,
                       int *precum_labels_length, int *cum_labels_length, int batch, int blank, cudaStream_t stream);

void GenLabelValue(int *label_value_sp, const int64_t *label_indices, const int *label_values,
                   int *label_squence_length, int *cum_labels_length, int *max_labels_length, int size, int blank,
                   int batch, cudaStream_t stream);

void CalculatePreLength(int *label_squence_length, int *precum_labels_length, int *cum_labels_length,
                        int *max_labels_length, const int64_t *label_indices, int batch, int size, cudaStream_t stream);
void CalculateMaxSequence(const int *sequence_length, int *max_labels_length, int batch, cudaStream_t stream);
template <typename T>
void CTCLoss(T *log_alpha_b, T *log_beta_b, T *softmax_probs, int *label_value_with_blank, int batch, int SOffSet,
             int maxtime, int numclass, const int *sequence_length, int *label_squence_length, int *cum_labels_length,
             T *cost, T *grads, T *prob_num, bool ignore_longer_outputs_than_inputs, cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_IMPL_CUH
