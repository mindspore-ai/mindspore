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

#include <limits>
#include "ctcloss_impl.cuh"
template <typename T>
__device__ T LogSumExp(const T logprob1, const T logprob2) {
  if (logprob1 == logprob2 && logprob1 == -std::numeric_limits<T>::infinity()) {
    return logprob1;
  } else {
    return (logprob1 > logprob2) ? logprob1 + log1pf(expf(logprob2 - logprob1))
                                 : logprob2 + log1pf(expf(logprob1 - logprob2));
  }
}

template <typename T>
__global__ void CalculateFwdVarKernel(T *log_alpha_b, int *label_value_with_blank, T *softmax_probs,
                                      const int *sequence_length, bool ctc_merge_repeated, int batch, int SOffSet,
                                      int maxtime, int blank, int *label_squence_length, int *cum_labels_length,
                                      bool ignore_longer_outputs_than_inputs) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch; i += blockDim.x * gridDim.x) {
    if (sequence_length[i] == 0 ||
        (ignore_longer_outputs_than_inputs && label_squence_length[i] > sequence_length[i])) {
    } else {
      T *log_alpha_b_cur = &log_alpha_b[i * SOffSet * maxtime];
      int *label_value_with_blank_cur = &label_value_with_blank[0];
      if (i > 0) {
        label_value_with_blank_cur = &label_value_with_blank[2 * cum_labels_length[i - 1] + i];
      }
      int numclass = blank + 1;
      int U = 2 * label_squence_length[i] + 1;
      int Ti = sequence_length[i];
      int low = 0;
      int high = 0;
      log_alpha_b_cur[0] = log(softmax_probs[i * numclass + blank]);
      int label0 = blank;
      if (U > 1) {
        label0 = label_value_with_blank_cur[1];
        log_alpha_b_cur[maxtime] = log(softmax_probs[i * numclass + label0]);
      }
      for (int t = 1; t < Ti; ++t) {
        low = 0;
        high = U;
        int low_limit = U - (2 * (Ti - t));
        int high_limit = 2 * (t + 1);
        if (low_limit > low) {
          low = low_limit;
        }
        if (high_limit < U) {
          high = high_limit;
        }
        for (int u = low; u < high; ++u) {
          T sum_log_alpha = -std::numeric_limits<T>::infinity();
          if (ctc_merge_repeated || label_value_with_blank_cur[u] == blank) {
            sum_log_alpha = log_alpha_b_cur[u * maxtime + t - 1];
          }
          if (u > 0) {
            sum_log_alpha = LogSumExp(sum_log_alpha, log_alpha_b_cur[(u - 1) * maxtime + t - 1]);
          }
          if (u > 1) {
            const bool matching_labels_merge =
              ctc_merge_repeated && (label_value_with_blank_cur[u] == label_value_with_blank_cur[u - 2]);
            if (label_value_with_blank_cur[u] != blank && !matching_labels_merge) {
              sum_log_alpha = LogSumExp(sum_log_alpha, log_alpha_b_cur[(u - 2) * maxtime + t - 1]);
            }
          }
          log_alpha_b_cur[u * maxtime + t] =
            log(softmax_probs[i * numclass + label_value_with_blank_cur[u] + t * numclass * batch]) + sum_log_alpha;
        }
      }
    }
  }
}

template <typename T>
__global__ void CalculateBwdVarKernel(T *log_beta_b, int *label_value_with_blank, T *softmax_probs,
                                      const int *sequence_length, bool ctc_merge_repeated, int batch, int SOffSet,
                                      int maxtime, int blank, int *label_squence_length, int *cum_labels_length,
                                      bool ignore_longer_outputs_than_inputs) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch; i += blockDim.x * gridDim.x) {
    if (sequence_length[i] == 0 ||
        (ignore_longer_outputs_than_inputs && label_squence_length[i] > sequence_length[i])) {
    } else {
      T *log_beta_b_cur = &log_beta_b[i * SOffSet * maxtime];
      int *label_value_with_blank_cur = &label_value_with_blank[0];
      if (i > 0) {
        label_value_with_blank_cur = &label_value_with_blank[2 * cum_labels_length[i - 1] + i];
      }
      int numclass = blank + 1;
      int U = 2 * label_squence_length[i] + 1;
      int Ti = sequence_length[i];
      int low = 0;
      int high = 0;
      if (U > 1) {
        for (int u = U - 2; u < U; ++u) {
          log_beta_b_cur[u * maxtime + Ti - 1] = 0;
        }
      } else {
        log_beta_b_cur[Ti - 1] = 0;
        log_beta_b_cur[Ti - 2] = 0;
      }
      for (int t = Ti - 2; t >= 0; --t) {
        low = 0;
        high = U;
        int low_limit = U - (2 * (Ti - t));
        int high_limit = 2 * (t + 1);
        if (low_limit > low) {
          low = low_limit;
        }
        if (high_limit < U) {
          high = high_limit;
        }
        for (int u = low; u < high; ++u) {
          if (ctc_merge_repeated || label_value_with_blank_cur[u] == blank) {
            log_beta_b_cur[u * maxtime + t] = LogSumExp(
              log_beta_b_cur[u * maxtime + t],
              log_beta_b_cur[u * maxtime + t + 1] +
                log(softmax_probs[i * numclass + label_value_with_blank_cur[u] + (t + 1) * numclass * batch]));
          }
          if (u + 1 < U) {
            log_beta_b_cur[u * maxtime + t] = LogSumExp(
              log_beta_b_cur[u * maxtime + t],
              log_beta_b_cur[(u + 1) * maxtime + t + 1] +
                log(softmax_probs[i * numclass + label_value_with_blank_cur[u + 1] + (t + 1) * numclass * batch]));
          }
          if (u + 2 < U) {
            const bool matching_labels_merge =
              ctc_merge_repeated && (label_value_with_blank_cur[u] == label_value_with_blank_cur[u + 2]);
            if (label_value_with_blank_cur[u] != blank && !matching_labels_merge) {
              log_beta_b_cur[u * maxtime + t] = LogSumExp(
                log_beta_b_cur[u * maxtime + t],
                log_beta_b_cur[(u + 2) * maxtime + t + 1] +
                  log(softmax_probs[i * numclass + label_value_with_blank_cur[u + 2] + (t + 1) * numclass * batch]));
            }
          }
        }
      }
    }
  }
}

template <typename T>
__global__ void ProbInitKernel(T *prob_num, int size) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    prob_num[i] = -std::numeric_limits<T>::infinity();
  }
}
template <typename T>
__global__ void LogBInitKernel(T *log_b, int log_prob_size) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < log_prob_size; i += blockDim.x * gridDim.x) {
    log_b[i] = -std::numeric_limits<T>::infinity();
  }
}

template <typename T>
__global__ void CTCLossKernel(T *log_alpha_b, T *log_beta_b, T *softmax_probs, int *label_value_with_blank, int batch,
                              int SOffSet, int maxtime, int numclass, const int *sequence_length,
                              int *label_squence_length, int *cum_labels_length, T *cost, T *grads, T *prob_num,
                              bool ignore_longer_outputs_than_inputs) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch; i += blockDim.x * gridDim.x) {
    if (sequence_length[i] == 0 ||
        (ignore_longer_outputs_than_inputs && label_squence_length[i] > sequence_length[i])) {
    } else {
      T *grad_cur = &grads[i * numclass];
      const T *softmax_probs_cur = &softmax_probs[i * numclass];
      T *prob_num_cur = &prob_num[i * numclass];
      int U = 2 * label_squence_length[i] + 1;
      T log_pzx = -std::numeric_limits<T>::infinity();
      const T *log_alpha_b_cur = &log_alpha_b[i * SOffSet * maxtime];
      const T *log_beta_b_cur = &log_beta_b[i * SOffSet * maxtime];
      int *label_value_with_blank_cur = &label_value_with_blank[0];
      if (i > 0) {
        label_value_with_blank_cur = &label_value_with_blank[2 * cum_labels_length[i - 1] + i];
      }
      for (int u = 0; u < U; ++u) {
        log_pzx = LogSumExp(log_pzx, log_alpha_b_cur[u * maxtime] + log_beta_b_cur[u * maxtime]);
      }
      cost[i] = -log_pzx;
      // grad
      int L = numclass;
      int Ti = sequence_length[i];
      if (log_pzx == -std::numeric_limits<T>::infinity()) {
        for (int t = 0; t < Ti; ++t) {
          for (int l = 0; l < L; ++l) {
            grad_cur[t * numclass * batch + l] = softmax_probs_cur[t * numclass * batch + l];
          }
        }
      } else {
        for (int t = 0; t < Ti; ++t) {
          for (int u = 0; u < U; ++u) {
            int l = label_value_with_blank_cur[u];
            prob_num_cur[t * batch * numclass + l] =
              LogSumExp(prob_num_cur[t * batch * numclass + l],
                        log_alpha_b_cur[u * maxtime + t] + log_beta_b_cur[u * maxtime + t]);
          }
          for (int l = 0; l < L; ++l) {
            grad_cur[t * numclass * batch + l] =
              softmax_probs_cur[t * numclass * batch + l] - expf(prob_num_cur[t * batch * numclass + l] - log_pzx);
          }
        }
      }
    }
  }
}

template <typename T>
__global__ void InnerSoftMaxKernel(const T *probs, T *softmax_probs, const int *sequence_length, int max_time,
                                   int batch, int numclass) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch * max_time; i += blockDim.x * gridDim.x) {
    int k = i / batch;
    int m = i % batch;
    if (k < sequence_length[m]) {
      T maxCoeff = 0.;
      T sumCoeff = 0.;
      for (int j = i * numclass; j < (i + 1) * numclass; ++j) {
        if (probs[j] > maxCoeff) {
          maxCoeff = probs[j];
        }
      }
      for (int j = i * numclass; j < (i + 1) * numclass; ++j) {
        sumCoeff += exp(probs[j] - maxCoeff);
        softmax_probs[j] = exp(probs[j] - maxCoeff);
      }
      for (int j = i * numclass; j < (i + 1) * numclass; ++j) {
        softmax_probs[j] /= sumCoeff;
      }
    }
  }
}

__global__ void GenLabelValuePCRKernel(int *label_value_sp, int *label_value_pcr, int *label_squence_length,
                                       int *cum_labels_length, int batch) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch; i += blockDim.x * gridDim.x) {
    int L = label_squence_length[i];
    label_squence_length[i] = 0;
    int offset = 0;
    if (i > 0) {
      offset = cum_labels_length[i - 1];
    }
    for (int l = offset; l < L; ++l) {
      if (l == offset || label_value_sp[l] != label_value_sp[l - 1]) {
        label_value_pcr[offset + label_squence_length[i]++] = label_value_sp[l];
      }
    }
  }
}

__global__ void UpdateLengthKernel(int *label_squence_length, int *cum_labels_length, int *max_labels_length,
                                   int batch) {
  max_labels_length[0] = 0;
  for (int i = 0; i < batch; ++i) {
    if (label_squence_length[i] > max_labels_length[0]) {
      max_labels_length[0] = label_squence_length[i];
    }
    if (i == 0) {
      cum_labels_length[i] = label_squence_length[i];
    } else {
      cum_labels_length[i] = label_squence_length[i] + cum_labels_length[i - 1];
    }
  }
}

template <typename T>
cudaError_t CalculateBwdVar(T *log_beta_b, int *label_value_with_blank, T *softmax_probs, const int *sequence_length,
                            bool ctc_merge_repeated, int batch, int SOffSet, int maxtime, int blank,
                            int *label_squence_length, int *cum_labels_length, bool ignore_longer_outputs_than_inputs,
                            cudaStream_t stream) {
  int log_prob_size = SOffSet * batch * maxtime;
  LogBInitKernel<<<GET_BLOCKS(log_prob_size), GET_THREADS, 0, stream>>>(log_beta_b, log_prob_size);
  CalculateBwdVarKernel<<<GET_BLOCKS(batch), GET_THREADS, 0, stream>>>(
    log_beta_b, label_value_with_blank, softmax_probs, sequence_length, ctc_merge_repeated, batch, SOffSet, maxtime,
    blank, label_squence_length, cum_labels_length, ignore_longer_outputs_than_inputs);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalculateFwdVar(T *log_alpha_b, int *label_value_with_blank, T *softmax_probs, const int *sequence_length,
                            bool ctc_merge_repeated, int batch, int SOffSet, int maxtime, int blank,
                            int *label_squence_length, int *cum_labels_length, bool ignore_longer_outputs_than_inputs,
                            cudaStream_t stream) {
  int log_prob_size = SOffSet * batch * maxtime;
  LogBInitKernel<<<GET_BLOCKS(log_prob_size), GET_THREADS, 0, stream>>>(log_alpha_b, log_prob_size);
  CalculateFwdVarKernel<<<GET_BLOCKS(batch), GET_THREADS, 0, stream>>>(
    log_alpha_b, label_value_with_blank, softmax_probs, sequence_length, ctc_merge_repeated, batch, SOffSet, maxtime,
    blank, label_squence_length, cum_labels_length, ignore_longer_outputs_than_inputs);
  return GetCudaStatus();
}

template <typename T>
cudaError_t InnerSoftMax(const T *probs, T *softmax_probs, const int *sequence_length, int max_time, int batch,
                         int numclass, cudaStream_t stream) {
  InnerSoftMaxKernel<<<GET_BLOCKS(batch * max_time), GET_THREADS, 0, stream>>>(probs, softmax_probs, sequence_length,
                                                                               max_time, batch, numclass);
  return GetCudaStatus();
}

__global__ void GenLabelWithBlankKernel(int *label_value, int *label_value_with_blank, int *label_squence_length,
                                        int *precum_labels_length, int *cum_labels_length, int batch, int blank) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch; i += blockDim.x * gridDim.x) {
    int offset = 0;
    int offset1 = 0;
    if (i > 0) {
      offset = 2 * cum_labels_length[i - 1] + i;
      offset1 = precum_labels_length[i - 1];
    }
    for (int j = 0; j < label_squence_length[i]; ++j) {
      label_value_with_blank[offset + 2 * j] = blank;
      label_value_with_blank[offset + 2 * j + 1] = label_value[offset1 + j];
    }
    label_value_with_blank[offset + 2 * label_squence_length[i]] = blank;
  }
}

cudaError_t GenLabelWithBlank(int *label_value, int *label_value_with_blank, int *label_squence_length,
                              int *precum_labels_length, int *cum_labels_length, int batch, int blank,
                              cudaStream_t stream) {
  GenLabelWithBlankKernel<<<GET_BLOCKS(batch), GET_THREADS, 0, stream>>>(
    label_value, label_value_with_blank, label_squence_length, precum_labels_length, cum_labels_length, batch, blank);
  return GetCudaStatus();
}

cudaError_t GenLabelValuePCR(int *label_value_sp, int *label_value_pcr, int *label_squence_length,
                             int *cum_labels_length, int *max_labels_length, int batch, cudaStream_t stream) {
  GenLabelValuePCRKernel<<<GET_BLOCKS(batch), GET_THREADS, 0, stream>>>(label_value_sp, label_value_pcr,
                                                                        label_squence_length, cum_labels_length, batch);
  UpdateLengthKernel<<<1, 1, 0, stream>>>(label_squence_length, cum_labels_length, max_labels_length, batch);
  return GetCudaStatus();
}

__global__ void GenLabelValueKernel(int *label_value_sp, const int64_t *label_indices, const int *label_values,
                                    int *label_squence_length, int *cum_labels_length, int size) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    int64_t b = label_indices[i * 2];
    int offset = 0;
    if (b > 0) {
      offset = cum_labels_length[b - 1];
    }
    int64_t index = offset + label_indices[i * 2 + 1];
    label_value_sp[index] = label_values[i];
  }
}
__global__ void LabelValueInitKernel(int *label_value_sp, int size, int blank) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    label_value_sp[i] = blank;
  }
}
__global__ void RecalculateLengthKernel(int *label_value_sp, int *label_squence_length, int *cum_labels_length,
                                        int batch, int blank) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch; i += blockDim.x * gridDim.x) {
    int offset = 0;
    if (i > 0) {
      offset = cum_labels_length[i - 1];
    }
    int L = label_squence_length[i];
    label_squence_length[i] = 0;
    for (int j = offset; j < offset + L; ++j) {
      if (label_value_sp[j] >= blank) {
        break;
      } else {
        label_squence_length[i]++;
      }
    }
  }
}
cudaError_t GenLabelValue(int *label_value_sp, const int64_t *label_indices, const int *label_values,
                          int *label_squence_length, int *cum_labels_length, int *max_labels_length, int size,
                          int blank, int batch, cudaStream_t stream) {
  LabelValueInitKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(label_value_sp, size, blank);
  GenLabelValueKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(label_value_sp, label_indices, label_values,
                                                                    label_squence_length, cum_labels_length, size);
  RecalculateLengthKernel<<<GET_BLOCKS(batch), GET_THREADS, 0, stream>>>(label_value_sp, label_squence_length,
                                                                         cum_labels_length, batch, blank);
  UpdateLengthKernel<<<1, 1, 0, stream>>>(label_squence_length, cum_labels_length, max_labels_length, batch);
  return GetCudaStatus();
}

__global__ void CalculatePreLengthKernel(int *label_squence_length, int *precum_labels_length, int *cum_labels_length,
                                         int *max_labels_length, const int64_t *label_indices, int batch, int size) {
  max_labels_length[0] = 0;
  for (int i = 0; i < size; ++i) {
    label_squence_length[label_indices[i * 2]]++;
    if (max_labels_length[0] < label_indices[i * 2]) {
      max_labels_length[0] = label_indices[i * 2];
    }
  }
  precum_labels_length[0] = label_squence_length[0];
  cum_labels_length[0] = label_squence_length[0];
  for (int i = 1; i < batch; ++i) {
    cum_labels_length[i] = cum_labels_length[i - 1] + label_squence_length[i];
    precum_labels_length[i] = precum_labels_length[i - 1] + label_squence_length[i];
  }
}

__global__ void CalculateMaxSequenceKernel(const int *sequence_length, int *max_labels_length, int batch) {
  max_labels_length[0] = 0;
  for (int i = 0; i < batch; ++i) {
    if (sequence_length[i] > max_labels_length[0]) {
      max_labels_length[0] = sequence_length[i];
    }
  }
}

cudaError_t CalculateMaxSequence(const int *sequence_length, int *max_labels_length, int batch, cudaStream_t stream) {
  CalculateMaxSequenceKernel<<<1, 1, 0, stream>>>(sequence_length, max_labels_length, batch);
  return GetCudaStatus();
}

cudaError_t CalculatePreLength(int *label_squence_length, int *precum_labels_length, int *cum_labels_length,
                               int *max_labels_length, const int64_t *label_indices, int batch, int size,
                               cudaStream_t stream) {
  CalculatePreLengthKernel<<<1, 1, 0, stream>>>(label_squence_length, precum_labels_length, cum_labels_length,
                                                max_labels_length, label_indices, batch, size);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CTCLoss(T *log_alpha_b, T *log_beta_b, T *softmax_probs, int *label_value_with_blank, int batch,
                    int SOffSet, int maxtime, int numclass, const int *sequence_length, int *label_squence_length,
                    int *cum_labels_length, T *cost, T *grads, T *prob_num, bool ignore_longer_outputs_than_inputs,
                    cudaStream_t stream) {
  ProbInitKernel<<<GET_BLOCKS(maxtime * batch * numclass), GET_THREADS, 0, stream>>>(prob_num,
                                                                                     maxtime * batch * numclass);
  CTCLossKernel<<<GET_BLOCKS(batch), GET_THREADS, 0, stream>>>(
    log_alpha_b, log_beta_b, softmax_probs, label_value_with_blank, batch, SOffSet, maxtime, numclass, sequence_length,
    label_squence_length, cum_labels_length, cost, grads, prob_num, ignore_longer_outputs_than_inputs);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalculateFwdVar<float>(
  float *log_alpha_b, int *label_value_with_blank, float *softmax_probs, const int *sequence_length,
  bool ctc_merge_repeated, int batch, int SOffSet, int maxtime, int blank, int *label_squence_length,
  int *cum_labels_length, bool ignore_longer_outputs_than_inputs, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CalculateBwdVar<float>(
  float *log_beta_b, int *label_value_with_blank, float *softmax_probs, const int *sequence_length,
  bool ctc_merge_repeated, int batch, int SOffSet, int maxtime, int blank, int *label_squence_length,
  int *cum_labels_length, bool ignore_longer_outputs_than_inputs, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t InnerSoftMax<float>(const float *probs, float *softmax_probs,
                                                         const int *sequence_length, int max_time, int batch,
                                                         int numclass, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CTCLoss<float>(float *log_alpha_b, float *log_beta_b, float *softmax_probs,
                                                    int *label_value_with_blank, int batch, int SOffSet, int maxtime,
                                                    int numclass, const int *sequence_length, int *label_squence_length,
                                                    int *cum_labels_length, float *cost, float *grads, float *prob_num,
                                                    bool ignore_longer_outputs_than_inputs, cudaStream_t stream);
