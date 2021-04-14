/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/ctcloss_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void CTCLossCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  probs_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  indice_dims_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  labels_dims_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);

  if (probs_shape_.size() != 3) {
    MS_LOG(EXCEPTION) << "Probs dims: " << probs_shape_.size() << " not support.";
  }
  if (labels_dims_.size() != 1) {
    MS_LOG(EXCEPTION) << "Labels dims: " << labels_dims_.size() << " not support.";
  }
  if (indice_dims_.size() != 2) {
    MS_LOG(EXCEPTION) << "Labels indice dims: " << indice_dims_.size() << " not support.";
  }

  preprocess_collapse_repeated_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "preprocess_collapse_repeated");
  ctc_merge_repeated_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "ctc_merge_repeated");
  ignore_longer_outputs_than_inputs_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "ignore_longer_outputs_than_inputs");

  max_time_ = probs_shape_[0];
  batch_size_ = probs_shape_[1];
  num_class_ = probs_shape_[2];
  blank_index_ = num_class_ - 1;
}

bool CTCLossCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                              const std::vector<kernel::AddressPtr> & /*workspace*/,
                              const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  }
  return true;
}

template <typename T>
inline T LogSumExp(T logprob1, T logprob2) {
  T kLogZero_ = -std::numeric_limits<T>::infinity();
  if (logprob1 == kLogZero_) {
    return logprob2;
  } else if (logprob2 == kLogZero_) {
    return logprob1;
  } else {
    return (logprob1 > logprob2) ? logprob1 + log1p(exp(logprob2 - logprob1))
                                 : logprob2 + log1p(exp(logprob1 - logprob2));
  }
}

template <typename TT>
void CTCLossCPUKernel::CalculateFwdVar(const std::vector<uint32_t> &label_with_blank,
                                       const std::vector<std::vector<TT>> &y,
                                       std::vector<std::vector<TT>> *log_alpha_b) {
  int U = label_with_blank.size();
  int T = (*log_alpha_b)[0].size();
  TT kLogZero_ = -std::numeric_limits<TT>::infinity();

  (*log_alpha_b)[0][0] = log(y[blank_index_][0]);
  auto label_0 = (label_with_blank.size() > 1) ? label_with_blank[1] : blank_index_;
  if (label_with_blank.size() > 1) {
    (*log_alpha_b)[1][0] = log(y[label_0][0]);
  }

  for (int t = 1; t < T; ++t) {
    int low = std::max(0, U - (2 * (T - t)));
    int high = std::min(U, 2 * (t + 1));
    for (int u = low; u < high; ++u) {
      auto sum_log_alpha_b = kLogZero_;
      if (ctc_merge_repeated_ || label_with_blank[u] == blank_index_) {
        sum_log_alpha_b = (*log_alpha_b)[u][t - 1];
      }

      if (u > 0) {
        sum_log_alpha_b = LogSumExp(sum_log_alpha_b, (*log_alpha_b)[u - 1][t - 1]);
      }

      if (u > 1) {
        bool matching_labels_merge = ctc_merge_repeated_ && (label_with_blank[u] == label_with_blank[u - 2]);
        if (label_with_blank[u] != blank_index_ && !matching_labels_merge) {
          sum_log_alpha_b = LogSumExp(sum_log_alpha_b, (*log_alpha_b)[u - 2][t - 1]);
        }
      }

      (*log_alpha_b)[u][t] = log(y[label_with_blank[u]][t]) + sum_log_alpha_b;
    }
  }
}

template <typename TT>
void CTCLossCPUKernel::CalculateBwdVar(const std::vector<uint32_t> &label_with_blank,
                                       const std::vector<std::vector<TT>> &y,
                                       std::vector<std::vector<TT>> *log_beta_b) {
  int T = (*log_beta_b)[0].size();
  int U = label_with_blank.size();
  if (U > 1) {
    for (int u = U - 2; u < U; ++u) {
      (*log_beta_b)[u][T - 1] = TT(0);
    }
  } else {
    (*log_beta_b)[0][T - 1] = TT(0);
    (*log_beta_b)[0][T - 2] = TT(0);
  }

  for (int t = T - 2; t >= 0; --t) {
    int low = std::max(0, U - (2 * (T - t)));
    int high = std::min(U, 2 * (t + 1));
    for (int u = low; u < high; ++u) {
      if (ctc_merge_repeated_ || label_with_blank[u] == blank_index_) {
        (*log_beta_b)[u][t] =
          LogSumExp((*log_beta_b)[u][t], (*log_beta_b)[u][t + 1] + TT(log(y[label_with_blank[u]][t + 1])));
      }

      if (u + 1 < U) {
        (*log_beta_b)[u][t] =
          LogSumExp((*log_beta_b)[u][t], (*log_beta_b)[u + 1][t + 1] + TT(log(y[label_with_blank[u + 1]][t + 1])));
      }

      if (u + 2 < U) {
        bool matching_labels_merge = ctc_merge_repeated_ && (label_with_blank[u] == label_with_blank[u + 2]);
        if (label_with_blank[u] != blank_index_ && !matching_labels_merge) {
          (*log_beta_b)[u][t] =
            LogSumExp((*log_beta_b)[u][t], (*log_beta_b)[u + 2][t + 1] + TT(log(y[label_with_blank[u + 2]][t + 1])));
        }
      }
    }
  }
}

template <typename TT>
void CTCLossCPUKernel::CalculateGrad(const std::vector<uint32_t> &label_with_blank,
                                     const std::vector<std::vector<TT>> &y,
                                     const std::vector<std::vector<TT>> &log_alpha_b,
                                     const std::vector<std::vector<TT>> &log_beta_b, const TT log_pzx,
                                     std::vector<std::vector<TT>> *dy) {
  auto dy_b = dy;
  TT kLogZero_ = -std::numeric_limits<TT>::infinity();
  if (log_pzx == kLogZero_) {
    MS_LOG(INFO) << "No valid path found";
    return;
  }

  size_t L = y.size();
  size_t T = y[0].size();
  size_t U = label_with_blank.size();

  for (size_t t = 0; t < T; ++t) {
    std::vector<TT> prob_sum(L, kLogZero_);

    for (size_t u = 0; u < U; ++u) {
      uint32_t l = label_with_blank[u];
      prob_sum[l] = LogSumExp(prob_sum[l], log_alpha_b[u][t] + log_beta_b[u][t]);
    }
    for (size_t l = 0; l < L; ++l) {
      (*dy_b)[l][t] = y[l][t] - exp(prob_sum[l] - log_pzx);
    }
  }
}

void CTCLossCPUKernel::GenLableWithBlank(const uint32_t *seq_len, const std::vector<std::vector<uint32_t>> &batch_label,
                                         std::vector<std::vector<uint32_t>> *label_with_blank) {
  for (size_t b = 0; b < batch_size_; ++b) {
    std::vector<uint32_t> l;
    const std::vector<uint32_t> &label = batch_label[b];
    bool has_blank = false;
    for (size_t i = 0; i < label.size(); ++i) {
      if (i == 0 || !preprocess_collapse_repeated_ || label[i] != label[i - 1]) {
        if (label[i] >= num_class_ - 1) {
          has_blank = true;
        } else {
          if (has_blank) {
            MS_LOG(EXCEPTION) << "Invalid labels(index >= num_class - 1) should not appear between two valid labels";
          }
          l.push_back(label[i]);
        }
      }
    }
    if (!ignore_longer_outputs_than_inputs_) {
      if (l.size() > seq_len[b]) {
        MS_LOG(EXCEPTION) << "Input time(sequence length) should greater than output size(label length), but gets "
                          << seq_len[b] << "< " << l.size();
      }
    }

    (*label_with_blank)[b].reserve(2 * l.size() + 1);
    for (auto l_i : l) {
      (*label_with_blank)[b].push_back(blank_index_);
      (*label_with_blank)[b].push_back(l_i);
    }
    (*label_with_blank)[b].push_back(blank_index_);
  }
}

template <typename T>
void InnerSoftMax(const T *inputs_addr, std::vector<std::vector<T>> *softmax_probs, const uint32_t sequence_length,
                  size_t num_class, size_t batch_size, size_t b) {
  for (size_t t = 0; t < sequence_length; ++t) {
    T maxCoeff(T(0));
    T sumCoeff(T(0));

    for (size_t c = 0; c < num_class; ++c) {
      if (inputs_addr[t * batch_size * num_class + b * num_class + c] > maxCoeff) {
        maxCoeff = inputs_addr[t * batch_size * num_class + b * num_class + c];
      }
    }

    for (size_t c = 0; c < num_class; ++c) {
      sumCoeff += exp(inputs_addr[t * batch_size * num_class + b * num_class + c] - maxCoeff);
      (*softmax_probs)[c][t] = exp(inputs_addr[t * batch_size * num_class + b * num_class + c] - maxCoeff);
    }

    for (size_t c = 0; c < num_class; ++c) {
      (*softmax_probs)[c][t] /= sumCoeff;
    }
  }
}

template <typename T>
void MatrixfromVector(uint32_t row, uint32_t col, std::vector<std::vector<T>> *array2D, const T init_value) {
  array2D->resize(row);
  for (size_t i = 0; i < row; ++i) {
    (*array2D)[i].resize(col, init_value);
  }
}

template <typename T>
void CTCLossCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto inputs_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto labels_indices_addr = reinterpret_cast<uint64_t *>(inputs[1]->addr);
  auto labels_values_addr = reinterpret_cast<uint32_t *>(inputs[2]->addr);
  auto sequence_length_addr = reinterpret_cast<uint32_t *>(inputs[3]->addr);
  auto loss_addr = reinterpret_cast<T *>(outputs[0]->addr);
  auto gradient_addr = reinterpret_cast<T *>(outputs[1]->addr);

  std::vector<std::vector<uint32_t>> label_batch;
  std::vector<std::vector<uint32_t>> labels_with_blank;
  std::vector<uint64_t> each_label_length;

  label_batch.resize(batch_size_);
  labels_with_blank.resize(batch_size_);
  each_label_length.resize(batch_size_, 0);

  T kLogZero_ = -std::numeric_limits<T>::infinity();
  // check validation of sequence length
  for (size_t b = 0; b < batch_size_; ++b) {
    if (sequence_length_addr[b] < uint32_t(0)) {
      MS_LOG(EXCEPTION) << "Sequence length should > 0, but gets " << sequence_length_addr[b];
    }

    if (sequence_length_addr[b] > max_time_) {
      MS_LOG(EXCEPTION) << "Max time should be greater than sequence length, but gets " << max_time_ << " < "
                        << sequence_length_addr[b];
    }
  }

  for (size_t i = 0; i < indice_dims_[0]; ++i) {
    each_label_length[labels_indices_addr[i * 2]]++;
  }

  // convert label format of label_value and label_indices to batch_label
  uint64_t cum_sum = 0;
  for (size_t b = 0; b < batch_size_; ++b) {
    std::vector<uint32_t> *b_value = &label_batch[b];
    for (size_t l = 0; l < each_label_length[b]; ++l) {
      b_value->push_back(labels_values_addr[cum_sum + l]);
    }
    cum_sum += each_label_length[b];
  }

  // convert label to label with blank
  GenLableWithBlank(sequence_length_addr, label_batch, &labels_with_blank);

  for (size_t b = 0; b < batch_size_; ++b) {
    std::vector<uint32_t> label_with_blank = labels_with_blank[b];
    // y_b [num_class, sequence_length]
    std::vector<std::vector<T>> y_b;
    std::vector<std::vector<T>> dy;
    std::vector<std::vector<T>> log_alpha_b;
    std::vector<std::vector<T>> log_beta_b;
    MatrixfromVector(num_class_, sequence_length_addr[b], &y_b, kLogZero_);
    MatrixfromVector(y_b.size(), y_b[0].size(), &dy, T(0));
    MatrixfromVector(label_with_blank.size(), sequence_length_addr[b], &log_alpha_b, kLogZero_);
    MatrixfromVector(label_with_blank.size(), sequence_length_addr[b], &log_beta_b, kLogZero_);
    InnerSoftMax(inputs_addr, &y_b, sequence_length_addr[b], num_class_, batch_size_, b);

    CalculateFwdVar(label_with_blank, y_b, &log_alpha_b);
    CalculateBwdVar(label_with_blank, y_b, &log_beta_b);

    T log_pzx = kLogZero_;
    for (size_t u = 0; u < label_with_blank.size(); ++u) {
      log_pzx = LogSumExp(log_pzx, log_alpha_b[u][0] + log_beta_b[u][0]);
    }

    loss_addr[b] = -log_pzx;

    CalculateGrad(label_with_blank, y_b, log_alpha_b, log_beta_b, log_pzx, &dy);

    for (size_t t = 0; t < sequence_length_addr[b]; ++t) {
      for (size_t c = 0; c < num_class_; ++c) {
        gradient_addr[t * batch_size_ * num_class_ + b * num_class_ + c] = dy[c][t];
      }
    }
  }
}

void CTCLossCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 4) {
    MS_LOG(EXCEPTION) << "CTCLossCPUKernel needs 4 inputs, but gets " << input_num;
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 2) {
    MS_LOG(EXCEPTION) << "CTCLossCPUKernel expects 2 outputs, but gets" << output_num;
  }
}
}  // namespace kernel
}  // namespace mindspore
