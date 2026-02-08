/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/ctcloss_cpu_kernel.h"
#include <map>

namespace mindspore {
namespace kernel {
namespace ctcloss_cpu {
namespace {
constexpr size_t kCTCLossInputsNum = 4;
constexpr size_t kCTCLossOutputsNum = 2;

template <typename T>
inline T LogSumExp(const T logprob1, const T logprob2) {
  T kLogZero_ = -std::numeric_limits<T>::infinity();
  if (logprob1 <= kLogZero_) {
    return logprob2;
  }
  if (logprob2 <= kLogZero_) {
    return logprob1;
  }
  return (logprob1 > logprob2) ? logprob1 + static_cast<T>(log1p(exp(logprob2 - logprob1)))
                               : logprob2 + static_cast<T>(log1p(exp(logprob1 - logprob2)));
}

template <typename T>
void InnerSoftMax(const T *inputs_addr, std::vector<std::vector<T>> *softmax_probs, const uint32_t sequence_length,
                  size_t num_class, size_t batch_size, size_t b) {
  for (size_t t = 0; t < sequence_length; ++t) {
    auto maxCoeff = static_cast<T>(0);
    auto sumCoeff = static_cast<T>(0);

    for (size_t c = 0; c < num_class; ++c) {
      if (inputs_addr[t * batch_size * num_class + b * num_class + c] > maxCoeff) {
        maxCoeff = inputs_addr[t * batch_size * num_class + b * num_class + c];
      }
    }

    for (size_t c = 0; c < num_class; ++c) {
      sumCoeff += static_cast<T>(exp(inputs_addr[t * batch_size * num_class + b * num_class + c] - maxCoeff));
      (*softmax_probs)[c][t] =
        static_cast<T>(exp(inputs_addr[t * batch_size * num_class + b * num_class + c] - maxCoeff));
    }

    for (size_t c = 0; c < num_class; ++c) {
      (*softmax_probs)[c][t] /= sumCoeff;
    }
  }
}

template <typename T>
void MatrixFromVector(uint32_t row, uint32_t col, std::vector<std::vector<T>> *array2D, const T init_value) {
  array2D->resize(row);
  for (size_t i = 0; i < row; ++i) {
    (*array2D)[i].resize(col, init_value);
  }
}
}  // namespace

bool CTCLossCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCTCLossInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCTCLossOutputsNum, kernel_name_);

  preprocess_collapse_repeated_ = GetValue<bool>(primitive_->GetAttr(PCR));
  ctc_merge_repeated_ = GetValue<bool>(primitive_->GetAttr(CTR));
  ignore_longer_outputs_than_inputs_ = GetValue<bool>(primitive_->GetAttr(ILOTI));
  return true;
}

int CTCLossCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  probs_shape_ = inputs[kIndex0]->GetShapeVector();
  indices_dims_ = inputs[kIndex1]->GetShapeVector();
  labels_dims_ = inputs[kIndex2]->GetShapeVector();
  dtype_ = inputs[kIndex0]->dtype_id();

  constexpr size_t kSizeOne = 1;
  constexpr size_t kSizeTwo = 2;
  constexpr size_t kSizeThree = 3;
  if (probs_shape_.size() != kSizeThree) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'probs' must be 3-D, but got " << probs_shape_.size()
                      << "-D.";
  }
  if (labels_dims_.size() != kSizeOne) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'labels' must be 1-D, but got " << labels_dims_.size()
                      << "-D.";
  }
  if (indices_dims_.size() != kSizeTwo) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'labels_indices' must be 2-D, but got "
                      << indices_dims_.size() << "-D.";
  }

  max_time_ = LongToSize(probs_shape_[kIndex0]);
  batch_size_ = LongToSize(probs_shape_[kIndex1]);
  num_class_ = LongToSize(probs_shape_[kIndex2]);
  blank_index_ = num_class_ - 1;
  return KRET_OK;
}

bool CTCLossCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                 const std::vector<kernel::KernelTensor *> &,
                                 const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCTCLossInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCTCLossOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of input 'x' must be float16 or float32 on CPU, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

template <typename TT>
void CTCLossCpuKernelMod::CalculateFwdVar(const std::vector<uint32_t> &label_with_blank,
                                          const std::vector<std::vector<TT>> &y,
                                          std::vector<std::vector<TT>> *log_alpha_b) const {
  int U = label_with_blank.size();
  int T = (*log_alpha_b)[0].size();
  TT kLogZero_ = -std::numeric_limits<TT>::infinity();

  (*log_alpha_b)[0][0] = static_cast<TT>(log(y[blank_index_][0]));
  auto label_0 = (label_with_blank.size() > 1) ? label_with_blank[1] : blank_index_;
  if (label_with_blank.size() > 1) {
    (*log_alpha_b)[1][0] = static_cast<TT>(log(y[label_0][0]));
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

      (*log_alpha_b)[u][t] =
        static_cast<TT>(log(static_cast<TT>(y[label_with_blank[IntToSize(u)]][IntToSize(t)]))) + sum_log_alpha_b;
    }
  }
}

template <typename TT>
void CTCLossCpuKernelMod::CalculateBwdVar(const std::vector<uint32_t> &label_with_blank,
                                          const std::vector<std::vector<TT>> &y,
                                          std::vector<std::vector<TT>> *log_beta_b) const {
  size_t T = (*log_beta_b)[0].size();
  size_t U = label_with_blank.size();
  if (U > 1) {
    for (size_t u = U - 2; u < U; ++u) {
      (*log_beta_b)[u][T - 1] = TT(0);
    }
  } else {
    (*log_beta_b)[0][T - 1] = TT(0);
    (*log_beta_b)[0][T - 2] = TT(0);
  }

  MS_EXCEPTION_IF_CHECK_FAIL(T > 1, "For CTCLoss, the length of log_beta_b must be greater than 1!");
  for (int64_t tt = T - 2; tt >= 0; --tt) {
    size_t low = std::max(static_cast<int64_t>(0), static_cast<int64_t>(U) - (2 * (static_cast<int64_t>(T) - tt)));
    size_t high = std::min(static_cast<int64_t>(U), 2 * (tt + 1));
    auto t = static_cast<size_t>(tt);
    for (size_t u = low; u < high; ++u) {
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
void CTCLossCpuKernelMod::CalculateGrad(const std::vector<uint32_t> &label_with_blank,
                                        const std::vector<std::vector<TT>> &y,
                                        const std::vector<std::vector<TT>> &log_alpha_b,
                                        const std::vector<std::vector<TT>> &log_beta_b, const TT log_pzx,
                                        std::vector<std::vector<TT>> *dy) const {
  auto dy_b = dy;
  TT kLogZero_ = -std::numeric_limits<TT>::infinity();
  if (log_pzx <= kLogZero_) {
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
      (*dy_b)[l][t] = y[l][t] - static_cast<TT>(exp(prob_sum[l] - log_pzx));
    }
  }
}

void CTCLossCpuKernelMod::GenLabelWithBlank(const uint32_t *seq_len,
                                            const std::vector<std::vector<uint32_t>> &batch_label,
                                            std::vector<std::vector<uint32_t>> *label_with_blank) const {
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
            MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of labels_values[" << i
                              << "] must be in the range of [0, num_classes), but got " << label[i];
          }
          l.push_back(label[i]);
        }
      }
    }
    if (!ignore_longer_outputs_than_inputs_ && l.size() > seq_len[b]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << ", input time(sequence length) must be greater than "
                           "output size(label length), but got sequence length: "
                        << seq_len[b] << " and label length: " << l.size();
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
void CTCLossCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) const {
  const auto *inputs_addr = GetDeviceAddress<T>(inputs, kIndex0);
  const auto *labels_indices_addr = GetDeviceAddress<uint64_t>(inputs, kIndex1);
  const auto *labels_values_addr = GetDeviceAddress<uint32_t>(inputs, kIndex2);
  const auto *sequence_length_addr = GetDeviceAddress<uint32_t>(inputs, kIndex3);
  auto *loss_addr = GetDeviceAddress<T>(outputs, kIndex0);
  auto *gradient_addr = GetDeviceAddress<T>(outputs, kIndex1);

  std::vector<std::vector<uint32_t>> label_batch;
  std::vector<std::vector<uint32_t>> labels_with_blank;
  std::vector<uint64_t> each_label_length;

  label_batch.resize(batch_size_);
  labels_with_blank.resize(batch_size_);
  each_label_length.resize(batch_size_, 0);

  T kLogZero_ = -std::numeric_limits<T>::infinity();
  // check validation of sequence length
  for (size_t b = 0; b < batch_size_; ++b) {
    if (sequence_length_addr[b] == static_cast<uint32_t>(0)) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", the 'sequence_length' must be greater than 0, but got "
                        << sequence_length_addr[b] << ".";
    }
    if (sequence_length_addr[b] > max_time_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << ", the 'max_time'(the 1st dimension value of 'probs') must be "
                           "greater than or equal to 'sequence_length', but got 'max_time': "
                        << max_time_ << " and 'sequence_length': " << sequence_length_addr[b];
    }
  }
  for (size_t i = 0; i < LongToSize(indices_dims_[0]); ++i) {
    const size_t factor = 2;
    auto index = labels_indices_addr[i * factor];
    if (index >= SizeToUlong(each_label_length.size())) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << ", 'index' must be less than the length of 'label', but got 'index': " << index
                        << " and the length of 'label': " << SizeToUlong(each_label_length.size());
    }
    each_label_length[index]++;
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
  GenLabelWithBlank(sequence_length_addr, label_batch, &labels_with_blank);

  for (size_t b = 0; b < batch_size_; ++b) {
    std::vector<uint32_t> label_with_blank = labels_with_blank[b];
    // y_b [num_class, sequence_length]
    std::vector<std::vector<T>> y_b;
    std::vector<std::vector<T>> dy;
    std::vector<std::vector<T>> log_alpha_b;
    std::vector<std::vector<T>> log_beta_b;
    MatrixFromVector(num_class_, sequence_length_addr[b], &y_b, kLogZero_);
    MatrixFromVector(y_b.size(), y_b[0].size(), &dy, T(0));
    MatrixFromVector(label_with_blank.size(), sequence_length_addr[b], &log_alpha_b, kLogZero_);
    MatrixFromVector(label_with_blank.size(), sequence_length_addr[b], &log_beta_b, kLogZero_);
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

std::vector<KernelAttr> CTCLossCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeFloat16)
                                                   .AddOutputAttr(kNumberTypeFloat16),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32)};

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CTCLoss, CTCLossCpuKernelMod);
}  // namespace ctcloss_cpu
}  // namespace kernel
}  // namespace mindspore
