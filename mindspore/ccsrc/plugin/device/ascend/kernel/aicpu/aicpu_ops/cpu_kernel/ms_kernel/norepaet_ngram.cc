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
#include "norepeat_ngram.h"
#include <map>
#include <algorithm>
#include <functional>
#include <memory.h>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"

#define FLT_MAX __FLT_MAX__
namespace {
const char *kNoRepeatNGram = "NoRepeatNGram";
constexpr auto kInputSize = 2;
constexpr auto kOutputSize = 1;
constexpr auto ngram_step_size = 2;
}  // namespace
namespace aicpu {
uint32_t NoRepeatNGramCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputSize, kOutputSize), "NoRepeatNGramCpu check input and output failed.");
  Tensor *input = ctx.Input(1);
  auto data_type_in = input->GetDataType();
  AttrValue *ngram_size_ptr = ctx.GetAttr("ngram_size");
  int64_t ngram_size = (ngram_size_ptr == nullptr) ? 0 : ngram_size_ptr->GetInt();
  switch (data_type_in) {
    case DT_FLOAT16:
      return ComputeKernel<Eigen::half>(ctx, ngram_size);
    case DT_FLOAT:
      return ComputeKernel<float>(ctx, ngram_size);
    case DT_DOUBLE:
      return ComputeKernel<double>(ctx, ngram_size);
    default:
      KERNEL_LOG_ERROR("NoRepeatNGramCpu kernel data type [%s] not support.", DTypeStr(data_type_in).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

std::vector<int> calculate_banned_tokens(int *state_seq, const int &step,
                                         std::vector<std::map<std::vector<int>, std::vector<int>>> gen_ngrams,
                                         const int &no_repeat_ngram_size, const int &bbsz_idx) {
  int begin = bbsz_idx * step + step + 1 - no_repeat_ngram_size;
  int end = bbsz_idx * step + step;
  std::vector<int> ngram_index(state_seq + begin, state_seq + end);
  return gen_ngrams[bbsz_idx][ngram_index];
}

template <typename T>
uint32_t NoRepeatNGramCpuKernel::ComputeKernel(CpuKernelContext &ctx, const int64_t &no_repeat_ngram_size) {
  int *state_seq = reinterpret_cast<int *>(ctx.Input(0)->GetData());
  T *log_probs_origin = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  T *log_probs = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  size_t batch_size = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  size_t beam_width = ctx.Input(0)->GetTensorShape()->GetDimSize(1);
  size_t step = ctx.Input(0)->GetTensorShape()->GetDimSize(2);
  size_t vocab_size = ctx.Input(1)->GetTensorShape()->GetDimSize(2);

  int copy_size = batch_size * beam_width * vocab_size * sizeof(T);
  memcpy_s(log_probs, copy_size, log_probs_origin, copy_size);

  int real_batch_size = batch_size * beam_width;
  std::vector<std::map<std::vector<int>, std::vector<int>>> gen_ngrams(real_batch_size);
  for (int bbsz_idx = 0; bbsz_idx < real_batch_size; ++bbsz_idx) {
    int begin = bbsz_idx * step;
    int end = (bbsz_idx + 1) * step;
    std::vector<int> gen_tokens(state_seq + begin, state_seq + end);
    std::vector<std::vector<int>> gen_tokens_ngram;
    for (int i = 0; i < static_cast<int>(gen_tokens.size() - no_repeat_ngram_size + 1); ++i) {
      std::vector<int> tokens(gen_tokens.begin() + i, gen_tokens.begin() + i + no_repeat_ngram_size);
      gen_tokens_ngram.emplace_back(tokens);
    }
    for (auto &ngram : gen_tokens_ngram) {
      std::vector<int> key(ngram.begin(), ngram.begin() + ngram.size() - 1);
      gen_ngrams[bbsz_idx][key].emplace_back(ngram.back());
    }
  }
  std::vector<std::vector<int>> banned_tokens;
  if (step + ngram_step_size - no_repeat_ngram_size >= 0) {
    for (int bbsz_idx = 0; bbsz_idx < real_batch_size; ++bbsz_idx) {
      std::vector<int> banned_token =
        calculate_banned_tokens(state_seq, step, gen_ngrams, no_repeat_ngram_size, bbsz_idx);
      banned_tokens.emplace_back(banned_token);
    }
  }
  for (int bbsz_idx = 0; bbsz_idx < real_batch_size; ++bbsz_idx) {
    for (auto &token : banned_tokens[bbsz_idx]) {
      log_probs[bbsz_idx * vocab_size + token] = static_cast<T>(-FLT_MAX);
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kNoRepeatNGram, NoRepeatNGramCpuKernel);
}  // namespace aicpu
