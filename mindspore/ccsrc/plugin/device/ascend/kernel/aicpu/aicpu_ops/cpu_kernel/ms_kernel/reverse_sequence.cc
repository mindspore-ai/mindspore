/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "reverse_sequence.h"
#include "Eigen/Core"
#include "context/inc/cpu_kernel_utils.h"
#include "log.h"
#include "utils/kernel_util.h"

namespace {
const char *const kReverseSequence = "ReverseSequence";
const int kOutputIndex = 2;
const int64_t kEven = 2;
}  // namespace

namespace aicpu {
template <typename Tlen>
static uint32_t CheckSequence(CpuKernelContext &ctx, size_t seq_dim, const Tlen *seq, std::vector<int64_t> &shape,
                              std::vector<int64_t> &seq_lengths_shape) {
  for (int64_t d = 0; d < static_cast<int64_t>(seq_lengths_shape[0]); d++) {
    if (seq[d] < 0 || seq[d] > shape[seq_dim]) {
      CUST_KERNEL_LOG_ERROR(ctx, "Invalid seq_lengths[%d]: %lu", d, seq[d]);
      return static_cast<uint32_t>(KERNEL_STATUS_INNER_ERROR);
    }
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

template <typename T, typename Tlen>
uint32_t CalReverseSequence(size_t seq_dim, size_t batch_dim, const std::vector<void *> &ioAddrs,
                            std::vector<int64_t> &shape, std::vector<int64_t> &seq_lengths_shape,
                            CpuKernelContext &ctx) {
  // inputs
  T *input = reinterpret_cast<T *>(ioAddrs[0]);
  Tlen *seq = reinterpret_cast<Tlen *>(ioAddrs[1]);
  // outputs
  T *output = reinterpret_cast<T *>(ioAddrs[kOutputIndex]);

  if (CheckSequence(ctx, seq_dim, seq, shape, seq_lengths_shape) != static_cast<uint32_t>(KERNEL_STATUS_OK)) {
    return static_cast<uint32_t>(KERNEL_STATUS_INNER_ERROR);
  }

  int64_t seq_step = 1;
  size_t shape_size = shape.size();
  for (size_t i = seq_dim + 1U; i < shape_size; i++) {
    seq_step *= shape[i];
  }

  int64_t run_len = seq_step;
  int64_t batch_size = 1;
  for (size_t i = batch_dim + 1U; i < shape_size; ++i) {
    batch_size *= shape[i];
  }

  int64_t total_size = 1;
  for (size_t i = 0; i < shape_size; ++i) {
    total_size *= shape[i];
  }
  CUST_KERNEL_CHECK_FALSE(ctx, (shape[seq_dim] != 0), static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID),
                          "The shape[%zu] of input[0] cannot be 0.", seq_dim);
  CUST_KERNEL_CHECK_FALSE(ctx, (batch_size != 0), static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID),
                          "The value of batch_size cannot be 0.");
  CUST_KERNEL_CHECK_FALSE(ctx, (shape[batch_dim] != 0), static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID),
                          "The shape[%zu] of input[0] cannot be 0.", batch_dim);
  int64_t n = total_size / (run_len * shape[seq_dim]);
  const int64_t kMaxCoreNum = std::max(static_cast<uint32_t>(1), aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

  auto shard = [&](const int64_t start, const int64_t end) {
    for (int64_t j = start; j < end; ++j) {  // 0~n
      int64_t begin = run_len * shape[seq_dim] * j;
      for (int64_t r = 0; r < run_len; ++r) {
        int64_t offset = r + begin;
        int64_t reverse_num = static_cast<int64_t>(seq[offset / batch_size % shape[batch_dim]]);
        for (int64_t i = 0; i < shape[seq_dim]; ++i) {
          if (i < reverse_num / kEven) {
            output[i * seq_step + offset] = input[((reverse_num - i) - 1) * seq_step + offset];
            output[((reverse_num - i) - 1) * seq_step + offset] = input[i * seq_step + offset];
          }
          if (i >= reverse_num || (i == reverse_num / kEven && reverse_num % kEven)) {
            output[i * seq_step + offset] = input[i * seq_step + offset];
          }
        }
      }
    }
  };
  uint32_t ret = CpuKernelUtils::ParallelFor(ctx, n, n / kMaxCoreNum, shard);
  if (ret != static_cast<uint32_t>(KERNEL_STATUS_OK)) {
    CUST_KERNEL_LOG_ERROR(ctx, "CpuKernelUtils::ParallelFor failed");
    return static_cast<uint32_t>(KERNEL_STATUS_INNER_ERROR);
  }

  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

KernelStatus ReverseSequenceMsCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  AttrValue *seq_dim = ctx.GetAttr("seq_dim");
  CUST_KERNEL_CHECK_NULLPTR(ctx, seq_dim, KERNEL_STATUS_PARAM_INVALID, "Get attr:[seq_dim] failed.");
  seq_dim_ = static_cast<size_t>(seq_dim->GetInt());

  AttrValue *batch_dim = ctx.GetAttr("batch_dim");
  CUST_KERNEL_CHECK_NULLPTR(ctx, batch_dim, KERNEL_STATUS_PARAM_INVALID, "Get attr:[batch_dim] failed.");
  batch_dim_ = static_cast<size_t>(batch_dim->GetInt());

  // input_0: x
  Tensor *x_tensor = ctx.Input(0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, x_tensor, KERNEL_STATUS_PARAM_INVALID, "Get input:[0] failed")
  x_dtype_ = static_cast<DataType>(x_tensor->GetDataType());
  std::shared_ptr<TensorShape> x_shape = x_tensor->GetTensorShape();

  for (auto i = 0; i < x_shape->GetDims(); i++) {
    x_shape_.emplace_back(x_shape->GetDimSize(i));
  }

  // input_1: seq_lengths
  Tensor *seq_lengths_tensor = ctx.Input(1);
  CUST_KERNEL_CHECK_NULLPTR(ctx, seq_lengths_tensor, KERNEL_STATUS_PARAM_INVALID, "Get input:[1] failed")
  seq_lengths_dtype_ = static_cast<DataType>(seq_lengths_tensor->GetDataType());
  std::shared_ptr<TensorShape> seq_lengths_shape = seq_lengths_tensor->GetTensorShape();
  for (auto i = 0; i < seq_lengths_shape->GetDims(); i++) {
    seq_lengths_shape_.emplace_back(seq_lengths_shape->GetDimSize(i));
  }

  if (seq_lengths_dtype_ != DT_INT32 && seq_lengths_dtype_ != DT_INT64) {
    CUST_KERNEL_LOG_ERROR(ctx, "Invalid type of seq_lengths: [%s]", DTypeStr(seq_lengths_dtype_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (seq_lengths_shape_.size() != 1) {
    CUST_KERNEL_LOG_ERROR(ctx, "Invalid seq_lengths shape size: [%d]", seq_lengths_shape_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (batch_dim_ == seq_dim_ || seq_dim_ >= x_shape_.size() || batch_dim_ >= x_shape_.size()) {
    CUST_KERNEL_LOG_ERROR(ctx, "Invalid batch_dim_: [%zu], seq_dim_: [%zu], x dims:[ %zu]", batch_dim_, seq_dim_,
                          x_shape_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (seq_lengths_shape_[0] != x_shape->GetDimSize(static_cast<int32_t>(batch_dim_))) {
    CUST_KERNEL_LOG_ERROR(ctx, "seq_lengths_shape_[0] != x_shape.dim(%zu) size: [%lld]", batch_dim_,
                          x_shape->GetDimSize(static_cast<int32_t>(batch_dim_)));
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *output_tensor = ctx.Output(0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, output_tensor, KERNEL_STATUS_PARAM_INVALID, "Get output:[0] failed")
  ioAddrs_.push_back(reinterpret_cast<void *>(x_tensor->GetData()));
  ioAddrs_.push_back(reinterpret_cast<void *>(seq_lengths_tensor->GetData()));
  ioAddrs_.push_back(reinterpret_cast<void *>(output_tensor->GetData()));

  CUST_KERNEL_LOG_INFO(ctx, "Parse done, seq_dim: [%zu], batch_dim: %zu, x_dtype: [%d]", seq_dim_, batch_dim_,
                       static_cast<int32_t>(x_dtype_));

  return KERNEL_STATUS_OK;
}

uint32_t ReverseSequenceMsCpuKernel::Compute(CpuKernelContext &ctx) {
  KernelStatus res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return static_cast<uint32_t>(res);
  }

  std::map<DataType,
           std::map<DataType, std::function<uint32_t(size_t, size_t, std::vector<void *> &, std::vector<int64_t> &,
                                                     std::vector<int64_t> &, CpuKernelContext &)>>>
    calls;
  calls[DT_FLOAT16][DT_INT32] = CalReverseSequence<Eigen::half, int32_t>;
  calls[DT_FLOAT][DT_INT32] = CalReverseSequence<float, int32_t>;
  calls[DT_DOUBLE][DT_INT32] = CalReverseSequence<double, int32_t>;
  calls[DT_INT8][DT_INT32] = CalReverseSequence<int8_t, int32_t>;
  calls[DT_INT16][DT_INT32] = CalReverseSequence<int16_t, int32_t>;
  calls[DT_INT32][DT_INT32] = CalReverseSequence<int32_t, int32_t>;
  calls[DT_INT64][DT_INT32] = CalReverseSequence<int64_t, int32_t>;
  calls[DT_UINT8][DT_INT32] = CalReverseSequence<uint8_t, int32_t>;
  calls[DT_UINT16][DT_INT32] = CalReverseSequence<uint16_t, int32_t>;
  calls[DT_UINT32][DT_INT32] = CalReverseSequence<uint32_t, int32_t>;
  calls[DT_UINT64][DT_INT32] = CalReverseSequence<uint64_t, int32_t>;
  calls[DT_BOOL][DT_INT32] = CalReverseSequence<bool, int32_t>;

  calls[DT_FLOAT16][DT_INT64] = CalReverseSequence<Eigen::half, int64_t>;
  calls[DT_FLOAT][DT_INT64] = CalReverseSequence<float, int64_t>;
  calls[DT_DOUBLE][DT_INT64] = CalReverseSequence<double, int64_t>;
  calls[DT_INT8][DT_INT64] = CalReverseSequence<int8_t, int64_t>;
  calls[DT_INT16][DT_INT64] = CalReverseSequence<int16_t, int64_t>;
  calls[DT_INT32][DT_INT64] = CalReverseSequence<int32_t, int64_t>;
  calls[DT_INT64][DT_INT64] = CalReverseSequence<int64_t, int64_t>;
  calls[DT_UINT8][DT_INT64] = CalReverseSequence<uint8_t, int64_t>;
  calls[DT_UINT16][DT_INT64] = CalReverseSequence<uint16_t, int64_t>;
  calls[DT_UINT32][DT_INT64] = CalReverseSequence<uint32_t, int64_t>;
  calls[DT_UINT64][DT_INT64] = CalReverseSequence<uint64_t, int64_t>;
  calls[DT_BOOL][DT_INT64] = CalReverseSequence<bool, int64_t>;

  return calls[x_dtype_][seq_lengths_dtype_](seq_dim_, batch_dim_, ioAddrs_, x_shape_, seq_lengths_shape_, ctx);
}

REGISTER_MS_CPU_KERNEL(kReverseSequence, ReverseSequenceMsCpuKernel);
}  // namespace aicpu
