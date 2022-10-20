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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_ARITHMETIC_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_ARITHMETIC_BASE_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/arithmetic.h"

namespace mindspore::kernel {
class ArithmeticBaseCPUKernel : public LiteKernel {
 public:
  ArithmeticBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<ArithmeticParameter *>(parameter);
  }
  ~ArithmeticBaseCPUKernel() override {
    for (auto buffer : broadcast_buffer_) {
      ms_context_->allocator->Free(buffer);
    }
    broadcast_buffer_.clear();
  }

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoArithmetic(int task_id);

 protected:
  template <class T>
  using ArithmeticFunc = std::function<int(const T *, const T *, T *, const int)>;
  template <class T>
  using ArithmeticOptFunc = std::function<int(const T *, const T *, T *, const int, const ArithmeticParameter *)>;
  struct MatricInfo {
    bool is_const{false};
    bool is_valid{false};
    void *data{nullptr};
    int64_t inner_size{1};  // the element num of once batch
    std::vector<int64_t> shape;
    std::vector<int64_t> batch_post_sum;
    void Reset() {
      is_valid = false;
      data = nullptr;
      inner_size = 1;
      shape.clear();
      batch_post_sum.clear();
    }
  };

  virtual void DoBroadcast(void *out_data, int input_index) = 0;
  virtual void InitRunFunction(int primitive_type) = 0;
  virtual int DoExecute(const void *input0, const void *input1, void *output, int64_t size) = 0;
  bool scalar_opt_{false};
  int in_data_size_{0};
  int out_data_size_{0};
  ArithmeticParameter *param_{nullptr};
  MatricInfo a_matric_;
  MatricInfo b_matric_;
  MatricInfo c_matric_;

 private:
  struct BlockBoundaryInfo {
    int64_t batch_begin{0};
    int64_t batch_end{0};
    int64_t size_begin{0};  // start-offset under the begin batch
    int64_t size_end{0};    // end-num under the ending batch
    std::vector<int64_t> a_offset;
    std::vector<int64_t> b_offset;
  };
  int ResetStatus();
  // horizon: {[2, 3, 4], [1, 1, 1]} -> {[24], [1]}, vertical: {[2, 3, 4], [3, 4]} -> {[2, 12], [12]}
  int OptimizeShape();
  int UpdateParameter();
  int BroadCastConstTensor();
  int ComputeOfflineInfo();
  int ChooseThreadCuttingStrategy();
  void ComputeOffset(int task_id);
  int primitive_type_{0};
  int batch_tail_dim_{-1};
  std::vector<void *> broadcast_buffer_;
  std::vector<BlockBoundaryInfo> block_boundary_infos_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_ARITHMETIC_BASE_H_
