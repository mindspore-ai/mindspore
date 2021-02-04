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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CTCLOSS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CTCLOSS_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <limits>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/gpu_memory_allocator.h"
#include "backend/kernel_compiler/gpu/cuda_impl/ctcloss_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
class CtcLossGpuKernel : public GpuKernel {
 public:
  CtcLossGpuKernel()
      : label_indice_size_(0),
        label_size_(0),
        squence_lengths_size_(0),
        preprocess_collapse_repeated_(false),
        ctc_merge_repeated_(true),
        ignore_longer_outputs_than_inputs_(false) {}
  ~CtcLossGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    LaunchInit(inputs, workspace, outputs);
    LaunchFirstHalf(inputs, workspace, outputs, stream_ptr);
    LaunchSecondHalf(inputs, workspace, outputs, stream_ptr);
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    InitResource();
    auto probs_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (probs_shape.size() != 3) {
      MS_LOG(EXCEPTION) << "probs dims: " << probs_shape.size() << " not support.";
    }
    probs_dims_[0] = probs_shape[0];
    probs_dims_[1] = probs_shape[1];
    probs_dims_[2] = probs_shape[2];
    auto indice_dims = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto labels_dims = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    if (labels_dims.size() != 1) {
      MS_LOG(EXCEPTION) << "labels dims: " << labels_dims.size() << " not support.";
    }
    if (indice_dims.size() != 2) {
      MS_LOG(EXCEPTION) << "labels indice dims: " << indice_dims.size() << " not support.";
    }
    label_size_ = sizeof(int);
    for (auto i : labels_dims) {
      label_size_ *= i;
    }
    label_indice_size_ = sizeof(int64_t);
    for (auto i : indice_dims) {
      label_indice_size_ *= i;
    }
    auto squence_length_dims = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    squence_lengths_size_ = squence_length_dims[0] * sizeof(int);
    preprocess_collapse_repeated_ = GetAttr<bool>(kernel_node, "preprocess_collapse_repeated");
    ctc_merge_repeated_ = GetAttr<bool>(kernel_node, "ctc_merge_repeated");
    ignore_longer_outputs_than_inputs_ = GetAttr<bool>(kernel_node, "ignore_longer_outputs_than_inputs");
    InitSizeLists();
    return true;
  }

 protected:
  void LaunchInit(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                  const std::vector<AddressPtr> &outputs) {
    probs = GetDeviceAddress<T>(inputs, 0);
    label_indices = GetDeviceAddress<int64_t>(inputs, 1);
    label_values = GetDeviceAddress<int>(inputs, 2);
    sequence_length = GetDeviceAddress<int>(inputs, 3);
    costs = GetDeviceAddress<T>(outputs, 0);
    grads = GetDeviceAddress<T>(outputs, 1);
    softmax_probs = GetDeviceAddress<T>(workspace, 0);
    cum_labels_length = GetDeviceAddress<int>(workspace, 1);
    label_squence_length = GetDeviceAddress<int>(workspace, 2);
    label_value_sp = GetDeviceAddress<int>(workspace, 3);
    label_value_pcr = GetDeviceAddress<int>(workspace, 4);
    prob_num = GetDeviceAddress<T>(workspace, 5);
    precum_labels_length = GetDeviceAddress<int>(workspace, 6);
    max_labels_length = GetDeviceAddress<int>(workspace, 7);
    numclass = SizeToInt(probs_dims_[2]);
    batch = SizeToInt(probs_dims_[1]);
    max_time = SizeToInt(probs_dims_[0]);
    max_sequence = 0;
    max_labels_length_host = 0;
    batch_label = 0;
    label_value_with_blank = nullptr;
    log_alpha_b = nullptr;
    log_beta_b = nullptr;
  }

  void LaunchFirstHalf(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                       const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    CalculateMaxSequence(sequence_length, max_labels_length, batch, stream);
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_, cudaMemcpyAsync(&max_sequence, max_labels_length, sizeof(int), cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");
    if (max_time < max_sequence) {
      MS_LOG(EXCEPTION) << "max_time should be greater than sequence length.";
    }
    InnerSoftMax(probs, softmax_probs, sequence_length, max_time, batch, numclass, stream);
    MemsetForWS(label_value_pcr, cum_labels_length, label_squence_length, costs, grads, stream);

    CalculatePreLength(label_squence_length, precum_labels_length, cum_labels_length, max_labels_length, label_indices,
                       batch, label_size_ / sizeof(int), stream);
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_, cudaMemcpyAsync(&batch_label, max_labels_length, sizeof(int), cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");
    if (batch != batch_label + 1) {
      MS_LOG(EXCEPTION) << "label batch should be equal to input batch.";
    }
    GenLabelValue(label_value_sp, label_indices, label_values, label_squence_length, cum_labels_length,
                  max_labels_length, label_size_ / sizeof(int), numclass - 1, batch, stream);
    if (preprocess_collapse_repeated_) {
      GenLabelValuePCR(label_value_sp, label_value_pcr, label_squence_length, cum_labels_length, max_labels_length,
                       batch, stream);
    }
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpyAsync(&max_labels_length_host, max_labels_length, sizeof(int), cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");
  }

  void LaunchSecondHalf(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                        const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int SOffSet = 2 * max_labels_length_host + 1;
    int log_prob_size = batch * SOffSet * max_time;

    if (!ignore_longer_outputs_than_inputs_ && max_labels_length_host > max_time) {
      MS_LOG(EXCEPTION) << "output size is greater than input size.";
    }
    MemManageForCus(&log_alpha_b, &log_beta_b, &label_value_with_blank, cum_labels_length, log_prob_size, batch,
                    stream);

    if (preprocess_collapse_repeated_) {
      GenLabelWithBlank(label_value_pcr, label_value_with_blank, label_squence_length, precum_labels_length,
                        cum_labels_length, batch, numclass - 1, stream);
    } else {
      GenLabelWithBlank(label_value_sp, label_value_with_blank, label_squence_length, precum_labels_length,
                        cum_labels_length, batch, numclass - 1, stream);
    }

    CalculateFwdVar(log_alpha_b, label_value_with_blank, softmax_probs, sequence_length, ctc_merge_repeated_, batch,
                    SOffSet, max_time, numclass - 1, label_squence_length, cum_labels_length,
                    ignore_longer_outputs_than_inputs_, stream);
    CalculateBwdVar(log_beta_b, label_value_with_blank, softmax_probs, sequence_length, ctc_merge_repeated_, batch,
                    SOffSet, max_time, numclass - 1, label_squence_length, cum_labels_length,
                    ignore_longer_outputs_than_inputs_, stream);
    CTCLoss(log_alpha_b, log_beta_b, softmax_probs, label_value_with_blank, batch, SOffSet, max_time, numclass,
            sequence_length, label_squence_length, cum_labels_length, costs, grads, prob_num,
            ignore_longer_outputs_than_inputs_, stream);
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");
    FreeMem(label_value_with_blank, log_alpha_b, log_beta_b);
  }

  void InitSizeLists() override {
    input_size_list_.push_back(probs_dims_[0] * probs_dims_[1] * probs_dims_[2] * sizeof(T));
    input_size_list_.push_back(label_indice_size_);
    input_size_list_.push_back(label_size_);
    input_size_list_.push_back(squence_lengths_size_);
    workspace_size_list_.push_back(probs_dims_[0] * probs_dims_[1] * probs_dims_[2] * sizeof(T));
    workspace_size_list_.push_back(squence_lengths_size_);
    workspace_size_list_.push_back(squence_lengths_size_);
    workspace_size_list_.push_back(label_size_);
    workspace_size_list_.push_back(label_size_);
    workspace_size_list_.push_back(probs_dims_[0] * probs_dims_[1] * probs_dims_[2] * sizeof(T));
    workspace_size_list_.push_back(squence_lengths_size_);
    workspace_size_list_.push_back(sizeof(int));
    output_size_list_.push_back(probs_dims_[1] * sizeof(T));
    output_size_list_.push_back(probs_dims_[0] * probs_dims_[1] * probs_dims_[2] * sizeof(T));
  }
  void MemsetForWS(int *label_value_pcr, int *cum_labels_length, int *label_squence_length, T *costs, T *grads,
                   cudaStream_t stream) {
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaMemsetAsync(label_value_pcr, static_cast<int>(0), label_size_, stream),
                               "cudaMemSet failed in CtcLossGpuKernel::Launch.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemsetAsync(cum_labels_length, static_cast<int>(0), squence_lengths_size_, stream),
                               "cudaMemSet failed in CtcLossGpuKernel::Launch.");
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_, cudaMemsetAsync(label_squence_length, static_cast<int>(0), squence_lengths_size_, stream),
      "cudaMemSet failed in CtcLossGpuKernel::Launch.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemsetAsync(costs, static_cast<T>(0), probs_dims_[1] * sizeof(T), stream),
                               "cudaMemSet failed in CtcLossGpuKernel::Launch.");
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemsetAsync(grads, static_cast<T>(0), probs_dims_[0] * probs_dims_[1] * probs_dims_[2] * sizeof(T), stream),
      "cudaMemSet failed in CtcLossGpuKernel::Launch.");
  }
  void MemManageForCus(T **log_alpha_b, T **log_beta_b, int **label_value_with_blank, int *cum_labels_length,
                       int log_prob_size, int batch, cudaStream_t stream) {
    int total_labels_size_host = 0;
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMalloc(reinterpret_cast<void **>(log_alpha_b), sizeof(T) * log_prob_size),
                               "cudaMalloc failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_, cudaMalloc(reinterpret_cast<void **>(log_beta_b), sizeof(T) * log_prob_size), "cudaMalloc failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&total_labels_size_host, cum_labels_length + batch - 1, sizeof(int),
                                               cudaMemcpyDeviceToHost, stream),
                               "cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMalloc(reinterpret_cast<void **>(label_value_with_blank), sizeof(int) * (2 * total_labels_size_host + batch)),
      "cudaMalloc failed.");
  }

  void FreeMem(int *label_value_with_blank, T *log_alpha_b, T *log_beta_b) {
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaFree(label_value_with_blank), "cudaFree failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaFree(log_alpha_b), "cudaFree failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaFree(log_beta_b), "cudaFree failed.");
  }

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  size_t probs_dims_[3] = {0};
  int label_indice_size_;
  int label_size_;
  int squence_lengths_size_;
  bool preprocess_collapse_repeated_;
  bool ctc_merge_repeated_;
  bool ignore_longer_outputs_than_inputs_;
  T kLogZero_ = -std::numeric_limits<T>::infinity();

  // Heap parameter
  T *probs;
  int64_t *label_indices;
  int *label_values;
  int *sequence_length;
  T *costs;
  T *grads;
  T *softmax_probs;
  int *cum_labels_length;
  int *label_squence_length;
  int *label_value_sp;
  int *label_value_pcr;
  T *prob_num;
  int *precum_labels_length;
  int *max_labels_length;
  int numclass;
  int batch;
  int max_time;
  int max_sequence;
  int max_labels_length_host;
  int batch_label;
  int *label_value_with_blank;
  T *log_alpha_b;
  T *log_beta_b;
};  // namespace kernel
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CTCLOSS_GPU_KERNEL_H_
