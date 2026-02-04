/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <string>
#include <limits>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "plugin/gpu/res_manager/mem_manager/gpu_memory_allocator.h"
#include "kernel/gpu/cuda_impl/cuda_ops/ctcloss_impl.cuh"
namespace mindspore {
namespace kernel {
constexpr size_t kPrevOutput0th = 0;
constexpr size_t kPrevOutput1st = 1;
constexpr size_t kPrevOutput2nd = 2;
constexpr size_t kPrevOutput3rd = 3;
constexpr size_t kProbDimSize = 3;
constexpr size_t kIndicesDimSize = 2;
constexpr size_t kInputIdxForProbs = 0;
constexpr size_t kInputIdxForLabelIndices = 1;
constexpr size_t kInputIdxForLabelValues = 2;
constexpr size_t kInputIdxForSeqLen = 3;
constexpr size_t kWsIdxForSoftmaxProbs = 0;
constexpr size_t kWsIdxForCumLabelLen = 1;
constexpr size_t kWsIdxForLabelSquenceLen = 2;
constexpr size_t kWsIdxForLabelValueSp = 3;
constexpr size_t kWsIdxForLabelValuePcr = 4;
constexpr size_t kWsIdxForProbNum = 5;
constexpr size_t kWsIdxForPrecumLabelLen = 6;
constexpr size_t kWsIdxForMaxLabelLen = 7;
constexpr size_t kProbDimsIdxForMaxTime = 0;
constexpr size_t kProbDimsIdxForBatch = 1;
constexpr size_t kProbDimsIdxForNumClass = 2;
constexpr size_t kCTCLossInputsNum = 4;
constexpr size_t kCTCLossOutputsNum = 2;

template <typename T>
class CtcLossGpuKernelMod : public NativeGpuKernelMod {
 public:
  CtcLossGpuKernelMod()
      : label_indice_size_(0),
        label_size_(0),
        sequence_lengths_size_(0),
        preprocess_collapse_repeated_(false),
        ctc_merge_repeated_(true),
        ignore_longer_outputs_than_inputs_(false),
        is_null_input_(false),
        kernel_name_("CTCLoss"),
        probs(nullptr),
        label_indices(nullptr),
        label_values(nullptr),
        sequence_length(nullptr),
        costs(nullptr),
        grads(nullptr),
        softmax_probs(nullptr),
        cum_labels_length(nullptr),
        label_squence_length(nullptr),
        label_value_sp(nullptr),
        label_value_pcr(nullptr),
        prob_num(nullptr),
        precum_labels_length(nullptr),
        max_labels_length(nullptr),
        numclass(0),
        batch(0),
        max_time(0),
        max_sequence(0),
        max_labels_length_host(0),
        batch_label(0),
        label_value_with_blank(nullptr),
        log_alpha_b(nullptr),
        log_beta_b(nullptr) {}
  ~CtcLossGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    LaunchInit(inputs, workspace, outputs);
    LaunchFirstHalf(inputs, workspace, outputs, stream_ptr);
    LaunchSecondHalf(inputs, workspace, outputs, stream_ptr);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCTCLossInputsNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCTCLossOutputsNum, kernel_name_);

    PrimitivePtr prim = primitive_;
    MS_EXCEPTION_IF_NULL(prim);

    preprocess_collapse_repeated_ = GetValue<bool>(prim->GetAttr("preprocess_collapse_repeated"));
    ctc_merge_repeated_ = GetValue<bool>(prim->GetAttr("ctc_merge_repeated"));
    ignore_longer_outputs_than_inputs_ = GetValue<bool>(prim->GetAttr("ignore_longer_outputs_than_inputs"));
    InitResource();
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    auto shape_signed = inputs[kPrevOutput0th]->GetShapeVector();
    auto probs_shape = Convert2SizeTClipNeg(shape_signed);
    auto indice_dims = inputs[kPrevOutput1st]->GetShapeVector();
    auto labels_dims = inputs[kPrevOutput2nd]->GetShapeVector();
    auto sequence_length_dims = inputs[kPrevOutput3rd]->GetShapeVector();
    is_null_input_ = CHECK_SHAPE_NULL(probs_shape, kernel_name_, "x") ||
                     CHECK_SHAPE_NULL(indice_dims, kernel_name_, "labels_indices") ||
                     CHECK_SHAPE_NULL(labels_dims, kernel_name_, "labels_values") ||
                     CHECK_SHAPE_NULL(sequence_length_dims, kernel_name_, "sequence_length");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    if (probs_shape.size() != kProbDimSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of x must be 3, but got " << probs_shape.size();
    }
    probs_dims_[kProbDimsIdxForMaxTime] = probs_shape[kProbDimsIdxForMaxTime];
    probs_dims_[kProbDimsIdxForBatch] = probs_shape[kProbDimsIdxForBatch];
    probs_dims_[kProbDimsIdxForNumClass] = probs_shape[kProbDimsIdxForNumClass];

    if (labels_dims.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of labels_values must be 1, but got "
                        << labels_dims.size();
    }
    if (indice_dims.size() != kIndicesDimSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of labels_indices must be 2, but got "
                        << indice_dims.size();
    }
    label_size_ = sizeof(int);
    label_size_ *= SizeOf(labels_dims);
    label_indice_size_ = sizeof(int64_t);
    label_indice_size_ *= SizeOf(indice_dims);

    sequence_lengths_size_ = LongToSizeClipNeg(sequence_length_dims[0]) * sizeof(int);
    InitSizeLists();
    return KRET_OK;
  }

 protected:
  void LaunchInit(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                  const std::vector<KernelTensor *> &outputs) {
    probs = GetDeviceAddress<T>(inputs, kInputIdxForProbs);
    label_indices = GetDeviceAddress<int64_t>(inputs, kInputIdxForLabelIndices);
    label_values = GetDeviceAddress<int>(inputs, kInputIdxForLabelValues);
    sequence_length = GetDeviceAddress<int>(inputs, kInputIdxForSeqLen);
    costs = GetDeviceAddress<T>(outputs, 0);
    grads = GetDeviceAddress<T>(outputs, 1);
    softmax_probs = GetDeviceAddress<T>(workspace, kWsIdxForSoftmaxProbs);
    cum_labels_length = GetDeviceAddress<int>(workspace, kWsIdxForCumLabelLen);
    label_squence_length = GetDeviceAddress<int>(workspace, kWsIdxForLabelSquenceLen);
    label_value_sp = GetDeviceAddress<int>(workspace, kWsIdxForLabelValueSp);
    label_value_pcr = GetDeviceAddress<int>(workspace, kWsIdxForLabelValuePcr);
    prob_num = GetDeviceAddress<T>(workspace, kWsIdxForProbNum);
    precum_labels_length = GetDeviceAddress<int>(workspace, kWsIdxForPrecumLabelLen);
    max_labels_length = GetDeviceAddress<int>(workspace, kWsIdxForMaxLabelLen);
    numclass = SizeToInt(probs_dims_[kProbDimsIdxForNumClass]);
    batch = SizeToInt(probs_dims_[kProbDimsIdxForBatch]);
    max_time = SizeToInt(probs_dims_[kProbDimsIdxForMaxTime]);
    max_sequence = 0;
    max_labels_length_host = 0;
    batch_label = 0;
    label_value_with_blank = nullptr;
    log_alpha_b = nullptr;
    log_beta_b = nullptr;
  }

  void LaunchFirstHalf(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaError_t status = cudaErrorNotReady;
    status = CalculateMaxSequence(sequence_length, max_labels_length, batch, stream);
    CHECK_CUDA_STATUS(status, "CalculateMaxSequence called by " + kernel_name_);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(&max_sequence, max_labels_length, sizeof(int), cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");
    if (max_time < max_sequence) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the x[0] must be equal to or greater than max_sequence, "
                        << "but got x[0]: " << max_time << ", max_sequence: " << max_sequence;
    }
    status = InnerSoftMax(probs, softmax_probs, sequence_length, max_time, batch, numclass, stream);
    CHECK_CUDA_STATUS(status, "InnerSoftMax called by " + kernel_name_);
    MemsetForWS(label_value_pcr, cum_labels_length, label_squence_length, costs, grads, stream);
    status = CalculatePreLength(label_squence_length, precum_labels_length, cum_labels_length, max_labels_length,
                                label_indices, batch, label_size_ / sizeof(int), stream);
    CHECK_CUDA_STATUS(status, "CalculatePreLength called by " + kernel_name_);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(&batch_label, max_labels_length, sizeof(int), cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");
    if (batch != batch_label + 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the batch size of input must be equal to "
                        << (batch_label + 1) << ", but got " << batch;
    }
    status = GenLabelValue(label_value_sp, label_indices, label_values, label_squence_length, cum_labels_length,
                           max_labels_length, label_size_ / sizeof(int), numclass - 1, batch, stream);
    CHECK_CUDA_STATUS(status, "GenLabelValue called by " + kernel_name_);
    if (preprocess_collapse_repeated_) {
      status = GenLabelValuePCR(label_value_sp, label_value_pcr, label_squence_length, cum_labels_length,
                                max_labels_length, batch, stream);
      CHECK_CUDA_STATUS(status, "GenLabelValuePCR called by " + kernel_name_);
    }
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(&max_labels_length_host, max_labels_length, sizeof(int), cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");
  }

  void LaunchSecondHalf(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    int64_t temp_result = 2 * static_cast<int64_t>(max_labels_length_host) + 1;
    if (temp_result > static_cast<int64_t>(std::numeric_limits<int>::max())) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', SOffSet calculation overflow: "
                        << "max_labels_length_host=" << max_labels_length_host << ", result would be " << temp_result
                        << ", max allowed=" << std::numeric_limits<int>::max();
    }
    const int SOffSet = static_cast<int>(temp_result);

    temp_result = static_cast<int64_t>(batch) * static_cast<int64_t>(SOffSet) * static_cast<int64_t>(max_time);
    if (temp_result > static_cast<int64_t>(std::numeric_limits<int>::max())) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', log_prob_size calculation overflow: "
                        << "batch=" << batch << ", SOffSet=" << SOffSet << ", max_time=" << max_time
                        << ", result would be " << temp_result << ", max allowed=" << std::numeric_limits<int>::max();
    }
    int log_prob_size = static_cast<int>(temp_result);
    cudaError_t status = cudaErrorNotReady;

    if (!ignore_longer_outputs_than_inputs_ && max_labels_length_host > max_time) {
      MS_LOG(EXCEPTION) << "output size is greater than input size.";
    }
    MemManageForCus(&log_alpha_b, &log_beta_b, &label_value_with_blank, cum_labels_length, log_prob_size, batch,
                    stream);

    if (preprocess_collapse_repeated_) {
      status = GenLabelWithBlank(label_value_pcr, label_value_with_blank, label_squence_length, precum_labels_length,
                                 cum_labels_length, batch, numclass - 1, stream);
    } else {
      status = GenLabelWithBlank(label_value_sp, label_value_with_blank, label_squence_length, precum_labels_length,
                                 cum_labels_length, batch, numclass - 1, stream);
    }
    CHECK_CUDA_STATUS(status, "GenLabelWithBlank called by " + kernel_name_);

    status = CalculateFwdVar(log_alpha_b, label_value_with_blank, softmax_probs, sequence_length, ctc_merge_repeated_,
                             batch, SOffSet, max_time, numclass - 1, label_squence_length, cum_labels_length,
                             ignore_longer_outputs_than_inputs_, stream);
    CHECK_CUDA_STATUS(status, "CalculateFwdVar called by " + kernel_name_);
    status = CalculateBwdVar(log_beta_b, label_value_with_blank, softmax_probs, sequence_length, ctc_merge_repeated_,
                             batch, SOffSet, max_time, numclass - 1, label_squence_length, cum_labels_length,
                             ignore_longer_outputs_than_inputs_, stream);
    CHECK_CUDA_STATUS(status, "CalculateBwdVar called by " + kernel_name_);
    status = CTCLoss(log_alpha_b, log_beta_b, softmax_probs, label_value_with_blank, batch, SOffSet, max_time, numclass,
                     sequence_length, label_squence_length, cum_labels_length, costs, grads, prob_num,
                     ignore_longer_outputs_than_inputs_, stream);
    CHECK_CUDA_STATUS(status, kernel_name_);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");
    FreeMem(label_value_with_blank, log_alpha_b, log_beta_b);
  }

  void InitSizeLists() {
    workspace_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.push_back(probs_dims_[kProbDimsIdxForMaxTime] * probs_dims_[kProbDimsIdxForBatch] *
                                   probs_dims_[kProbDimsIdxForNumClass] * sizeof(T));
    workspace_size_list_.push_back(sequence_lengths_size_);
    workspace_size_list_.push_back(sequence_lengths_size_);
    workspace_size_list_.push_back(label_size_);
    workspace_size_list_.push_back(label_size_);
    workspace_size_list_.push_back(probs_dims_[kProbDimsIdxForMaxTime] * probs_dims_[kProbDimsIdxForBatch] *
                                   probs_dims_[kProbDimsIdxForNumClass] * sizeof(T));
    workspace_size_list_.push_back(sequence_lengths_size_);
    workspace_size_list_.push_back(sizeof(int));
    output_size_list_.push_back(probs_dims_[kProbDimsIdxForBatch] * sizeof(T));
    output_size_list_.push_back(probs_dims_[kProbDimsIdxForMaxTime] * probs_dims_[kProbDimsIdxForBatch] *
                                probs_dims_[kProbDimsIdxForNumClass] * sizeof(T));
  }
  void MemsetForWS(int *label_value_pcr, int *cum_labels_length, int *label_squence_length, T *costs, T *grads,
                   cudaStream_t stream) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(label_value_pcr, static_cast<int>(0), label_size_, stream),
                                       "cudaMemSet failed in CtcLossGpuKernelMod::Launch.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemsetAsync(cum_labels_length, static_cast<int>(0), sequence_lengths_size_, stream),
      "cudaMemSet failed in CtcLossGpuKernelMod::Launch.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemsetAsync(label_squence_length, static_cast<int>(0), sequence_lengths_size_, stream),
      "cudaMemSet failed in CtcLossGpuKernelMod::Launch.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemsetAsync(costs, static_cast<T>(0), probs_dims_[kProbDimsIdxForBatch] * sizeof(T), stream),
      "cudaMemSet failed in CtcLossGpuKernelMod::Launch.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemsetAsync(grads, static_cast<T>(0),
                      probs_dims_[kProbDimsIdxForMaxTime] * probs_dims_[kProbDimsIdxForBatch] *
                        probs_dims_[kProbDimsIdxForNumClass] * sizeof(T),
                      stream),
      "cudaMemSet failed in CtcLossGpuKernelMod::Launch.");
  }
  void MemManageForCus(T **log_alpha_b, T **log_beta_b, int **label_value_with_blank, int *cum_labels_length,
                       int log_prob_size, int batch, cudaStream_t stream) {
    int total_labels_size_host = 0;
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMalloc(reinterpret_cast<void **>(log_alpha_b), sizeof(T) * log_prob_size),
                                       "cudaMalloc failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMalloc(reinterpret_cast<void **>(log_beta_b), sizeof(T) * log_prob_size),
                                       "cudaMalloc failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&total_labels_size_host, cum_labels_length + batch - 1,
                                                       sizeof(int), cudaMemcpyDeviceToHost, stream),
                                       "cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMalloc(reinterpret_cast<void **>(label_value_with_blank), sizeof(int) * (2 * total_labels_size_host + batch)),
      "cudaMalloc failed.");
  }

  void FreeMem(int *label_value_with_blank, T *log_alpha_b, T *log_beta_b) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaFree(label_value_with_blank), "cudaFree failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaFree(log_alpha_b), "cudaFree failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaFree(log_beta_b), "cudaFree failed.");
  }

  size_t probs_dims_[3] = {0};
  int label_indice_size_;
  int label_size_;
  int sequence_lengths_size_;
  bool preprocess_collapse_repeated_;
  bool ctc_merge_repeated_;
  bool ignore_longer_outputs_than_inputs_;
  bool is_null_input_;
  std::string kernel_name_;
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
