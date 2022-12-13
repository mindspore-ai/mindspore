/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_GRU_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_GRU_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <utility>

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
class GruGpuKernelMod;
using GruGpuKernelFunc = std::function<bool(GruGpuKernelMod *, const std::vector<AddressPtr> &,
                                            const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *)>;

class GruGpuKernelMod : public NativeGpuKernelMod {
 public:
  GruGpuKernelMod()
      : batch_size_(0),
        seq_len_(0),
        input_size_(0),
        hidden_size_(0),
        num_layers_(0),
        has_bias_(false),
        bidirectional_(false),
        states_init_(false),
        is_null_input_(false),
        dropout_(0),
        weight_size_(0),
        reserved_size_(0),
        x_desc_(nullptr),
        hx_desc_(nullptr),
        cx_desc_(nullptr),
        w_desc_(nullptr),
        dropout_desc_(nullptr),
        y_desc_(nullptr),
        hy_desc_(nullptr),
        cy_desc_(nullptr),
        rnn_desc_(nullptr),
        handle_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT) {}
  ~GruGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  void DestroyResource() noexcept override;
  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  void InitResource() override;

 private:
  void CreateTensorDescGrp();
  void ResetResource() noexcept;
  void InitSizeLists();
  int CheckInputsShape(const std::vector<KernelTensorPtr> &inputs);

  int batch_size_;
  int seq_len_;
  int input_size_;
  int hidden_size_;
  int num_layers_;
  bool has_bias_;
  bool bidirectional_;
  bool states_init_;
  bool is_null_input_;
  float dropout_;
  size_t weight_size_;
  size_t reserved_size_;
  size_t input_type_size_;  // sizeof(T)
  GruGpuKernelFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, GruGpuKernelFunc>> func_list_;

  // input desc
  std::unique_ptr<cudnnTensorDescriptor_t[]> x_desc_;
  cudnnTensorDescriptor_t hx_desc_;
  cudnnTensorDescriptor_t cx_desc_;
  cudnnFilterDescriptor_t w_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;
  std::unique_ptr<cudnnTensorDescriptor_t[]> y_desc_;
  cudnnTensorDescriptor_t hy_desc_;
  cudnnTensorDescriptor_t cy_desc_;
  cudnnRNNDescriptor_t rnn_desc_;
  cudnnHandle_t handle_;
  cudaStream_t cuda_stream_;
  cudnnDataType_t cudnn_data_type_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_GRU_GPU_KERNEL_H_
