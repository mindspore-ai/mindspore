/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONV2D_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONV2D_H_

#include <vector>
#include <string>
#include "src/tensor.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "schema/model_generated.h"
#include "nnacl/conv_parameter.h"
#include "schema/ops_generated.h"

namespace mindspore::kernel {

using lite::opencl::MemType;

constexpr size_t CI_TILE = C4NUM;
constexpr size_t CO_TILE = C4NUM;

enum FilterFormat {
  OHWI,            // CO KH KW CI
  HWII4OO4,        // KH KW CI/CI_TILE CI_TILE CO/CO_TILE CO_TILE
  OHWIOgroupI4O4,  // CO/Ogroup/CO_TILE KH KW CI/CI_TILE Ogroup CI_TILE CO_TILE
};

void ConvertFilter(void *src, void *dst, TypeId src_dtype, TypeId dst_dtype, FilterFormat src_format,
                   FilterFormat dst_format, size_t CO, size_t KH, size_t KW, size_t CI, size_t OGroup = 1);

class Conv2DOpenCLKernel : public OpenCLKernel {
 public:
  Conv2DOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : OpenCLKernel(parameter, inputs, outputs, ctx), param_(reinterpret_cast<ConvParameter *>(parameter)) {
    bool is_adreno = ocl_runtime_->GetGpuInfo().type == lite::opencl::GpuType::ADRENO;
    filter_type_ = is_adreno ? MemType::IMG : MemType::BUF;
  }
  ~Conv2DOpenCLKernel() override = default;

  int CheckSpecs() override;
  int Prepare() override;
  int InitWeights() override;
  void SetConstArgs() override;
  void SetGlobalLocal() override;
  int Run() override;

  std::string Key() override {
    auto key = OpenCLKernel::Key();
    key += "_" + std::to_string(KH_) + "_" + std::to_string(KW_) + "_" + std::to_string(param_->stride_h_) + "_" +
           std::to_string(param_->stride_w_) + "_" + std::to_string(param_->dilation_h_) + "_" +
           std::to_string(param_->dilation_w_);
    return key;
  }
  std::vector<BaseTuningParameter> GenerateTuningParam() override;
  int Tune() override { return OpenCLKernel::Tune(); }

  // for opencl fusion: Conv2D + PReLU(weight is scalar) -> param_.act_type=ActivationType_LEAKY_RELU
  float alpha_{0.0f};

  // for opencl fusion
  bool use_winograd_ = false;

 protected:
  void InitAttrs();
  virtual void BuildKernel();
  virtual void InitFilter();
  void InitBias();
  bool use_fp16_{false};
  size_t sizeof_FLT_{4};
  ConvParameter *param_{nullptr};
  int batch_size_{};
  int CI_{};
  int IH_{};
  int IW_{};
  int CO_{};
  int OH_{};
  int OW_{};
  int CI_SLICES_{};
  int CO_SLICES_{};
  int KH_{};
  int KW_{};
  void *packed_filter_{nullptr};
  void *packed_bias_{nullptr};
  MemType filter_type_{MemType::BUF};
  bool has_bias_{false};
  int TILE_HW_{};

 private:
  void SetBlockSize();
  struct {
    int H{1};
    int W{1};
    int C{1};
  } block_size_;
};

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONV2D_H_
