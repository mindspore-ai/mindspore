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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BOLT_CONVOLUTION_BOLT_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BOLT_CONVOLUTION_BOLT_H_

#include <memory>
#include <vector>
#include "bolt/bolt_kernel.h"
#include "bolt/common/uni/include/algorithm_map.h"

namespace mindspore::kernel::bolt {
class ConvolutionBoltCPUKernel : public BoltKernel {
 public:
  ConvolutionBoltCPUKernel(const ParameterSpec &param_spec, const std::vector<lite::Tensor *> &inputs,
                           const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : BoltKernel(param_spec, inputs, outputs, ctx) {
    conv_param_spec_ = param_spec.conv_spec;
    pw_act_param_.mode = conv_param_spec_.pw_activation_type;
    dw_act_param_.mode = conv_param_spec_.dw_activation_type;
    pw_alg_ = CONVOLUTION_ALGORITHM_NULL;
    dw_alg_ = DEPTHWISE_CONVOLUTION_ALGORITHM_NULL;
  }

  ~ConvolutionBoltCPUKernel() override;

  int Prepare() override;
  int Run() override;

  schema::PrimitiveType type() const override { return schema::PrimitiveType_Conv2DFusion; }

 private:
  int InitWeightBiasDesc();
  int InitWeightBiasTensor();
  int InferForwardAlgorithm();
  int InferFilterTransformBytes(int *bytes, int *bytes_extra);
  int TransformFilter();
  int MallocTmpTensor();

  ConvolutionParamSpec conv_param_spec_;
  ActivationParamSpec pw_act_param_;
  ActivationParamSpec dw_act_param_;
  ConvolutionForwardAlgorithm pw_alg_;
  DepthwiseConvolutionForwardAlgorithm dw_alg_;

  std::vector<BoltTensor> weight_tensors_;
  std::vector<BoltTensor> bias_tensors_;
  std::shared_ptr<BoltTensor> weight_tmp_tensor_;

  bool weight_is_packed_ = false;
  void *tile_weight_ = nullptr;  // tile weight for channel 8
  void *tile_bias_ = nullptr;    // tile bias for channel 8
  void *tmp_weight_ = nullptr;
  void *pw_weight_ = nullptr;

  void *run_buffer_ = nullptr;
};
}  // namespace mindspore::kernel::bolt

#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BOLT_CONVOLUTION_BOLT_H_
