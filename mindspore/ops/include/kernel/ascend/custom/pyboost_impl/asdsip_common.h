/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MS_EXTENSION_ASCEND_ASDSIP_COMMON_H_
#define MINDSPORE_CCSRC_MS_EXTENSION_ASCEND_ASDSIP_COMMON_H_

#include <string>
#include <memory>
#include <mutex>
#include <utility>
#include <list>
#include <unordered_map>
#include "kernel/ascend/custom/pyboost_impl/asdsip/asdsip_utils.h"

namespace ms::pynative {
/// \brief A runner for executing Ascend SiP Boost (ASDSIP) FFT operations.
///
/// This class extends the `PyboostRunner` class and provides additional functionality
/// for initializing and running ASDSIP FFT operations on Ascend hardware. It manages input
/// and output tensors, memory allocation, and kernel launches specific to ASDSIP.
class OPS_ASCEND_API AsdSipFFTOpRunner : public PyboostRunner {
 public:
  explicit AsdSipFFTOpRunner(std::string op_name) : PyboostRunner(op_name) {}
  void ProcessWithWorkspace() override;
  size_t CalcWorkspace() override;
  void LaunchKernel() override;
  /// \brief [API] Initializes the ASDSIP operation with the given parameters.
  ///
  /// This method retrieves the corresponding ASDSIP operation instance using
  /// the provided parameters and operation name.
  /// The operation instance is used to configure and execute the operation.
  ///
  /// \param[in] param The parameters required to initialize the ASDSIP operation.
  ///
  /// \details
  /// - The `param` is passed to the `FFTCache` singleton, which manages
  ///   caching of ASDSIP operation instances.
  void Init(const FFTParam &param);
  static void SetCacheSize(size_t capaticy);

 protected:
  void _Run() override;

 private:
  asdFftHandle asd_fft_handle_;
  uint64_t workspace_size_{0};
  aclTensor *input_tensor_;
  aclTensor *output_tensor_;
  static bool cache_set_flag_;
  static size_t cache_capaticy_;
};

/// \brief [API] Executes an ASDSIP FFT operation using the provided parameters, inputs, and outputs.
///
/// This function creates an `AsdSipFFTOpRunner` instance, initializes it with the given parameters,
/// and executes the operation on Ascend hardware.
///
/// \param[in] op_name The name of the ASDSIP FFT operation to execute.
/// \param[in] fft_param The parameters required to initialize the ASDSIP FFT operation.
/// \param[in] input An input tensor for the operation.
/// \param[in] output An output tensor for the operation.
///
/// \details
/// - The function first creates a shared pointer to an `AsdSipFFTOpRunner` instance.
/// - It calls the `Init` method of the runner to initialize the operation.
/// - Finally, it invokes the `Run` method of the runner to execute the operation.
inline void RunAsdSipFFTOp(const std::string &op_name, const FFTParam &fft_param, const ms::Tensor &input,
                           const ms::Tensor &output) {
  auto runner = std::make_shared<AsdSipFFTOpRunner>(op_name);
  runner->Init(fft_param);
  runner->Run({input}, {output});
}
}  // namespace ms::pynative
#endif  // MINDSPORE_CCSRC_MS_EXTENSION_ASCEND_ASDSIP_COMMON_H_
