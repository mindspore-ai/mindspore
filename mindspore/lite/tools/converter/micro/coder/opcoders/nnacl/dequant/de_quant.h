/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MICRO_LITE_MICRO_CODER_OPCODERS_NNACL_DEQUANT_DEQUANT_H_
#define MICRO_LITE_MICRO_CODER_OPCODERS_NNACL_DEQUANT_DEQUANT_H_

#include <string>
#include <vector>
#include "src/tensor.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
namespace mindspore::lite::micro::nnacl {
class Dequant {
 public:
  Dequant(const Dequant &) = delete;
  Dequant &operator=(const Dequant &) = delete;
  static Dequant *GetInstance() {
    static Dequant dequant;
    return &dequant;
  }

  void set_de_quant_buffer_str(const std::string &de_quant_buffer_str);

  const size_t de_quant_max_workspace() const { return de_quant_max_workspace_; }

  const std::string de_quant_buffer_str() const { return de_quant_buffer_str_; }

  bool CheckDequantFlag(const Tensor *quant_tensor);

  std::string GetMicroDeQuantFunction(const Tensor *quant_tensor, const std::string &quant_tensor_addr);

 private:
  void DeQuantFunctionPerTensor(const Tensor *quant_tensor, const std::vector<DeQuantArg> &de_quant_args,
                                const std::string &de_quant_arg_base_str, NNaclFp32Serializer *de_quant_code);

  void DeQuantFunction(const Tensor *quant_tensor, const std::vector<DeQuantArg> &de_quant_args,
                       const std::string &de_quant_arg_base_str, NNaclFp32Serializer *de_quant_code);

  void DeQuantFunctionPerChannel(const Tensor *quant_tensor, const std::vector<DeQuantArg> &de_quant_args,
                                 const std::string &de_quant_arg_base_str, NNaclFp32Serializer *de_quant_code);

  Dequant() = default;
  ~Dequant() = default;
  void DequantRecordWorkspcae(size_t curr_workspace);

  std::string de_quant_buffer_str_;
  std::string quant_tensor_addr_;
  size_t de_quant_max_workspace_{0};
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MICRO_LITE_MICRO_CODER_OPCODERS_NNACL_DEQUANT_DEQUANT_H_
