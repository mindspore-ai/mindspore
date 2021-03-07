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

#include "coder/opcoders/nnacl/dequant/de_quant.h"
#include <string>
#include <vector>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

static constexpr int kPerTensor = 1;
static constexpr size_t kPerBatch = 3;
namespace mindspore::lite::micro::nnacl {

void Dequant::set_de_quant_buffer_str(const std::string &dequant_buffer_str) {
  de_quant_buffer_str_ = "(float *)(" + dequant_buffer_str + ")";
}

void Dequant::DequantRecordWorkspcae(size_t curr_workspace) {
  de_quant_max_workspace_ = de_quant_max_workspace_ > curr_workspace ? de_quant_max_workspace_ : curr_workspace;
}

bool Dequant::CheckDequantFlag(const Tensor *weight_tensor) {
  if (weight_tensor == nullptr) {
    return false;
  }
  return !weight_tensor->quant_params().empty() && weight_tensor->quant_params().front().inited &&
         weight_tensor->data_c() != nullptr;
}

void Dequant::DeQuantFunctionPerChannel(const Tensor *quant_tensor, const std::vector<DeQuantArg> &de_quant_args,
                                        const std::string &de_quant_arg_base_str,
                                        NNaclFp32Serializer *const de_quant_code) {
  int quant_arg_dims = static_cast<int>(quant_tensor->quant_params().size());
  int de_quant_nums = quant_tensor->ElementsNum();
  for (int i = 0; i < quant_arg_dims; ++i) {
    auto de_quant_arg = de_quant_args.at(i);
    std::string de_quant_arg_str = de_quant_arg_base_str + std::to_string(i);
    de_quant_code->CodeStruct(de_quant_arg_str, de_quant_arg);
  }
  std::string de_quant_args_name = "de_quant_args";
  *de_quant_code << "const DeQuantArg *" << de_quant_args_name << "[" << quant_arg_dims << "] = {\n";
  for (int i = 0; i < quant_arg_dims - 1; ++i) {
    *de_quant_code << "&" << de_quant_arg_base_str << std::to_string(i) << ", ";
  }
  *de_quant_code << "&" << de_quant_arg_base_str << std::to_string(quant_arg_dims - 1);
  *de_quant_code << "};\n";
  size_t per_batch_size = quant_tensor->shape().at(0);
  std::string quant_tensor_addr_str = "(int8_t *)(" + quant_tensor_addr_ + ")";
  de_quant_code->CodeFunction("DequantDataPerChannel", quant_tensor_addr_str, de_quant_args_name, de_quant_nums,
                              per_batch_size, de_quant_buffer_str_);
}

void Dequant::DeQuantFunction(const Tensor *quant_tensor, const std::vector<DeQuantArg> &de_quant_args,
                              const std::string &de_quant_arg_base_str, NNaclFp32Serializer *const de_quant_code) {
  int quant_arg_dims = static_cast<int>(quant_tensor->quant_params().size());
  int de_quant_nums = quant_tensor->ElementsNum();
  for (int i = 0; i < quant_arg_dims; ++i) {
    auto de_quant_arg = de_quant_args.at(i);
    std::string de_quant_arg_str = de_quant_arg_base_str + std::to_string(i);
    de_quant_code->CodeStruct(de_quant_arg_str, de_quant_arg);
  }
  std::string de_quant_args_name = "de_quant_args";
  *de_quant_code << "const DeQuantArg *" << de_quant_args_name << "[" << quant_arg_dims << "] = {\n";
  for (int i = 0; i < quant_arg_dims - 1; ++i) {
    *de_quant_code << "&" << de_quant_arg_base_str << std::to_string(i) << ", ";
  }
  *de_quant_code << "&" << de_quant_arg_base_str << std::to_string(quant_arg_dims - 1);
  *de_quant_code << "};\n";
  auto channels = static_cast<size_t>(quant_tensor->Batch());
  std::string quant_tensor_addr_str = "(int8_t *)(" + quant_tensor_addr_ + ")";
  de_quant_code->CodeFunction("DequantData", quant_tensor_addr_str, de_quant_args_name, de_quant_nums, channels,
                              de_quant_buffer_str_);
}

void Dequant::DeQuantFunctionPerTensor(const Tensor *quant_tensor, const std::vector<DeQuantArg> &de_quant_args,
                                       const std::string &de_quant_arg_base_str,
                                       NNaclFp32Serializer *const de_quant_code) {
  size_t de_quant_nums = quant_tensor->ElementsNum();
  auto de_quant_arg = de_quant_args.at(0);
  std::string de_quant_arg_str = de_quant_arg_base_str + std::to_string(0);
  de_quant_code->CodeStruct(de_quant_arg_str, de_quant_arg);
  std::string de_quant_args_name = "de_quant_args";
  *de_quant_code << "const DeQuantArg *" << de_quant_args_name << "[" << 1 << "] = {\n";
  *de_quant_code << "&" << de_quant_arg_base_str << std::to_string(0);
  *de_quant_code << "};\n";
  std::string quant_tensor_addr_str = "(int8_t *)(" + quant_tensor_addr_ + ")";
  de_quant_code->CodeFunction("DequantDataPerTensor", quant_tensor_addr_str, de_quant_args_name, de_quant_nums,
                              de_quant_buffer_str_);
}

std::string Dequant::GetMicroDeQuantFunction(const Tensor *quant_tensor, const std::string &quant_tensor_addr) {
  std::string de_quant_block;
  if (quant_tensor == nullptr || de_quant_buffer_str_.empty()) {
    return de_quant_block;
  }
  quant_tensor_addr_ = quant_tensor_addr;
  size_t de_quant_nums = quant_tensor->ElementsNum();
  size_t quant_arg_dims = quant_tensor->quant_params().size();
  DequantRecordWorkspcae(static_cast<size_t>(de_quant_nums * sizeof(float)));
  NNaclFp32Serializer de_quant_code;
  de_quant_code << "{\n";
  size_t quant_tensor_dims = quant_tensor->shape().size();
  std::vector<DeQuantArg> de_quant_args;
  std::string de_quant_arg_base_str = "de_quant_arg_";
  for (size_t i = 0; i < quant_arg_dims; ++i) {
    auto curr_quant_param = quant_tensor->quant_params().at(i);
    DeQuantArg de_quant_arg = {
      .scale = static_cast<float>(curr_quant_param.scale),
      .zeroPoint = curr_quant_param.zeroPoint,
      .var_corr = curr_quant_param.var_corr,
      .mean_corr = curr_quant_param.mean_corr,
      // this clusters is meaningless which will be supported in future
      .clusters = {},
      .clusters_nums = static_cast<int>(curr_quant_param.clusters.size()),
      .bitNum = quant_tensor->quant_params().at(i).bitNum,
    };
    de_quant_args.emplace_back(de_quant_arg);
  }
  de_quant_code.CodeFunction("memset", de_quant_buffer_str_, 0, de_quant_nums * sizeof(float));
  if (quant_tensor_dims == kPerBatch && quant_arg_dims == static_cast<size_t>(quant_tensor->shape().at(0))) {
    DeQuantFunctionPerChannel(quant_tensor, de_quant_args, de_quant_arg_base_str, &de_quant_code);
  } else if (quant_arg_dims != kPerTensor) {
    DeQuantFunction(quant_tensor, de_quant_args, de_quant_arg_base_str, &de_quant_code);
  } else {
    DeQuantFunctionPerTensor(quant_tensor, de_quant_args, de_quant_arg_base_str, &de_quant_code);
  }
  de_quant_code << "}\n";
  de_quant_block = de_quant_code.str();
  return de_quant_block;
}
}  // namespace mindspore::lite::micro::nnacl
