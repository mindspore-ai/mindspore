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

#include <vector>
#include <memory>
#include <utility>
#include "tools/converter/converter_packed_node.h"
#include "tools/converter/offline_packing_optimizer.h"
#include "src/litert/kernel/cpu/int8/matmul_dynamic_base_int8.h"
#include "mindspore/core/ops/op_name.h"
#include "src/litert/kernel/cpu/fp32/matmul_fp32.h"

namespace mindspore {
namespace {
constexpr auto kMatmulCustomType = "MatmulFusionPacked";
}

namespace lite {
void AddCustomAttr(std::vector<std::unique_ptr<mindspore::schema::AttributeT>> *attrs, const std::string &&key,
                   const std::string &&value) {
  auto attr = std::make_unique<schema::AttributeT>();
  attr->name = key;
  std::vector<uint8_t> attr_data(value.begin(), value.end());
  attr->data = attr_data;
  attrs->emplace_back(std::move(attr));
}

int AddWeightSumsToInputs(const mindspore::kernel::MatmulDynamicBaseInt8CPUKernel *matmul_kernel,
                          schema::MetaGraphT *meta_graph, const std::unique_ptr<schema::CNodeT> &cnode,
                          size_t weight_sum_size) {
  auto weight_sums_tensor = std::make_unique<schema::TensorT>();
  weight_sums_tensor->nodeType = lite::NodeType_ValueNode;
  weight_sums_tensor->format = schema::Format_NHWC;
  weight_sums_tensor->dataType = TypeId::kNumberTypeInt32;
  weight_sums_tensor->dims = {};
  weight_sums_tensor->dims.emplace_back(weight_sum_size / sizeof(int));
  weight_sums_tensor->data.resize(weight_sum_size);
  weight_sums_tensor->name = cnode->name + "_weight_sums";
  if (memcpy_s(weight_sums_tensor->data.data(), weight_sums_tensor->data.size(), matmul_kernel->GetWeightSums(),
               weight_sum_size) != EOK) {
    MS_LOG(ERROR) << "new CustomT error.";
    return RET_ERROR;
  }
  cnode->inputIndex.emplace_back(meta_graph->allTensors.size());
  meta_graph->allTensors.emplace_back(std::move(weight_sums_tensor));
  return RET_OK;
}

int ReplaceMatMulFusionToCustom(schema::MetaGraphT *meta_graph, const std::unique_ptr<schema::CNodeT> &cnode,
                                const std::unique_ptr<mindspore::schema::TensorT> &b_input,
                                const std::string &cpu_option) {
  auto lite_kernel = PackDataWrapper::GetInstance().GetPackedKernel(cnode->name);
  if (lite_kernel == nullptr) {
    MS_LOG(ERROR) << "Get Packed Kernel error.";
    return RET_ERROR;
  }
  auto param = lite_kernel->op_parameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return RET_ERROR;
  }
  auto matmul_param = reinterpret_cast<MatMulParameter *>(param);
  if (matmul_param->matmul_type_ == kNotImplemented) {
    MS_LOG(ERROR) << "Unsupported matmul type, only support fp32 and dynamic quant int8.";
    return RET_ERROR;
  }
  cnode->primitive->value.type = schema::PrimitiveType_Custom;
  auto primitive = new (std::nothrow) schema::CustomT;
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new CustomT error.";
    return RET_NULL_PTR;
  }
  primitive->type = kMatmulCustomType;

  // activation_type
  AddCustomAttr(&(primitive->attr), ops::kActivationType, std::to_string(matmul_param->act_type_));
  // transpose_a
  AddCustomAttr(&(primitive->attr), ops::kTransposeA, std::to_string(matmul_param->a_transpose_));
  // transpose_b
  AddCustomAttr(&(primitive->attr), ops::kTransposeB, std::to_string(matmul_param->b_transpose_));

  int b_batch;
  const void *pack_b_ptr = nullptr;
  size_t pack_b_size;
  if (matmul_param->matmul_type_ == kMatmulDynamicSdotInt8Cpu) {
    // replace packed data
    auto matmul_kernel = reinterpret_cast<const mindspore::kernel::MatmulDynamicBaseInt8CPUKernel *>(lite_kernel);
    b_batch = matmul_kernel->GetBBatch();
    pack_b_size = b_batch * matmul_param->col_align_ * matmul_param->deep_align_ * sizeof(int8_t);
    pack_b_ptr = reinterpret_cast<const void *>(matmul_kernel->GetPackBPtr());
    auto weight_sum_size = b_batch * matmul_param->col_align_ * sizeof(int);
    int ret = AddWeightSumsToInputs(matmul_kernel, meta_graph, cnode, weight_sum_size);
    if (ret != RET_OK) {
      delete primitive;
      MS_LOG(ERROR) << "add weight sums to inputs error.";
      return ret;
    }
  } else if (matmul_param->matmul_type_ == kMatmulFp32BaseCpu || matmul_param->matmul_type_ == kMatmulFp32Arm64Cpu) {
    auto matmul_kernel = reinterpret_cast<const mindspore::kernel::MatmulCPUKernel *>(lite_kernel);
    auto matmul_kernel_base = matmul_kernel->GetMatmulBase();
    b_batch = matmul_kernel_base->GetBBatch();
    pack_b_size = b_batch * matmul_param->col_align_ * matmul_param->deep_ * sizeof(float);
    pack_b_ptr = reinterpret_cast<const void *>(matmul_kernel_base->GetPackBPtr());
  }

  if (pack_b_ptr == nullptr) {
    delete primitive;
    MS_LOG(ERROR) << "pack_b_ptr is nullptr.";
    return RET_NULL_PTR;
  }

  // copy packed weight to meta graph
  b_input->data.resize(pack_b_size);
  if (memcpy_s(b_input->data.data(), b_input->data.size(), pack_b_ptr, pack_b_size) != EOK) {
    delete primitive;
    MS_LOG(ERROR) << "memcpy packed weight error.";
    return RET_ERROR;
  }

  // add scalar to attr
  AddCustomAttr(&(primitive->attr), "b_batch", std::to_string(b_batch));
  AddCustomAttr(&(primitive->attr), "deep", std::to_string(matmul_param->deep_));
  AddCustomAttr(&(primitive->attr), "col", std::to_string(matmul_param->col_));
  AddCustomAttr(&(primitive->attr), "col_align", std::to_string(matmul_param->col_align_));
  AddCustomAttr(&(primitive->attr), "deep_align", std::to_string(matmul_param->deep_align_));

  // add cpu option
  std::string cpu_option_str = cpu_option;
  AddCustomAttr(&(primitive->attr), "cpu_option", std::move(cpu_option_str));

  cnode->primitive->value.value = primitive;
  return RET_OK;
}

int ConverterPackedNode(schema::MetaGraphT *meta_graph, const std::string &cpu_option) {
  for (auto &dst_node : meta_graph->nodes) {
    if (dst_node->primitive == nullptr || dst_node->primitive->value.type != schema::PrimitiveType_MatMulFusion) {
      continue;
    }
    MS_CHECK_TRUE_MSG(dst_node->inputIndex.size() >= kInputSize1, RET_ERROR, "inputs size is wrong.");
    auto a_index = dst_node->inputIndex[FIRST_INPUT];
    MS_CHECK_TRUE_MSG(meta_graph->allTensors.size() > a_index, RET_ERROR, "allTensors size is wrong.");
    auto &a_input = meta_graph->allTensors.at(a_index);
    CHECK_NULL_RETURN(a_input);

    auto b_index = dst_node->inputIndex[SECOND_INPUT];
    MS_CHECK_TRUE_MSG(meta_graph->allTensors.size() > b_index, RET_ERROR, "allTensors size is wrong.");
    auto &b_input = meta_graph->allTensors.at(b_index);
    CHECK_NULL_RETURN(b_input);

    if (a_input->dataType != b_input->dataType) {
      MS_LOG(ERROR) << "inputs dataType is not same." << a_input->dataType << " " << b_input->dataType;
      return RET_ERROR;
    }

    if (b_input->data.empty()) {
      continue;
    }
    auto ret = ReplaceMatMulFusionToCustom(meta_graph, dst_node, b_input, cpu_option);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ReplaceMatmulToCustom error.";
      return ret;
    }
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
