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

#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include <algorithm>
#include <vector>
#include <utility>
#include <unordered_map>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
static std::unordered_map<DataType, std::pair<OperandCode, OperandCode>> nnapi_data_type = {
  {DataType::kNumberTypeBool, {ANEURALNETWORKS_BOOL, ANEURALNETWORKS_TENSOR_BOOL8}},
  {DataType::kNumberTypeInt8, {ANEURALNETWORKS_BOOL, ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL}},
  {DataType::kNumberTypeUInt8, {ANEURALNETWORKS_BOOL, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}},
  {DataType::kNumberTypeInt32, {ANEURALNETWORKS_INT32, ANEURALNETWORKS_TENSOR_INT32}},
  {DataType::kNumberTypeFloat16, {ANEURALNETWORKS_FLOAT16, ANEURALNETWORKS_TENSOR_FLOAT16}},
  {DataType::kNumberTypeFloat32, {ANEURALNETWORKS_FLOAT32, ANEURALNETWORKS_TENSOR_FLOAT32}}};

void ConverTensorQuantSymmToASymm(MSTensor *ms_tensor) {
  MS_ASSERT(ms_tensor != nullptr);
  MS_CHECK_TRUE_RET_VOID(ms_tensor->DataType() == DataType::kNumberTypeInt8);
  MS_CHECK_TRUE_RET_VOID(ms_tensor->QuantParams().size() == 1);
  ms_tensor->SetDataType(DataType::kNumberTypeUInt8);
  auto quant_param = ms_tensor->QuantParams().front();
  quant_param.zero_point += 128;
  ms_tensor->SetQuantParams({quant_param});
  if (ms_tensor->IsConst()) {
    auto data = ms_tensor->MutableData();
    for (int idx = 0; idx < ms_tensor->ElementNum(); idx++) {
      *(reinterpret_cast<uint8_t *>(data) + idx) = *(reinterpret_cast<int8_t *>(data) + idx) + 128;
    }
  }
}

int AddNNAPIOperand(ANeuralNetworksModel *nnapi_model, MSTensor ms_tensor, int idx, int quant_channel_dim,
                    bool is_scalar) {
  MS_ASSERT(nnapi_model != nullptr);
  ANeuralNetworksOperandType nnapi_tensor;
  auto ms_data_type = ms_tensor.DataType();
  if (nnapi_data_type.find(ms_data_type) == nnapi_data_type.end()) {
    MS_LOG(ERROR) << "Unsupported data type: " << static_cast<int>(ms_data_type);
    return RET_ERROR;
  }
  nnapi_tensor.type = is_scalar ? nnapi_data_type.at(ms_data_type).first : nnapi_data_type.at(ms_data_type).second;
  nnapi_tensor.scale = 0.f;    // These fields are used for quantized tensors
  nnapi_tensor.zeroPoint = 0;  // These fields are used for quantized tensors

  ANeuralNetworksSymmPerChannelQuantParams quant_params;
  std::vector<float> scales;
  if (ms_tensor.QuantParams().size() == 1) {
    nnapi_tensor.scale = ms_tensor.QuantParams().front().scale;
    nnapi_tensor.zeroPoint = ms_tensor.QuantParams().front().zero_point;
  } else {
    quant_params.channelDim = quant_channel_dim;
    quant_params.scaleCount = ms_tensor.QuantParams().size();
    std::transform(ms_tensor.QuantParams().begin(), ms_tensor.QuantParams().end(), std::back_inserter(scales),
                   [](QuantParam param) { return static_cast<float>(param.scale); });
    quant_params.scales = scales.data();
  }

  auto shape = ms_tensor.Shape();
  if (shape.empty() && !is_scalar) {
    shape.push_back(1);
  }
  nnapi_tensor.dimensionCount = static_cast<uint32_t>(shape.size());
  std::vector<uint32_t> dims;
  std::transform(shape.begin(), shape.end(), std::back_inserter(dims),
                 [](int64_t dim) { return static_cast<uint32_t>(dim); });
  nnapi_tensor.dimensions = dims.empty() ? nullptr : dims.data();

  if (nnapi_->ANeuralNetworksModel_addOperand(nnapi_model, &nnapi_tensor) != ANEURALNETWORKS_NO_ERROR) {
    MS_LOG(ERROR) << "Add operand to NNAPI model failed: " << ms_tensor.Name();
    return RET_ERROR;
  }
  if (nnapi_tensor.type == ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL) {
    nnapi_->ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(nnapi_model, idx, &quant_params);
  }
  if (ms_tensor.IsConst() &&
      nnapi_->ANeuralNetworksModel_setOperandValue(nnapi_model, idx, ms_tensor.MutableData(), ms_tensor.DataSize()) !=
        ANEURALNETWORKS_NO_ERROR) {
    MS_LOG(ERROR) << "Set operand value for NNAPI model failed: " << ms_tensor.Name();
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
