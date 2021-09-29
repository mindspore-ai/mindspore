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

#include "tools/converter/legacy_optimizer/graph/tensor_quant_pass.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include "tools/converter/converter_context.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/common/tensor_util.h"
#include "tools/common/graph_util.h"
#include "tools/common/node_util.h"
#include "src/common/quant_utils.h"
#include "src/common/log_util.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
namespace {
bool TensorNeedQuant(const std::unique_ptr<TensorT> &tensor) {
  if (!quant::TensorQuantParamsInited(*tensor)) {
    return false;
  }
  if (tensor->dataType != TypeId::kNumberTypeFloat32 && tensor->dataType != TypeId::kNumberTypeFloat &&
      tensor->dataType != TypeId::kNumberTypeUInt8 && tensor->dataType != TypeId::kTypeUnknown) {
    return false;
  }
  return !tensor->data.empty();
}

STATUS ComputeDataToInt8(const std::unique_ptr<TensorT> &tensor, int32_t index) {
  MS_ASSERT(tensor != nullptr);
  size_t wShapeSize = tensor->data.empty() ? 0 : GetShapeSize(*(tensor.get()));
  void *oriWeightData = tensor->data.data();
  if (oriWeightData == nullptr) {
    return RET_OK;
  }
  std::vector<int8_t> qDatas(wShapeSize);
  auto weightQauntParam = GetTensorQuantParam(tensor);
  if (tensor->dataType == TypeId::kNumberTypeFloat ||
      tensor->dataType == TypeId::kNumberTypeFloat32) {  // normal awareing quant
    auto *weightData = static_cast<float *>(oriWeightData);
    for (size_t j = 0; j < wShapeSize; j++) {
      qDatas[j] = QuantizeData<int8_t>(weightData[j], weightQauntParam.get());
    }
  } else {  // convert uint8 to int8
    auto *weightData = static_cast<uint8_t *>(oriWeightData);
    for (size_t j = 0; j < wShapeSize; j++) {
      qDatas[j] = (int32_t)weightData[j] - 128;
    }
    weightQauntParam->zeroPoint -= 128;
    tensor->quantParams.clear();
    tensor->quantParams.emplace_back(weightQauntParam.release());
  }
  tensor->dataType = TypeId::kNumberTypeInt8;
  if (tensor->data.empty()) {
    return RET_OK;
  }
  tensor->data.clear();
  MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW_THRESHOLD(wShapeSize, sizeof(int8_t), SIZE_MAX), RET_ERROR, "int mul overflow");
  tensor->data.resize(wShapeSize * sizeof(int8_t));
  if (memcpy_s(tensor->data.data(), tensor->data.size(), qDatas.data(), wShapeSize * sizeof(int8_t)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS ComputeDataToInt32(const std::unique_ptr<TensorT> &tensor) {
  MS_ASSERT(tensor != nullptr);
  auto bShapeSize = GetShapeSize(*(tensor));
  auto qDatas = std::make_unique<int32_t[]>(bShapeSize);
  if (qDatas == nullptr) {
    MS_LOG(ERROR) << "new qDatas failed";
    return RET_ERROR;
  }
  void *biasData = tensor->data.data();
  auto *rawDatas = static_cast<float *>(biasData);
  if (fabs(tensor->quantParams.front()->scale) <= 0.0f) {
    MS_LOG(ERROR) << "divisor 'scale' cannot be 0";
    return RET_ERROR;
  }
  for (size_t i = 0; i < bShapeSize; ++i) {
    qDatas[i] = (int32_t)std::round(rawDatas[i] / tensor->quantParams.front()->scale);
  }
  tensor->dataType = TypeId::kNumberTypeInt32;
  tensor->data.clear();
  MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW_THRESHOLD(bShapeSize, sizeof(int32_t), SIZE_MAX), RET_ERROR, "int mul overflow");
  tensor->data.resize(bShapeSize * sizeof(int32_t));
  if (memcpy_s(tensor->data.data(), tensor->data.size(), qDatas.get(), bShapeSize * sizeof(int32_t)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS ComputeQuantTensorPerChannel(TensorT *tensor, const int &tensor_index, const schema::MetaGraphT &graph) {
  bool channel_at_first = true;
  int channel_cnt = -1;
  auto used_nodes_idx = GetLinkedPostIdx(graph, tensor_index);
  if (used_nodes_idx.size() != 1) {
    MS_LOG(ERROR) << "Tensor is used by nodes more than one";
    return RET_ERROR;
  }
  auto &used_node = graph.nodes.at(used_nodes_idx.front());
  auto &primitive = used_node->primitive;
  int input_index = GetTensorInputIndexInCNode(tensor_index, *used_node);
  quant::CalQuantAssitInfo(*primitive, tensor->dims, input_index, &channel_at_first, &channel_cnt);

  auto *raw_datas = reinterpret_cast<float *>(tensor->data.data());
  ShapeVector dims;
  std::transform(tensor->dims.begin(), tensor->dims.end(), std::back_inserter(dims),
                 [&](int32_t dim) { return (int64_t)dim; });
  auto channels = quant::CalChannels(dims, channel_cnt, &channel_at_first);
  if (channels == 0) {
    MS_LOG(ERROR) << "channels is zero";
    return RET_ERROR;
  }
  int32_t dst_dtype = tensor->quantParams.front()->dstDtype == kNumberTypeInt32 ? kNumberTypeInt32 : kNumberTypeInt8;
  size_t elem_count = tensor->data.size() / sizeof(float);
  MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW_THRESHOLD(elem_count, sizeof(int32_t), SIZE_MAX), RET_ERROR, "int mul overflow");
  size_t data_size = dst_dtype == kNumberTypeInt32 ? elem_count * sizeof(int32_t) : elem_count * sizeof(int8_t);
  std::vector<int8_t> dst_data(data_size);
  MS_CHECK_TRUE_MSG(channels != 0, RET_ERROR, "divide 0");
  size_t one_filter_size = elem_count / channels;
  for (int i = 0; i < channels; i++) {
    if (tensor->quantParams.at(i)->scale <= 0.0f) {
      MS_LOG(ERROR) << "scale:" << tensor->quantParams.at(i)->scale << " <= 0";
      return RET_OK;
    }
    // do quantization
    for (size_t j = 0; j < one_filter_size; j++) {
      auto index = j + i * one_filter_size;
      if (!channel_at_first) {
        index = j * channels + i;
      }
      MS_CHECK_TRUE_MSG(index < elem_count, RET_ERROR, "out of range");
      float raw_data = raw_datas[index];
      if (tensor->quantParams.at(i)->dstDtype == kNumberTypeInt32) {
        auto quant_data = (int32_t)std::round(raw_datas[i] / tensor->quantParams.at(i)->scale);
        auto *dst_data_int32 = reinterpret_cast<int32_t *>(dst_data.data());
        dst_data_int32[index] = quant_data;
      } else {
        auto quant_data = QuantizeData<int8_t>(raw_data, tensor->quantParams.at(i).get());
        dst_data[index] = quant_data;
      }
    }
  }
  tensor->data.clear();
  tensor->data.resize(data_size);
  tensor->dataType = dst_dtype;
  if (memcpy_s(tensor->data.data(), data_size, dst_data.data(), data_size) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace

STATUS TensorQuantPass::Run(schema::MetaGraphT *graph) {
  CHECK_NULL_RETURN(graph);
  int32_t index = 0;
  auto status = RET_OK;
  for (auto &tensor : graph->allTensors) {
    if (!TensorNeedQuant(tensor)) {
      index++;
      continue;
    }

    if (tensor->quantParams.size() > 1) {  // perchannel
      status = ComputeQuantTensorPerChannel(tensor.get(), index, *graph);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "compute tensor to int8 prechannel failed.";
        return RET_ERROR;
      }
      int bit_num = tensor->quantParams.front()->numBits;
      if (DoBitPack(bit_num, tensor.get()) != RET_OK) {
        MS_LOG(ERROR) << "bit pack failed.";
        return RET_ERROR;
      }
      index++;
      continue;
    }
    // perlayer
    auto &quantParam = tensor->quantParams.front();
    if (quantParam->dstDtype == TypeId::kNumberTypeInt8 || quantParam->dstDtype == TypeId::kNumberTypeUInt8 ||
        quantParam->dstDtype == TypeId::kNumberTypeFloat32 || quantParam->dstDtype == TypeId::kNumberTypeFloat) {
      status = ComputeDataToInt8(tensor, index);
      int bit_num = tensor->quantParams.front()->numBits;
      if (DoBitPack(bit_num, tensor.get()) != RET_OK) {
        MS_LOG(ERROR) << "bit pack failed.";
        return RET_ERROR;
      }
    } else if (quantParam->dstDtype == TypeId::kNumberTypeInt32) {
      // quant bias data
      status = ComputeDataToInt32(tensor);
    }
    if (status != RET_OK) {
      MS_LOG(ERROR) << "compute data to int8 or int32 failed.";
      return status;
    }
    index++;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
