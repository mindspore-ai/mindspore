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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZE_UTIL_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZE_UTIL_H_

#ifndef _MSC_VER
#include <dirent.h>
#endif

#include <sys/stat.h>
#include <memory>
#include <string>
#include <cmath>
#include <set>
#include <array>
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>
#include <map>
#include <functional>
#include "ops/mat_mul.h"
#include "ops/lstm.h"
#include "ops/fusion/full_connection.h"
#include "tools/converter/quantizer/quantizer.h"
#include "include/errorcode.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/model.h"
#include "base/base.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "tools/converter/quantizer/huffman_encode.h"
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/quantizer/mixed_bit_weight_quantization.h"
#include "src/litert/lite_session.h"
#include "src/common/file_utils.h"
#include "src/common/quant_utils.h"
#include "include/api/model.h"
#include "tools/converter/cxx_api/converter_para.h"
#include "tools/common/string_util.h"

namespace mindspore::lite::quant {
QuantParamHolderPtr GetCNodeQuantHolder(const PrimitivePtr &primitive);

QuantParamHolderPtr GetCNodeQuantHolder(const CNodePtr &cnode);

int UpdateTensorDataAndSize(const AnfNodePtr &node, const tensor::TensorPtr &weight, void *quant_datas, int new_size,
                            TypeId new_data_type);

void CalQuantAssitInfo(const schema::PrimitiveT &primitive, const std::vector<int> &shapes, int index,
                       bool *channel_at_first, int *channel_cnt);

bool TensorQuantParamsInited(const schema::TensorT &tensor);

int CalChannels(const std::vector<int> &dims, int channel_cnt, bool *channel_at_first);

int GetPreferredDim(const CNodePtr &cnode, int input_index, const std::vector<int> &dims);

std::vector<int> ConvertShapeVectorToInt32(const ShapeVector &dims);

int DoParameterBiasQuant(const ParameterPtr &bias, const PrimitivePtr &primitive);

int DeQuantData(const mindspore::MSTensor *tensor, std::vector<double> *dequant_data, int preferred_dim = 0);

int GetQuantType(const CNodePtr &cnode);

void GetFuncGraphs(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *all_func_graphs);

template <typename T>
int DeQuantData(const int8_t *tensor_data, int64_t elements_num, std::vector<mindspore::QuantParam> quant_params,
                std::vector<T> *dequant_data, int preferred_dim = 0) {
  if (quant_params.size() != 1) {
    MS_LOG(ERROR) << "unexpected quant_params size: " << quant_params.size() << " only support per-layer now.";
    return RET_ERROR;
  }
  auto scale = quant_params[0].scale;
  auto zp = quant_params[0].zero_point;
  dequant_data->resize(elements_num);
  for (int64_t i = 0; i < elements_num; i++) {
    dequant_data->at(i) = scale * (tensor_data[i] - zp);
  }
  return RET_OK;
}

template <typename T>
int FixedBitQuantFilter(const AnfNodePtr &parameter_node, const tensor::TensorPtr &weight,
                        const PrimitivePtr &primitive, schema::QuantType quant_type, int quant_max, int quant_min,
                        size_t bit_num, WeightQuantType weight_quant_type, TypeId quant_data_type, int index,
                        int preferred_dim, bool symmetric = false, bool narrow_range = false) {
  MS_ASSERT(weight != nullptr);
  MS_ASSERT(primitive != nullptr);
  auto dims = weight->shape();
  if (weight_quant_type == FIXED_BIT_PER_CHANNEL) {
    if (dims.size() <= 1) {
      MS_LOG(WARNING) << "dims is " << dims.size() << " can not per_channel";
      weight_quant_type = FIXED_BIT_PER_LAYER;
    }
  }

  std::vector<schema::QuantParamT> quant_params;
  size_t elem_count = weight->DataSize();
  auto *raw_data = static_cast<float *>(weight->data_c());
  if (raw_data == nullptr) {
    MS_LOG(ERROR) << "rawDatas is nullptr";
    return RET_ERROR;
  }

  std::vector<T> quant_data(elem_count);
  int ret = RET_OK;
  if (weight_quant_type == FIXED_BIT_PER_CHANNEL) {
    ret = DoPerChannelQuant<T>(static_cast<float *>(weight->data_c()), weight->DataSize(),
                               static_cast<mindspore::schema::QuantType>(quant_type), &quant_params, quant_max,
                               quant_min, bit_num, &quant_data, ConvertShapeVectorToInt32(dims), preferred_dim,
                               symmetric, narrow_range);
    if (ret == RET_NO_CHANGE) {
      return ret;
    } else if (ret != RET_OK) {
      MS_LOG(ERROR) << "Do per channel quant failed.";
      return ret;
    }
  } else if (weight_quant_type == FIXED_BIT_PER_LAYER) {
    ret = DoPerLayerQuant<T>(static_cast<float *>(weight->data_c()), weight->DataSize(), &quant_params, quant_max,
                             quant_min, bit_num, &quant_data, symmetric, narrow_range);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Do per layer quant failed.";
      return ret;
    }
  } else {
    MS_LOG(ERROR) << "Unsupported weight quant type:" << weight_quant_type;
    return RET_ERROR;
  }
  auto status =
    UpdateTensorDataAndSize(parameter_node, weight, quant_data.data(), quant_data.size() * sizeof(T), quant_data_type);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UpdateTensorDataAndSize error";
    return RET_ERROR;
  }
  if (quant_params.empty()) {
    MS_LOG(ERROR) << "quant_params empty";
    return RET_ERROR;
  }
  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  quant_param_holder->set_input_quant_param(index, quant_params);
  quant_param_holder->set_quant_type(quant_type);
  return ret;
}

std::string NodePrimitiveType(const CNodePtr &cnode);

Status BuildModelByFuncGraph(const std::shared_ptr<mindspore::Model> &model, const FuncGraphPtr &func_graph,
                             const std::shared_ptr<mindspore::ConverterPara> &param, size_t *size);

mindspore::lite::Tensor *MSTensorToLiteTensor(const mindspore::MSTensor &tensor);

std::vector<mindspore::lite::Tensor *> MSTensorToLiteTensors(const std::vector<mindspore::MSTensor> &src_tensors);

void GetLiteParameter(const AnfNodePtr &node, ParameterPtr *param_node, tensor::TensorPtr *tensor_info);

bool CheckNodeInSet(const CNodePtr &cnode, const std::set<PrimitivePtr> &support_primitive_types);

int GetElementNumFromShape(const std::vector<int> &dims, int *total_size);

int GetBucketAllIndex(const std::vector<int> &dims, int preferred_dim,
                      std::vector<std::vector<int>> *buckets_data_index);
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZE_UTIL_H_
