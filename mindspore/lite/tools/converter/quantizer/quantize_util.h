/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "ir/anf.h"
#include "src/tensor.h"
#include "include/api/model.h"
#include "include/errorcode.h"
#include "tools/converter/cxx_api/converter_para.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/quantizer/mixed_bit_weight_quantization.h"
#include "tools/common/string_util.h"
#include "ops/core_ops.h"
#include "ops/quant_dtype_cast.h"

namespace mindspore::lite::quant {
int UpdateTensorDataAndSize(const AnfNodePtr &node, const tensor::TensorPtr &weight, void *quant_datas, size_t new_size,
                            TypeId new_data_type);

int GetPreferredDim(const CNodePtr &cnode, int input_index, const std::vector<int> &dims);

std::vector<int> ConvertShapeVectorToInt32(const ShapeVector &dims);

int DeQuantData(const mindspore::MSTensor *tensor, std::vector<double> *dequant_data);

int GetQuantType(const CNodePtr &cnode, quant::QuantType *quant_type);

void GetFuncGraphs(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *all_func_graphs);

int UpdateDataType(const AnfNodePtr &node, TypeId new_data_type);

bool IsGraphInDTypeCast(const CNodePtr &cnode);

bool IsGraphOutDTypeCast(const FuncGraphPtr &func_graph, const CNodePtr &cnode);

int GetCastNodeType(const FuncGraphPtr &func_graph, const CNodePtr &cnode, CastNodeType *cast_node_type);

std::string NodePrimitiveType(const CNodePtr &cnode);

Status BuildModelByFuncGraph(const std::shared_ptr<mindspore::Model> &model, const FuncGraphPtr &func_graph,
                             const std::shared_ptr<mindspore::ConverterPara> &param, size_t *size);

mindspore::lite::Tensor *MSTensorToLiteTensor(const mindspore::MSTensor &tensor);

std::vector<mindspore::lite::Tensor *> MSTensorToLiteTensors(const std::vector<mindspore::MSTensor> &src_tensors);

void GetParameterAndTensor(const AnfNodePtr &node, ParameterPtr *param_node, tensor::TensorPtr *tensor_info);

bool CheckNodeInSet(const CNodePtr &cnode, const std::set<PrimitivePtr> &support_primitive_types);

int GetElementNumFromShape(const std::vector<int> &dims, int *total_size);

int GetBucketAllIndex(const std::vector<int> &dims, int preferred_dim,
                      std::vector<std::vector<size_t>> *buckets_data_index);

bool CheckControlFlowType(const AnfNodePtr &node);

int CloneFuncGraph(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param,
                   FuncGraphPtr *func_graph_bak);

int ConvertFp16ToFp32(const FuncGraphPtr &old_graph);

int ConvertFp32ToFp16(const FuncGraphPtr &old_graph);

int ConvertCNodeFp32ToFp16(const CNodePtr &cnode);

int ConvertCNodeFp16ToFp32(const CNodePtr &cnode);

int MarkOriginDataType(const FuncGraphPtr &func_graph);

int DumpGraph(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param,
              const std::string &save_path);

template <typename T>
int DeQuantData(const int8_t *tensor_data, int64_t elements_num, std::vector<mindspore::QuantParam> quant_params,
                std::vector<T> *dequant_data) {
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
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZE_UTIL_H_
