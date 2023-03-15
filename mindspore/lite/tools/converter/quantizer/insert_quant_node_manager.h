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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_INSERT_QUANT_NODE_MANAGER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_INSERT_QUANT_NODE_MANAGER_H_
#include <vector>
#include <set>
#include <string>
#include <memory>
#include "include/errorcode.h"
#include "ir/anf.h"
#include "ir/dtype/type_id.h"
#include "ir/func_graph.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "ops/dynamic_quant.h"

namespace mindspore::lite::quant {
class InsertQuantNodeManager {
 public:
  InsertQuantNodeManager() = default;

  ~InsertQuantNodeManager() = default;

  int InsertDynamicQuantNode(const FuncGraphPtr &graph, const std::set<PrimitivePtr> &support_dynamic_quant_ops,
                             const std::set<std::string> &skip_quant_node);

  int InsertDequantNode(const FuncGraphPtr &graph);

  int InsertForwardCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype,
                            quant::QuantType curr_quant_type);

  int InsertCastNodeForFullQuant(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype,
                                 quant::QuantType curr_quant_type);

  int InsertBackwardCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype,
                             quant::QuantType curr_quant_type);

  int InsertQuantDtypeCastFlyNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t input_index,
                                  TypeId src_dtype, TypeId dst_dtype, int axis);

  int InsertFSEDecodeNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t input_index, TypeId dst_dtype);

  int InsertAscendQuantNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode);

  int InsertAscendDeQuantNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode);

  int InsertTransposeNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t index);

  int AdjustTransposeNodeForMatMul(const FuncGraphPtr &func_graph);

 private:
  int InsertAscendQuantNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t input_index);

  int CheckDataType(const AnfNodePtr &input_node, TypeId check_type_id) const;

  int NewDynamicQuantNode(const FuncGraphPtr &graph, const CNodePtr &cnode);

  int MarkDynamicQuantize(const CNodePtr &cnode);

  int InsertDynamicQuantWithIndex(const FuncGraphPtr &graph, const CNodePtr &cnode, size_t index);

  int SetCastNodeAbstract(const CNodePtr &cnode, const AnfNodePtr &input_node, const CNodePtr &cast_cnode);

  int InsertForwardQuantNode(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype, size_t index,
                             CastNodeType cast_node_type);

  int InsertBackwardDeQuantNode(const FuncGraphPtr &graph, const CNodePtr &cnode, TypeId cast_dtype, size_t index,
                                const AnfNodePtr &output_node);

  int InsertQuantDtypeCastNode(const FuncGraphPtr &graph, const CNodePtr &cnode, InsertDirection insert_direction,
                               TypeId cast_dtype, CastNodeType cast_node_type, size_t index,
                               const AnfNodePtr &output_node);

  int CreateFSEInputs(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, std::vector<AnfNodePtr> *op_inputs,
                      TypeId dst_dtype);

  ValueNodePtr NewQuantCastPrimitive(int src_type, int dst_type,
                                     const std::vector<schema::QuantParamT> &input_quant_params,
                                     const std::vector<schema::QuantParamT> &output_quant_params, int axis = 0,
                                     bool set_quant_flag = true);

  ValueNodePtr NewFSEDecodePrimitive(int dst_type, uint64_t curr_chunk, int64_t curr_chunk_index,
                                     int64_t curr_bit_count, int64_t table_log);

  template <typename T>
  void Transpose2Dim(const T *in_data, T *out_data, const int *strides, const int *perm, const int *output_shape) {
    const int stride0 = strides[perm[0]];
    const int stride1 = strides[perm[1]];
    const int output0 = output_shape[0];
    const int output1 = output_shape[1];
    for (int i = 0; i < output0; ++i) {
      int out_stride0_i = i * output1;
      int stride0_i = i * 1 * stride0;
      for (int j = 0; j < output1; ++j) {
        out_data[out_stride0_i + j] = in_data[stride0_i + j * stride1];
      }
    }
  }

  template <typename T>
  int TransposeData(const ParameterPtr &param_node, const tensor::TensorPtr &tensor_info) {
    if (tensor_info->shape_c().size() != 2) {
      MS_LOG(ERROR) << "shape size is " << tensor_info->shape_c().size();
      return RET_ERROR;
    }
    T *out_data = new T[tensor_info->DataSize()];
    CHECK_NULL_RETURN(out_data);

    int strides[2] = {static_cast<int>(tensor_info->shape_c().at(1)), 1};
    int perm[2] = {1, 0};

    ShapeVector transfer_shape = {tensor_info->shape_c().at(1), tensor_info->shape_c().at(0)};
    tensor_info->set_shape(transfer_shape);
    auto abstract = param_node->abstract();
    abstract->set_shape(std::make_shared<abstract::Shape>(transfer_shape));
    T *origin_data = static_cast<T *>(tensor_info->data_c());
    Transpose2Dim(origin_data, out_data, strides, perm, lite::quant::ConvertShapeVectorToInt32(transfer_shape).data());
    auto mem_ret = memcpy_s(tensor_info->data_c(), tensor_info->Size(), out_data, tensor_info->DataSize());
    delete[] out_data;
    if (mem_ret != EOK) {
      MS_LOG(ERROR) << "memcpy failed.";
      return RET_ERROR;
    }
    return RET_OK;
  }

 private:
  TypeId dst_type_ = kNumberTypeInt8;
  bool symmetric_ = false;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_INSERT_QUANT_NODE_MANAGER_H_
