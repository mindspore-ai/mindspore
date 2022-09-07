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

#include "tools/converter/quantizer/quant_helper/graph_inout_transform_pass.h"
#include <set>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "tools/common/node_util.h"
#include "tools/converter/quantizer/insert_quant_node_manager.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/optimizer/common/format_utils.h"
#include "ops/quant_dtype_cast.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::lite::quant {
int GraphInoutTransformPass::Transform() {
  auto ret = DoGraphInputDTypeTransform();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoGraphInputDTypeTransform failed.";
    return ret;
  }
  ret = DoGraphOutputDTypeTransform();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoGraphOutputDTypeTransform failed.";
    return ret;
  }
  return RET_OK;
}

int GraphInoutTransformPass::DoGraphInputDTypeTransform() {
  CHECK_NULL_RETURN(this->func_graph_);
  if (this->graph_input_dtype_ != TypeId::kNumberTypeFloat32 && this->graph_input_dtype_ != TypeId::kNumberTypeUInt8 &&
      this->graph_input_dtype_ != TypeId::kNumberTypeInt8 && this->graph_input_dtype_ != TypeId::kTypeUnknown) {
    MS_LOG(ERROR) << "Invalid graph input dtype: " << this->graph_input_dtype_;
    return RET_ERROR;
  }
  // not specify inputDataType
  if (this->graph_input_dtype_ == TypeId::kTypeUnknown) {
    return RET_OK;
  }
  auto cnodes = this->func_graph_->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    for (size_t index = 1; index < cnode->size(); index++) {
      auto input_node = cnode->input(index);
      CHECK_NULL_RETURN(input_node);
      if (!IsGraphInput(input_node)) {
        continue;
      }
      TypeId input_node_dtype = TypeId::kTypeUnknown;
      if (CheckControlFlowType(input_node)) {
        continue;
      }
      if (opt::GetDataTypeFromAnfNode(input_node, &input_node_dtype) != RET_OK) {
        MS_LOG(ERROR) << "GetDataTypeFromAnfNode failed, input node name: " << input_node->fullname_with_scope();
        return RET_ERROR;
      }
      // graph input dtype transform
      if (this->graph_input_dtype_ != input_node_dtype) {
        auto ret = InsertDTypeCastNode(cnode, index, this->graph_input_dtype_, input_node_dtype, kQuant);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "InsertDTypeCastNode failed, cnode name: " << cnode->fullname_with_scope()
                        << " input index: " << index;
          return ret;
        }
      }
    }  // for
  }    // for
  return RET_OK;
}

int GraphInoutTransformPass::DoGraphOutputDTypeTransform() {
  CHECK_NULL_RETURN(this->func_graph_);
  if (this->graph_output_dtype_ != TypeId::kNumberTypeFloat32 &&
      this->graph_output_dtype_ != TypeId::kNumberTypeUInt8 && this->graph_output_dtype_ != TypeId::kNumberTypeInt8 &&
      this->graph_output_dtype_ != TypeId::kTypeUnknown) {
    MS_LOG(ERROR) << "Invalid graph output dtype: " << this->graph_input_dtype_;
    return RET_ERROR;
  }
  if (this->graph_output_dtype_ == TypeId::kTypeUnknown) {
    return RET_OK;
  }
  auto cnodes = this->func_graph_->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (!IsGraphOutput(cnode)) {
      continue;
    }
    for (size_t index = 1; index < cnode->size(); index++) {
      auto input_node = cnode->input(index);
      CHECK_NULL_RETURN(input_node);
      auto input_cnode = std::dynamic_pointer_cast<mindspore::CNode>(input_node);
      if (opt::CheckPrimitiveType(input_cnode, prim::kPrimMakeTuple)) {  // MakeTuple
        for (size_t input_index = 1; input_index < input_cnode->size(); input_index++) {
          auto ret = DoOutputTransform(input_cnode, input_index);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "DoOutputTransform failed, cnode name: " << input_node->fullname_with_scope()
                          << " input index: " << input_index;
            return ret;
          }
        }
      } else {  // Return
        auto ret = DoOutputTransform(cnode, index);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "DoOutputTransform failed, cnode name: " << cnode->fullname_with_scope()
                        << " input index: " << index;
          return ret;
        }
      }
    }  // for
  }    // for
  return RET_OK;
}

int GraphInoutTransformPass::DoOutputTransform(const CNodePtr &cnode, size_t index) {
  CHECK_NULL_RETURN(cnode);
  auto input_node = cnode->input(index);
  CHECK_NULL_RETURN(input_node);
  if (!input_node->isa<mindspore::CNode>()) {
    return RET_OK;
  }
  TypeId input_node_dtype = TypeId::kTypeUnknown;
  if (opt::GetDataTypeFromAnfNode(input_node, &input_node_dtype) != RET_OK) {
    MS_LOG(ERROR) << "GetDataTypeFromAnfNode failed, input node name: " << input_node->fullname_with_scope();
    return RET_ERROR;
  }
  // graph output dtype transform
  if (this->graph_output_dtype_ != input_node_dtype) {
    auto ret = InsertDTypeCastNode(cnode, index, input_node_dtype, this->graph_output_dtype_, kDeQuant);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "InsertDTypeCastNode failed, cnode name: " << cnode->fullname_with_scope()
                    << " input index: " << index;
      return ret;
    }
  }
  return RET_OK;
}

bool GraphInoutTransformPass::IsGraphOutput(const CNodePtr &cnode) {
  return (opt::CheckPrimitiveType(cnode, prim::kPrimReturn));
}

int GraphInoutTransformPass::InsertDTypeCastNode(const CNodePtr &cnode, size_t input_index, TypeId src_dtype,
                                                 TypeId dst_dtype, CastNodeType node_type) {
  std::vector<schema::QuantParamT> input_quant_params;
  std::vector<schema::QuantParamT> output_quant_params;
  CHECK_NULL_RETURN(cnode);
  auto input_node = cnode->input(input_index);
  CHECK_NULL_RETURN(input_node);
  if (node_type == kQuant) {
    auto curr_primitive_quant_param_holder = GetCNodeQuantHolder(cnode);
    CHECK_NULL_RETURN(curr_primitive_quant_param_holder);
    if (curr_primitive_quant_param_holder->get_input_quant_params().size() < input_index) {
      MS_LOG(ERROR) << "Invalid quant params.";
      return RET_ERROR;
    }
    input_quant_params = curr_primitive_quant_param_holder->get_input_quant_params()[input_index - 1];
  } else if (node_type == kDeQuant) {
    auto input_cnode = std::dynamic_pointer_cast<mindspore::CNode>(input_node);
    auto input_primitive_quant_param_holder = GetCNodeQuantHolder(input_cnode);
    CHECK_NULL_RETURN(input_primitive_quant_param_holder);
    if (input_primitive_quant_param_holder->get_output_quant_params().empty()) {
      MS_LOG(ERROR) << "Invalid quant params.";
      return RET_ERROR;
    }
    input_quant_params = input_primitive_quant_param_holder->get_output_quant_params()[0];
  }
  std::copy(input_quant_params.cbegin(), input_quant_params.cend(), std::back_inserter(output_quant_params));
  // update zeroPoint(uint8toint8, int8touint8)
  if (src_dtype == TypeId::kNumberTypeUInt8 && dst_dtype == TypeId::kNumberTypeInt8) {
    output_quant_params.front().zeroPoint -= kU8ZeroPointOffset;
  } else if (src_dtype == TypeId::kNumberTypeInt8 && dst_dtype == TypeId::kNumberTypeUInt8) {
    input_quant_params.front().zeroPoint -= kU8ZeroPointOffset;
  }

  auto primitive = quant::NewQuantCastPrimitive(src_dtype, dst_dtype, input_quant_params, output_quant_params);
  std::vector<AnfNodePtr> op_inputs = {primitive, input_node};
  auto quant_cast_cnode = this->func_graph_->NewCNode(op_inputs);
  CHECK_NULL_RETURN(quant_cast_cnode);
  quant_cast_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_dtype_cast_" +
                                            std::to_string(input_index));

  // set quant_cast_cnode quant type
  AbstractBasePtr abstract;
  if (cnode->abstract() != nullptr) {
    abstract = cnode->abstract()->Clone();
  } else if (input_node->abstract() != nullptr) {
    abstract = input_node->abstract()->Clone();
  } else {
    MS_LOG(ERROR) << "Abstract is nullptr, cnode name: " << cnode->fullname_with_scope()
                  << " input node: " << input_node->fullname_with_scope();
    return RET_NULL_PTR;
  }
  quant_cast_cnode->set_abstract(abstract);
  auto quant_cast_cnode_param_holder = GetCNodeQuantHolder(quant_cast_cnode);
  CHECK_NULL_RETURN(quant_cast_cnode_param_holder);
  quant_cast_cnode_param_holder->set_quant_type(schema::QuantType_QUANT_NONE);

  // update dtype: input_cnode, quant_cast_cnode
  auto ret = quant::UpdateDataType(input_node, src_dtype);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdateDataType failed, input node name: " << input_node->fullname_with_scope();
    return RET_ERROR;
  }
  cnode->set_input(input_index, quant_cast_cnode);

  ret = quant::UpdateDataType(std::dynamic_pointer_cast<mindspore::AnfNode>(quant_cast_cnode), dst_dtype);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdateDataType failed, cnode name: " << quant_cast_cnode->fullname_with_scope();
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
