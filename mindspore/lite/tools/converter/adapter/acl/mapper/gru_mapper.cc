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

#define USE_DEPRECATED_API
#include "tools/converter/adapter/acl/mapper/gru_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "nnacl/op_base.h"
#include "ops/op_name.h"
#include "src/common/log_util.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace {
constexpr size_t kNumOnnxInputSize = 3;
constexpr size_t kKeyXIndex = 1;
constexpr size_t kKeyXShapeLength = 3;
constexpr size_t kKeyBIndex = 4;
constexpr size_t kKeyBRows = 2;
constexpr size_t kKeyBCols = 192;
constexpr size_t kKeyBShapeLength = 2;
constexpr size_t kKeySequence_lensIndex = 5;
constexpr size_t kKeySequenceShapeLength = 1;
constexpr size_t kKeyInitial_hIndex = 6;
}  // namespace
namespace lite {
std::vector<std::vector<float>> create2DVector(int32_t rows, int32_t cols, float initValue) {
  std::vector<std::vector<float>> vec(rows, std::vector<float>(cols, initValue));
  return vec;
}

std::vector<int32_t> createiniVector(int32_t length, int32_t initValue) {
  std::vector<int32_t> vec(length, initValue);
  return vec;
}

STATUS MapperGruInputs(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "CNodePtr is nullptr";
    return lite::RET_ERROR;
  }

  // Optional input parameters have been set.
  if (cnode->inputs().size() > kKeyInitial_hIndex) {
    return lite::RET_OK;
  }

  auto func_graph = cnode->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is nullptr";
    return lite::RET_ERROR;
  }

  // Add the first three mandatory parameters.
  std::vector<AnfNodePtr> new_inputs;
  new_inputs.insert(new_inputs.end(), cnode->inputs().begin(), cnode->inputs().end());

  auto manager = Manage(func_graph);
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "manager is nullptr.");
  bool is_add_input = false;

  // Check the input parameter kKeyBIndex. If the parameter is not input, construct one.
  if ((cnode->inputs().size() > kKeyBIndex) && (cnode->inputs().size() <= kKeyInitial_hIndex)) {
    ShapeVector input_shape;
    auto abstract = opt::GetCNodeInputAbstract(cnode, kKeyBIndex);
    if (opt::FetchShapeFromAbstract(abstract, &input_shape) != RET_OK) {
      MS_LOG(ERROR) << "FetchShapeFromAbstract failed.";
      return lite::RET_ERROR;
    }
    if (input_shape.size() != kKeyBShapeLength) {
      MS_LOG(DEBUG) << "GRU not input B";
      std::vector<std::vector<float>> data = create2DVector(kKeyBRows, kKeyBCols, 0);
      MS_LOG(DEBUG) << "construct GRU kKeyB data" << data;
      auto value_param = opt::BuildFloatVec2DParameterNode(func_graph, data, cnode->fullname_with_scope() + "B_value");
      new_inputs.insert(new_inputs.begin() + kKeyBIndex, value_param);
      manager->AddEdge(cnode, value_param);
      is_add_input = true;
    }
  }

  // Check the input parameter kKeySequence_lensIndex. If the parameter is not input, construct one.
  if ((cnode->inputs().size() > kKeySequence_lensIndex) && (cnode->inputs().size() <= kKeyInitial_hIndex)) {
    ShapeVector input_shape;
    auto abstract = opt::GetCNodeInputAbstract(cnode, kKeySequence_lensIndex);
    if (opt::FetchShapeFromAbstract(abstract, &input_shape) != RET_OK) {
      MS_LOG(ERROR) << "FetchShapeFromAbstract failed.";
      return lite::RET_ERROR;
    }
    if (input_shape.size() != kKeySequenceShapeLength) {
      MS_LOG(DEBUG) << "GRU not input Sequence_lens";
      // get x shape
      ShapeVector input_shape_X;
      auto abstractX = opt::GetCNodeInputAbstract(cnode, kKeyXIndex);
      if (opt::FetchShapeFromAbstract(abstractX, &input_shape_X) != RET_OK) {
        MS_LOG(ERROR) << "FetchShapeFromAbstract failed.";
        return lite::RET_ERROR;
      }
      if (input_shape_X.size() == kKeyXShapeLength) {
        auto data = createiniVector(input_shape_X.at(1), input_shape_X.at(0));
        MS_LOG(DEBUG) << "construct GRU kKeySequence_lens data" << data;
        auto value_param =
          opt::BuildIntVecParameterNode(func_graph, data, cnode->fullname_with_scope() + "Sequence_lens");
        new_inputs.insert(new_inputs.begin() + kKeySequence_lensIndex, value_param);
        manager->AddEdge(cnode, value_param);
        is_add_input = true;
      }
    }
  }

  if (is_add_input) {
    for (size_t i = 1; i < new_inputs.size(); i++) {
      manager->SetEdge(cnode, i, new_inputs[i]);
    }
  }
  return lite::RET_OK;
}

STATUS SetGruAttr(PrimitivePtr src_prim, PrimitivePtr dst_prim) {
  if (src_prim == nullptr || dst_prim == nullptr) {
    MS_LOG(ERROR) << "SetGruAttr is nullptr.";
    return lite::RET_ERROR;
  }

  auto bidirectional_attr = src_prim->GetAttr("bidirectional");
  if (bidirectional_attr == nullptr) {
    MS_LOG(ERROR) << "bidirectional_attr is nullptr.";
    return lite::RET_ERROR;
  }
  bool bidirectional = GetValue<bool>(bidirectional_attr);
  if (!bidirectional) {
    MS_LOG(ERROR) << "not support bidirectional is false.";
    return lite::RET_ERROR;
  }
  dst_prim->SetAttrs({{"direction", MakeValue("bidirectional")}});

  auto activation_alpha = src_prim->GetAttr("activation_alpha");
  if (activation_alpha != nullptr) {
    dst_prim->SetAttrs({{"activation_alpha", MakeValue(activation_alpha)}});
  }

  auto activation_beta = src_prim->GetAttr("activation_beta");
  if (activation_beta != nullptr) {
    dst_prim->SetAttrs({{"activation_beta", MakeValue(activation_beta)}});
  }

  auto activations = src_prim->GetAttr("activations");
  if (activations != nullptr) {
    dst_prim->SetAttrs({{"activations", MakeValue(activations)}});
  }

  auto clip = src_prim->GetAttr("clip");
  if (clip != nullptr) {
    dst_prim->SetAttrs({{"clip", MakeValue(clip)}});
  }

  auto hidden_size = src_prim->GetAttr("hidden_size");
  if (hidden_size != nullptr) {
    dst_prim->SetAttrs({{"hidden_size", MakeValue(hidden_size)}});
  }

  auto linear_before_reset = src_prim->GetAttr("linear_before_reset");
  if (linear_before_reset != nullptr) {
    dst_prim->SetAttrs({{"linear_before_reset", MakeValue(linear_before_reset)}});
  }
  return lite::RET_OK;
}

STATUS GRUMapper::Mapper(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "CNodePtr is nullptr";
    return lite::RET_ERROR;
  }

  // get src prim
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  if (value_node == nullptr || src_prim == nullptr) {
    MS_LOG(ERROR) << "value node or src prim is nullptr.";
    return lite::RET_ERROR;
  }

  // make dst prim
  auto dst_prim = std::make_shared<acl::CommonGRU>();
  if (dst_prim == nullptr) {
    MS_LOG(ERROR) << "make dst prim failed.";
    return lite::RET_ERROR;
  }
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);

  // get fmk type
  auto fmk_attr = src_prim->GetAttr(ops::kFmkType);
  if (fmk_attr == nullptr) {
    MS_LOG(ERROR) << "attr val is nullptr.";
    return lite::RET_ERROR;
  }
  int fmk_type = GetValue<int64_t>(fmk_attr);

  // fmk type onnx proc
  if (fmk_type == converter::kFmkTypeOnnx) {
    if (cnode->inputs().size() < kNumOnnxInputSize + 1) {
      MS_LOG(ERROR) << "onnx gru op input size is: " << cnode->inputs().size() << ", but export size is: > "
                    << kNumOnnxInputSize + 1;
      return lite::RET_ERROR;
    }

    int ret = RET_OK;
    ret = SetGruAttr(src_prim, dst_prim);
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "set GRU Attr fail";
      return ret;
    }

    ret = MapperGruInputs(cnode);
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "Mapper GRU Inputs fail";
      return ret;
    }
  } else {
    MS_LOG(ERROR) << "not support in gru mapper.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameGRU, GRUMapper)
}  // namespace lite
}  // namespace mindspore
