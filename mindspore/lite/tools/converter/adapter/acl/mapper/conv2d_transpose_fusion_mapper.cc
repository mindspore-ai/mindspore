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

#include "tools/converter/adapter/acl/mapper/conv2d_transpose_fusion_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "include/registry/converter_context.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kNameOutputPaddingNum = 2;
constexpr auto kNameFormat = "data_format";
}  // namespace
STATUS Conv2dTransposeMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }

  auto attr_val = src_prim->GetAttr(ops::kFmkType);
  int fmk_type = attr_val != nullptr ? GetValue<int>(attr_val) : converter::kFmkTypeTf;
  PrimitivePtr dst_prim = nullptr;
  if (fmk_type == converter::kFmkTypeCaffe) {
    dst_prim = std::make_shared<acl::Deconvolution>();
  } else {
    dst_prim = std::make_shared<acl::Conv2DTransposeV2>();
  }
  MS_CHECK_TRUE_MSG(dst_prim != nullptr, RET_ERROR, "dst_prim is nullptr.");
  dst_prim->SetAttrs(src_prim->attrs());
  if (fmk_type != converter::kFmkTypeCaffe) {
    if (AdjustGeAttr(cnode, dst_prim) != RET_OK) {
      MS_LOG(ERROR) << "Adjust ge attr failed.";
      return RET_ERROR;
    }

    // Construction input input_size
    auto func_graph = cnode->func_graph();
    CHECK_NULL_RETURN(func_graph);
    ParameterPtr value_param = nullptr;
    std::vector<int32_t> values = {0, 0, 0, 0};
    value_param = opt::BuildIntVecParameterNode(func_graph, values, cnode->fullname_with_scope() + "_values");
    MS_CHECK_TRUE_MSG(value_param != nullptr, RET_ERROR, "Build parameter node failed.");

    // Add input input_size
    auto inputs = cnode->inputs();
    inputs.insert(inputs.begin() + 1, value_param);

    auto f_graph = cnode->func_graph();
    MS_CHECK_TRUE_MSG(f_graph != nullptr, RET_ERROR, "func_graph is nullptr.");

    auto manager = Manage(f_graph, true);
    MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "manager is nullptr.");

    manager->AddEdge(cnode, value_param);
    for (size_t i = 0; i < inputs.size(); i++) {
      manager->SetEdge(cnode, i, inputs[i]);
    }
  }
  auto status = AttrAdjust(dst_prim, ops::kDilation);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust dilation failed.";
    return status;
  }
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

STATUS Conv2dTransposeMapper::AdjustGeAttr(const CNodePtr &cnode, const PrimitivePtr &dst_prim) {
  MS_CHECK_TRUE_MSG(dst_prim != nullptr, RET_ERROR, "dst_prim is nullptr.");

  if (AttrAdjust(dst_prim, ops::kStride) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust strides failed.";
    return RET_ERROR;
  }
  if (AdjustAttrFormat(dst_prim, kNameFormat) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust format failed.";
    return RET_ERROR;
  }
  if (AdjustOutputPadding(dst_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust output padding failed.";
    return RET_ERROR;
  }

  // Add attr offset
  int64_t offset = 0;
  dst_prim->AddAttr(ops::kOffset, MakeValue(offset));
  return RET_OK;
}

STATUS Conv2dTransposeMapper::AdjustOutputPadding(const PrimitivePtr &dst_prim) {
  const int kDim3D = 2;
  const int kDim4D = 3;
  std::vector<int64_t> output_padding = {0, 0, 0, 0};
  auto value_ptr = dst_prim->GetAttr(ops::kOutputPaddings);
  if (value_ptr != nullptr) {
    std::vector<int64_t> val = GetValue<std::vector<int64_t>>(value_ptr);
    if (val.size() != kNameOutputPaddingNum) {
      MS_LOG(ERROR) << "Num of output padding must be " << kNameOutputPaddingNum << ", real num: " << val.size();
      return RET_ERROR;
    }
    output_padding[kDim3D] = val[0];
    output_padding[kDim4D] = val[1];
  }
  dst_prim->set_attr(ops::kOutputPaddings, MakeValue(output_padding));
  auto pad_list_value_ptr = dst_prim->GetAttr(ops::kPadList);
  if (!pad_list_value_ptr) {
    dst_prim->set_attr(ops::kPadList, MakeValue(std::vector<int64_t>{0, 0, 0, 0}));
  }
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameConv2dTransposeFusion, Conv2dTransposeMapper)
}  // namespace lite
}  // namespace mindspore
