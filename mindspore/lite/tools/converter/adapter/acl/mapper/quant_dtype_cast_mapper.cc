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

#include "tools/converter/adapter/acl/mapper/quant_dtype_cast_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/log_util.h"
#include "ops/op_name.h"
#include "ops/quant_dtype_cast.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kQuantInputNum = 2;
constexpr auto kDequantInputNum = 3;
}  // namespace

STATUS QuantDTypeCastMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }

  PrimitivePtr dst_prim = nullptr;
  if (cnode->inputs().size() == kQuantInputNum) {
    // map to Quant.
    auto quant_params_holder_attr = src_prim->GetAttr("quant_params");
    CHECK_NULL_RETURN(quant_params_holder_attr);
    auto quant_params_holder = quant_params_holder_attr->cast<QuantParamHolderPtr>();
    CHECK_NULL_RETURN(quant_params_holder);
    MS_CHECK_TRUE_RET(!quant_params_holder->get_output_quant_params().empty(), RET_ERROR);
    auto quant_param = quant_params_holder->get_output_quant_params().front();
    MS_CHECK_TRUE_RET(!quant_param.empty(), RET_ERROR);
    dst_prim = std::make_shared<acl::Quant>();
    CHECK_NULL_RETURN(dst_prim);
    dst_prim->AddAttr("scale", MakeValue(static_cast<float>(quant_param.front().scale)));
    dst_prim->AddAttr("offset", MakeValue(static_cast<float>(quant_param.front().zeroPoint)));
    MS_LOG(INFO) << cnode->fullname_with_scope() << " scale:" << quant_param.front().scale;
    MS_LOG(INFO) << cnode->fullname_with_scope() << " offset:" << quant_param.front().zeroPoint;
  } else if (cnode->inputs().size() == kDequantInputNum) {
    // map to Dequant.
    dst_prim = std::make_shared<acl::Dequant>();
    auto dst_type = src_prim->GetAttr(mindspore::ops::kDstT);
    if (dst_type != nullptr) {
      auto origin_type = static_cast<TypeId>(opt::CastToInt(dst_type).front());
      dst_prim->AddAttr("dtype", TypeIdToType(origin_type));
    }
    CHECK_NULL_RETURN(dst_prim);
  } else {
    MS_LOG(ERROR) << "Invalid input size: " << cnode->inputs().size();
    return lite::RET_ERROR;
  }

  value_node->set_value(dst_prim);
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameQuantDTypeCast, QuantDTypeCastMapper)
}  // namespace lite
}  // namespace mindspore
