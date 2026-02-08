/*
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_FUNC_OP_CONV2D_PADDING_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_FUNC_OP_CONV2D_PADDING_H_

#include <vector>
#include <memory>
#include <string>
#include "mindspore/ccsrc/frontend/operator/meta_dsl/common/meta_impl.h"

namespace mindspore::prim {
void CheckConv2DPaddingInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args);
class Conv2DPaddingMetaImpl : public MetaImpl {
 public:
  Conv2DPaddingMetaImpl() : MetaImpl("Conv2DPadding") {}
  ~Conv2DPaddingMetaImpl() override = default;
  MS_DECLARE_PARENT(Conv2DPaddingMetaImpl, MetaImpl)
  void GenerateFunction() override;

 private:
  NodePtr CalcPadding(const NodePtr &in_shape, const NodePtr &w_shape, const NodePtr &stride, const NodePtr &dilation);
};
REGISTER_META_IMPL(Conv2DPadding, CheckConv2DPaddingInputs);
}  // namespace mindspore::prim
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_FUNC_OP_CONV2D_PADDING_H_
