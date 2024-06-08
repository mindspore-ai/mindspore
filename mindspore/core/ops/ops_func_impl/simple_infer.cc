/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore::ops {
SimpleInfer &SimpleInfer::Instance() noexcept {
  static SimpleInfer instance;
  return instance;
}

ops::OpFuncImplPtr SimpleInfer::GetFunc(const string &op_name) {
  auto iter = simple_infer_fun_.find(op_name);
  if (iter == simple_infer_fun_.end()) {
    return nullptr;
  }
  return iter->second;
}

void SimpleInfer::Register(const std::string &op_name, ops::OpFuncImplPtr &&func) {
  MS_LOG(DEBUG) << "Reg simple infer for op " << op_name;
  auto ret = simple_infer_fun_.try_emplace(op_name, func);
  if (!ret.second) {
    MS_LOG(WARNING) << "Duplicate simpler infer for " << op_name;
  }
}

void SimpleInfer::DoSimpleInfer(const PrimitivePtr &primitive, const ValueSimpleInfoPtr &value_simple_info,
                                const ops::OpFuncImplPtr &simple_infer_func, const ValuePtrList &input_values) {
  value_simple_info->shape_vector_ = simple_infer_func->InferShape(primitive, input_values);
  value_simple_info->dtype_vector_ = simple_infer_func->InferType(primitive, input_values);
  value_simple_info->size_ = value_simple_info->shape_vector_.size();
  if (value_simple_info->size_ != value_simple_info->dtype_vector_.size()) {
    MS_LOG(EXCEPTION) << "Infer shape size " << value_simple_info->size_ << " is not equal to dtype size "
                      << value_simple_info->dtype_vector_.size();
  }
}

ValueSimpleInfoPtr InferBySimple(const PrimitivePtr &primitive, const ValuePtrList &input_values) {
  const auto &simple_infer_func = SimpleInfer::Instance().GetFunc(primitive->name());
  if (simple_infer_func == nullptr) {
    return nullptr;
  }
  auto value_simple_info = std::make_shared<ValueSimpleInfo>();
  SimpleInfer::Instance().DoSimpleInfer(primitive, value_simple_info, simple_infer_func, input_values);
  return value_simple_info;
}
}  // namespace mindspore::ops
