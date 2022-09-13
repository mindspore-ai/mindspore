/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_MINDIR_MODEL_UTIL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_MINDIR_MODEL_UTIL_H_

#include <memory>
#include "ir/anf.h"
#include "mindapi/base/type_id.h"
#include "proto/mind_ir.pb.h"
#include "include/api/context.h"

namespace mindspore::infer::mindir {
class MindirModelUtil {
 public:
  static mindspore::ValuePtr MakeValueFromAttribute(const mind_ir::AttributeProto &attr_proto);

  static mindspore::ValuePtr MakeValueFromTensorOrTypeAttribute(const mind_ir::AttributeProto &attr_proto);
  static mindspore::ValuePtr MakeValueFromTensorAttribute(const mind_ir::TensorProto &attr_tensor,
                                                          bool need_load_data = false);
  static mindspore::ValuePtr MakeValueFromListAttribute(const mind_ir::AttributeProto &attr_proto);
  static mindspore::ValuePtr MakeValueFromScalarAttribute(const mind_ir::AttributeProto &attr_proto);

  static mindspore::TypeId ProtoTypeToTypeId(int32_t proto_type);
  static bool NeedRuntimeConvert(const void *model_data, size_t data_size,
                                 const std::shared_ptr<mindspore::Context> &context);
};
}  // namespace mindspore::infer::mindir

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_MINDIR_MODEL_UTIL_H_
