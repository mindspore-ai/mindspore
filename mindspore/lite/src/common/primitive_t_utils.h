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

#ifndef MINDSPORE_LITE_SRC_COMMON_PRIMITIVE_T_UTILS_H_
#define MINDSPORE_LITE_SRC_COMMON_PRIMITIVE_T_UTILS_H_
#ifdef PRIMITIVE_WRITEABLE
#include <memory>
#include "schema/inner/model_generated.h"
#include "src/common/ops/populate/populate_register.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace lite {
const schema::Primitive *ConvertToPrimitive(const schema::PrimitiveT *primitive_t, flatbuffers::FlatBufferBuilder *fbb);
OpParameter *GetOpParameter(const schema::PrimitiveT *primitive_t);
std::unique_ptr<schema::PrimitiveT> GetPrimitiveT(const std::shared_ptr<mindspore::ops::BaseOperator> &op);
}  // namespace lite
}  // namespace mindspore
#endif
#endif  // MINDSPORE_LITE_SRC_COMMON_PRIMITIVE_T_UTILS_H_
