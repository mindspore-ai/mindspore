/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/gelu.h"
#include <memory>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
// int GeLU::GetApproximate() const { return this->primitive_->value.AsGeLU()->approximate; }

// void GeLU::SetApproximate(bool approximate) { this->primitive_->value.AsGeLU()->approximate = approximate; }

#else
// int GeLU::GetApproximate() const { return this->primitive_->value_as_GeLU()->approximate(); }

// PrimitiveC *GeLUCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<GeLU>(primitive); }
// Registry GeLURegistry(schema::PrimitiveType_GeLU, GeLUCreator);

#endif

}  // namespace lite
}  // namespace mindspore
