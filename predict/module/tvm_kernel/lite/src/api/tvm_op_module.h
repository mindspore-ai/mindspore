/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this ${file} except in compliance with the License.
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
#ifndef PREDICT_MODULE_TVM_KERNEL_LITE_SRC_API_TVM_OP_MODULE_H_
#define PREDICT_MODULE_TVM_KERNEL_LITE_SRC_API_TVM_OP_MODULE_H_

#include "src/op_registry.h"
namespace mindspore {
namespace predict {
class TVMOpRegistry : public OpRegistry {
 public:
  TVMOpRegistry();
  OpCreator GetOpCreator(const OpDesc &desc) override;
};

class TVMOpModule : public Module<OpRegistry> {
 public:
  OpRegistry *GetInstance() override;
};
}  // namespace predict
}  // namespace mindspore
#endif  // PREDICT_MODULE_TVM_KERNEL_LITE_SRC_API_TVM_OP_MODULE_H_
