/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/util/services.h"

namespace mindspore {
namespace dataset {
// This is a one-time global initializer which includes the call to instantiate singletons.
// It is external api call and not a member of the GlobalContext directly.
Status GlobalInit() {
  // Bring up all the services (logger, task, bufferpool)
  return (Services::CreateInstance());
}
}  // namespace dataset
}  // namespace mindspore
