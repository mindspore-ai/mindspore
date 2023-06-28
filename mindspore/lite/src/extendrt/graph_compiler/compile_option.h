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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_COMPILER_COMPILE_OPTION_H
#define MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_COMPILER_COMPILE_OPTION_H

#include <string>
#include <memory>
#include "mindapi/base/format.h"
#include "mindapi/base/type_id.h"
#include "src/extendrt/kernel/kernel_spec_infos.h"

namespace mindspore::lite {
struct CompileOption {
  Format graph_format{Format::NCHW};
  Format graph_input_format{Format::NCHW};
  std::string backend{kernel::kBackendCPU};
  TypeId datatype{kNumberTypeFloat32};
};

using CompileOptionPtr = std::shared_ptr<CompileOption>;
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_COMPILER_COMPILE_OPTION_H
