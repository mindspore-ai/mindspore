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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_OVERFLOW_DUMPER_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_OVERFLOW_DUMPER_H_
#include <memory>
#include <string>
#include <map>

#ifndef ENABLE_SECURITY
#include "include/common/utils/anfalgo.h"
#include "mindspore/ccsrc/kernel/kernel.h"
#endif

namespace mindspore {
namespace debug {
class BACKEND_EXPORT OverflowDumper {
 public:
  OverflowDumper() = default;
  ~OverflowDumper() = default;

  static std::shared_ptr<OverflowDumper> GetInstance(const std::string &name) noexcept;
  static bool Register(const std::string &name, const std::shared_ptr<OverflowDumper> &instance);
  static void Clear();
  virtual void OpLoadDumpInfo(const CNodePtr &kernel) = 0;
  virtual void Init() = 0;
  virtual void OpDebugRegisterForStream(const CNodePtr &kernel) = 0;
  virtual void OpDebugUnregisterForStream() = 0;

 private:
  static std::map<std::string, std::shared_ptr<OverflowDumper>> &GetInstanceMap();
};
}  // namespace debug
}  // namespace mindspore

#define DUMPER_REG(NAME, CLAZZ)                      \
  static bool g_OverflowDumper_##NAME##_reg_result = \
    mindspore::debug::OverflowDumper::Register(NAME, std::make_shared<CLAZZ>())

#endif  // MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_OVERFLOW_DUMPER_H_
