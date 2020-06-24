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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_OP_ATTR_PACKER_H_
#define MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_OP_ATTR_PACKER_H_

#include <utility>
#include <string>
#include <unordered_map>
#include "session/anf_runtime_algorithm.h"
#include "predict/schema/inner/ms_generated.h"

static constexpr size_t kNIndex = 0;
static constexpr size_t kCIndex = 1;
static constexpr size_t kHIndex = 2;
static constexpr size_t kWIndex = 3;
static constexpr size_t kNCHWSize = 4;
namespace mindspore {
namespace predict {
namespace convert {
using OpAttrPackFun = bool (*)(const CNodePtr &c_node_ptr, OpDefT *ms_op);
class OpAttrFactory {
 public:
  static OpAttrFactory *GetInstance() {
    static OpAttrFactory instance;
    return &instance;
  }
  OpAttrFactory(const OpAttrFactory &) = delete;
  OpAttrFactory &operator=(const OpAttrFactory &) = delete;
  OpAttrPackFun GetPackFun(const std::string &op_type);
  ~OpAttrFactory() { pack_funs_.clear(); }
  OpAttrFactory();

 private:
  std::unordered_map<std::string, OpAttrPackFun> pack_funs_;
};

mindspore::predict::Format GetAttrFormat(const std::string &format);

mindspore::predict::PadMode GetAttrPadMode(const std::string &pad_mode);
}  // namespace convert
}  // namespace predict
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_CPU_OP_INFO_OP_ATTR_FACTORY_H_
