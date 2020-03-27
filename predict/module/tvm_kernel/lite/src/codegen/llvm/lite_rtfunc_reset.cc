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

#include <dmlc/logging.h>
#include <string>
#include "tvm/api_registry.h"

namespace tvm {
namespace codegen {
class LiteRTFuncReseter {
 public:
  LiteRTFuncReseter() {}
  ~LiteRTFuncReseter() {}
  int InsertFuncPair(std::string sfunc, std::string dfunc) {
    CHECK_NE(sfunc.size(), 0);
    CHECK_NE(dfunc.size(), 0);
    func_map_[sfunc] = dfunc;
    return 0;
  }

  /*
   * the llvm::Function::Create need a longe life scope const char* as input
   * so here not return block life scopte tmp std::string.
   */
  const char* GetResetFunc(std::string sfunc) {
    CHECK_NE(sfunc.size(), 0);
    auto it_dfunc = func_map_.find(sfunc);
    if (it_dfunc != func_map_.end()) {
      return it_dfunc->second.c_str();
    } else {
      func_map_[sfunc] = sfunc;
      return func_map_[sfunc].c_str();
    }
  }

  /*
   * not real delete item paire, just set orig function pair
   */
  int DeleteFuncPair(std::string sfunc) {
    CHECK_NE(sfunc.size(), 0);
    func_map_[sfunc] = sfunc;
    return 0;
  }
  static LiteRTFuncReseter* GetRTFuncReseter() {
    static LiteRTFuncReseter inst;
    return &inst;
  }

 private:
  std::map<std::string, std::string> func_map_;
};

TVM_REGISTER_API("codegen.SetRTFuncTransPair").set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  *rv = LiteRTFuncReseter::GetRTFuncReseter()->InsertFuncPair(targs[0], targs[1]);
});

TVM_REGISTER_API("codegen.DelRTFuncTransPair").set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  *rv = LiteRTFuncReseter::GetRTFuncReseter()->DeleteFuncPair(targs[0]);
});

/*
 * now no operator=(const char* ) provide for TVMRetValue
 * here using explicit operator call function to make sure not using operator=(int)
 */
TVM_REGISTER_API("codegen.GetTransRTFunc").set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  (*rv).operator=(
    reinterpret_cast<void*>(const_cast<char*>(LiteRTFuncReseter::GetRTFuncReseter()->GetResetFunc(targs[0]))));
});
}  // namespace codegen
}  // namespace tvm
