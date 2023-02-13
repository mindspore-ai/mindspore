/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_API_CALLBACK_CKPT_SAVER_H
#define MINDSPORE_INCLUDE_API_CALLBACK_CKPT_SAVER_H

#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include "include/api/callback/callback.h"
#include "include/api/dual_abi_helper.h"

namespace mindspore {

class MS_API CkptSaver : public TrainCallBack {
 public:
  inline CkptSaver(int save_every_n, const std::string &filename_prefix);
  virtual ~CkptSaver();

 private:
  CkptSaver(int save_every_n, const std::vector<char> &filename_prefix);
};

CkptSaver::CkptSaver(int save_every_n, const std::string &filename_prefix)
    : CkptSaver(save_every_n, StringToChar(filename_prefix)) {}

}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_CALLBACK_CKPT_SAVER_H
