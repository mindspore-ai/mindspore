/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_CXX_API_CALLBACK_CALLBACK_IMPL_H_
#define MINDSPORE_LITE_SRC_CXX_API_CALLBACK_CALLBACK_IMPL_H_

#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <unordered_map>
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/cell.h"
#include "include/lite_session.h"
#include "include/train/train_loop_callback.h"

namespace mindspore {

class CallbackImpl {
 public:
  CallbackImpl() = delete;
  explicit CallbackImpl(session::TrainLoopCallBack *cb) : internal_call_back_(cb) {}
  session::TrainLoopCallBack *GetInternalCallback() { return internal_call_back_; }

 protected:
  session::TrainLoopCallBack *internal_call_back_ = nullptr;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_CXX_API_CALLBACK_CALLBACK_IMPL_H_
