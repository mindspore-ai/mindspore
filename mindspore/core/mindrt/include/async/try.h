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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_TRY_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_TRY_H

#include "async/failure.h"
#include "async/option.h"
#include "async/status.h"

namespace mindspore {
template <typename T, typename F = Failure>
class Try {
 public:
  Try() : errorCode(Status::KOK) {}

  Try(const T &t) {
    data = Some(t);
    errorCode = Status::KOK;
  }

  Try(const F &errCode) { errorCode = errCode; }

  virtual ~Try() {}

  bool IsOK() { return !IsError(); }

  bool IsError() { return data.IsNone(); }

  const T &Get() const { return data.Get(); }

  const int GetErrorCode() const { return errorCode.GetErrorCode(); }

 private:
  Option<T> data;
  Failure errorCode;
};
}  // namespace mindspore
#endif
