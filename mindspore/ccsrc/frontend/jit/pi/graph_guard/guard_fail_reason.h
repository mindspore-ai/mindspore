/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_PI_JIT_GUARD_FAIL_REASON_H
#define MINDSPORE_PI_JIT_GUARD_FAIL_REASON_H

// guard fail reason enum
#define GUARD_FAIL_REASON_ENUM                                           \
  GUARD_FAIL_REASON_KIND(kValueNotEqual, "Value equality check failed.") \
  GUARD_FAIL_REASON_KIND(kShapeNotEqual, "Shape equality check failed.") \
  GUARD_FAIL_REASON_KIND(kTypeNotEqual, "Type equality check failed.")   \
  GUARD_FAIL_REASON_KIND(kReasonUnknown, "Check failed reason unknown.")

enum class GuardFailReason {
#define GUARD_FAIL_REASON_KIND(kind, desc) kind,
  GUARD_FAIL_REASON_ENUM
#undef GUARD_FAIL_REASON_KIND
};

constexpr const char *GetGuardFailReasonDesc(GuardFailReason reason) {
#define GUARD_FAIL_REASON_KIND(kind, desc) \
  if (GuardFailReason::kind == reason) {   \
    return desc;                           \
  }

  GUARD_FAIL_REASON_ENUM
#undef GUARD_FAIL_REASON_KIND
  return "???";
}
#undef GUARD_FAIL_REASON_ENUM

#endif  // MINDSPORE_PI_JIT_GUARD_FAIL_REASON_H
