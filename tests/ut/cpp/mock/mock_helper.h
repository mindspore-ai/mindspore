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

#ifndef MINDSPORE_UT_MOCK_MOCK_HELPER_H_
#define MINDSPORE_UT_MOCK_MOCK_HELPER_H_

#include <gmock/gmock.h>

#define REMOVE_PARENS(x) \
  GMOCK_PP_GENERIC_IF(GMOCK_PP_IS_BEGIN_PARENS(x), (GMOCK_PP_REMOVE_PARENS(x)), (GMOCK_PP_IDENTITY(x)))

#define EXPAND_PARAMS_NAME_0(...)
#define EXPAND_PARAMS_NAME_1(a) REMOVE_PARENS(a) _0
#define EXPAND_PARAMS_NAME_2(a, b) REMOVE_PARENS(a) _0, REMOVE_PARENS(b) _1
#define EXPAND_PARAMS_NAME_3(a, b, c) REMOVE_PARENS(a) _0, REMOVE_PARENS(b) _1, REMOVE_PARENS(c) _2
#define EXPAND_PARAMS_NAME_4(a, b, c, d) \
  REMOVE_PARENS(a) _0, REMOVE_PARENS(b) _1, REMOVE_PARENS(c) _2, REMOVE_PARENS(d) _3
#define EXPAND_PARAMS_NAME_5(a, b, c, d, e) \
  REMOVE_PARENS(a)                          \
  _0, REMOVE_PARENS(b) _1, REMOVE_PARENS(c) _2, REMOVE_PARENS(d) _3, REMOVE_PARENS(e) _4
#define EXPAND_PARAMS_NAME_6(a, b, c, d, e, f) \
  REMOVE_PARENS(a)                             \
  _0, REMOVE_PARENS(b) _1, REMOVE_PARENS(c) _2, REMOVE_PARENS(d) _3, REMOVE_PARENS(e) _4, REMOVE_PARENS(f) _5
#define EXPAND_PARAMS_NAME_7(a, b, c, d, e, f, g)                                                              \
  REMOVE_PARENS(a)                                                                                             \
  _0, REMOVE_PARENS(b) _1, REMOVE_PARENS(c) _2, REMOVE_PARENS(d) _3, REMOVE_PARENS(e) _4, REMOVE_PARENS(f) _5, \
    REMOVE_PARENS(g) _6
#define EXPAND_PARAMS_NAME_8(a, b, c, d, e, f, g, h)                                                           \
  REMOVE_PARENS(a)                                                                                             \
  _0, REMOVE_PARENS(b) _1, REMOVE_PARENS(c) _2, REMOVE_PARENS(d) _3, REMOVE_PARENS(e) _4, REMOVE_PARENS(f) _5, \
    REMOVE_PARENS(g) _6, REMOVE_PARENS(h) _7
#define EXPAND_PARAMS_NAME_9(a, b, c, d, e, f, g, h, i)                                                        \
  REMOVE_PARENS(a)                                                                                             \
  _0, REMOVE_PARENS(b) _1, REMOVE_PARENS(c) _2, REMOVE_PARENS(d) _3, REMOVE_PARENS(e) _4, REMOVE_PARENS(f) _5, \
    REMOVE_PARENS(g) _6, REMOVE_PARENS(h) _7, REMOVE_PARENS(i) _8

#define EXTRACT_TYPE_0(...)
#define EXTRACT_TYPE_1(a) REMOVE_PARENS(a)
#define EXTRACT_TYPE_2(a, b) REMOVE_PARENS(a), REMOVE_PARENS(b)
#define EXTRACT_TYPE_3(a, b, c) REMOVE_PARENS(a), REMOVE_PARENS(b), REMOVE_PARENS(c)
#define EXTRACT_TYPE_4(a, b, c, d) REMOVE_PARENS(a), REMOVE_PARENS(b), REMOVE_PARENS(c), REMOVE_PARENS(d)
#define EXTRACT_TYPE_5(a, b, c, d, e) \
  REMOVE_PARENS(a), REMOVE_PARENS(b), REMOVE_PARENS(c), REMOVE_PARENS(d), REMOVE_PARENS(e)
#define EXTRACT_TYPE_6(a, b, c, d, e, f) \
  REMOVE_PARENS(a), REMOVE_PARENS(b), REMOVE_PARENS(c), REMOVE_PARENS(d), REMOVE_PARENS(e), REMOVE_PARENS(f)
#define EXTRACT_TYPE_7(a, b, c, d, e, f, g)                                                                   \
  REMOVE_PARENS(a), REMOVE_PARENS(b), REMOVE_PARENS(c), REMOVE_PARENS(d), REMOVE_PARENS(e), REMOVE_PARENS(f), \
    REMOVE_PARENS(g)
#define EXTRACT_TYPE_8(a, b, c, d, e, f, g, h)                                                                \
  REMOVE_PARENS(a), REMOVE_PARENS(b), REMOVE_PARENS(c), REMOVE_PARENS(d), REMOVE_PARENS(e), REMOVE_PARENS(f), \
    REMOVE_PARENS(g), REMOVE_PARENS(h)
#define EXTRACT_TYPE_9(a, b, c, d, e, f, g, h, i)                                                             \
  REMOVE_PARENS(a), REMOVE_PARENS(b), REMOVE_PARENS(c), REMOVE_PARENS(d), REMOVE_PARENS(e), REMOVE_PARENS(f), \
    REMOVE_PARENS(g), REMOVE_PARENS(h), REMOVE_PARENS(i)

#define EXTRACT_TYPE_WITH_PARENS_0(...)
#define EXTRACT_TYPE_WITH_PARENS_1(a) (REMOVE_PARENS(a))
#define EXTRACT_TYPE_WITH_PARENS_2(a, b) (REMOVE_PARENS(a)), (REMOVE_PARENS(b))
#define EXTRACT_TYPE_WITH_PARENS_3(a, b, c) (REMOVE_PARENS(a)), (REMOVE_PARENS(b)), (REMOVE_PARENS(c))
#define EXTRACT_TYPE_WITH_PARENS_4(a, b, c, d) \
  (REMOVE_PARENS(a)), (REMOVE_PARENS(b)), (REMOVE_PARENS(c)), (REMOVE_PARENS(d))
#define EXTRACT_TYPE_WITH_PARENS_5(a, b, c, d, e) \
  (REMOVE_PARENS(a)), (REMOVE_PARENS(b)), (REMOVE_PARENS(c)), (REMOVE_PARENS(d)), (REMOVE_PARENS(e))
#define EXTRACT_TYPE_WITH_PARENS_6(a, b, c, d, e, f) \
  (REMOVE_PARENS(a)), (REMOVE_PARENS(b)), (REMOVE_PARENS(c)), (REMOVE_PARENS(d)), (REMOVE_PARENS(e)), (REMOVE_PARENS(f))
#define EXTRACT_TYPE_WITH_PARENS_7(a, b, c, d, e, f, g)                                               \
  (REMOVE_PARENS(a)), (REMOVE_PARENS(b)), (REMOVE_PARENS(c)), (REMOVE_PARENS(d)), (REMOVE_PARENS(e)), \
    (REMOVE_PARENS(f)), (REMOVE_PARENS(g))
#define EXTRACT_TYPE_WITH_PARENS_8(a, b, c, d, e, f, g, h)                                            \
  (REMOVE_PARENS(a)), (REMOVE_PARENS(b)), (REMOVE_PARENS(c)), (REMOVE_PARENS(d)), (REMOVE_PARENS(e)), \
    (REMOVE_PARENS(f)), (REMOVE_PARENS(g)), (REMOVE_PARENS(h))
#define EXTRACT_TYPE_WITH_PARENS_9(a, b, c, d, e, f, g, h, i)                                         \
  (REMOVE_PARENS(a)), (REMOVE_PARENS(b)), (REMOVE_PARENS(c)), (REMOVE_PARENS(d)), (REMOVE_PARENS(e)), \
    (REMOVE_PARENS(f)), (REMOVE_PARENS(g)), (REMOVE_PARENS(h)), (REMOVE_PARENS(i))

#define TRANS_TYPE_TO_NAME_0(...)
#define TRANS_TYPE_TO_NAME_1(a) _0
#define TRANS_TYPE_TO_NAME_2(a, b) _0, _1
#define TRANS_TYPE_TO_NAME_3(a, b, c) _0, _1, _2
#define TRANS_TYPE_TO_NAME_4(a, b, c, d) _0, _1, _2, _3
#define TRANS_TYPE_TO_NAME_5(a, b, c, d, e) _0, _1, _2, _3, _4
#define TRANS_TYPE_TO_NAME_6(a, b, c, d, e, f) _0, _1, _2, _3, _4, _5
#define TRANS_TYPE_TO_NAME_7(a, b, c, d, e, f, g) _0, _1, _2, _3, _4, _5, _6
#define TRANS_TYPE_TO_NAME_8(a, b, c, d, e, f, g, h) _0, _1, _2, _3, _4, _5, _6, _7
#define TRANS_TYPE_TO_NAME_9(a, b, c, d, e, f, g, h, i) _0, _1, _2, _3, _4, _5, _6, _7, _8

#define CONCAT(x, y) x##y
#define EXPAND_THEN_CONCAT(x, y) CONCAT(x, y)
#define GEN_PARAMS_LIST_TYPE_AND_NAME(...) \
  EXPAND_THEN_CONCAT(EXPAND_PARAMS_NAME_, GMOCK_PP_NARG(__VA_ARGS__))(__VA_ARGS__)
#define GEN_PARAMS_LIST_ONLY_NAME(...) EXPAND_THEN_CONCAT(TRANS_TYPE_TO_NAME_, GMOCK_PP_NARG(__VA_ARGS__))(__VA_ARGS__)
#define GEN_PARAMS_LIST_ONLY_TYPE(...) EXPAND_THEN_CONCAT(EXTRACT_TYPE_, GMOCK_PP_NARG(__VA_ARGS__))(__VA_ARGS__)
#define GEN_PARAMS_LIST_ONLY_TYPE_WITH_PARENS(...) \
  EXPAND_THEN_CONCAT(EXTRACT_TYPE_WITH_PARENS_, GMOCK_PP_NARG(__VA_ARGS__))(__VA_ARGS__)

#define MOCK_CPP_FUNC(func_name, return_type, ...)                                           \
  return_type func_name(GEN_PARAMS_LIST_TYPE_AND_NAME(__VA_ARGS__)) {                        \
    return g_mock_##func_name##_instance->func_name(GEN_PARAMS_LIST_ONLY_NAME(__VA_ARGS__)); \
  }

#define MOCK_C_FUNC(func_name, return_type, ...)                                             \
  extern "C" {                                                                               \
  return_type func_name(GEN_PARAMS_LIST_TYPE_AND_NAME(__VA_ARGS__)) {                        \
    return g_mock_##func_name##_instance->func_name(GEN_PARAMS_LIST_ONLY_NAME(__VA_ARGS__)); \
  }                                                                                          \
  }

#define MOCK_CLASS_FUNC(class_name, func_name, return_type, ...)                                            \
  return_type class_name::func_name(GEN_PARAMS_LIST_TYPE_AND_NAME(__VA_ARGS__)) {                           \
    return g_mock_##class_name##_##func_name##_instance->func_name(GEN_PARAMS_LIST_ONLY_NAME(__VA_ARGS__)); \
  }

#define MOCK_H(func_name, return_type, ...)                                                                \
  class func_name##BaseClass {                                                                             \
   public:                                                                                                 \
    virtual ~func_name##BaseClass() = default;                                                             \
    virtual return_type func_name(GEN_PARAMS_LIST_ONLY_TYPE(__VA_ARGS__)) = 0;                             \
  };                                                                                                       \
  class func_name##MockClass : public func_name##BaseClass {                                               \
   public:                                                                                                 \
    ~func_name##MockClass() override = default;                                                            \
    MOCK_METHOD(return_type, func_name, (GEN_PARAMS_LIST_ONLY_TYPE_WITH_PARENS(__VA_ARGS__)), (override)); \
  };                                                                                                       \
  inline std::shared_ptr<func_name##MockClass> g_mock_##func_name##_instance = nullptr;                    \
  inline std::mutex g_mock_##func_name##_instance_mutex;

#define START_MOCK(func_name)                                                                  \
  std::lock_guard<std::mutex> lock_##func_name(g_mock_##func_name##_instance_mutex);           \
  auto local_##func_name##_instance = std::make_shared<func_name##MockClass>();                \
  struct func_name##GlobalInstanceGuard {                                                      \
   public:                                                                                     \
    explicit func_name##GlobalInstanceGuard(const std::shared_ptr<func_name##MockClass> &in) { \
      g_mock_##func_name##_instance = in;                                                      \
    }                                                                                          \
    ~func_name##GlobalInstanceGuard() { g_mock_##func_name##_instance = nullptr; }             \
  } func_name##_guard(local_##func_name##_instance);

#define MOCK_OBJECT(func_name) *g_mock_##func_name##_instance

#endif  // MINDSPORE_UT_MOCK_MOCK_HELPER_H_
