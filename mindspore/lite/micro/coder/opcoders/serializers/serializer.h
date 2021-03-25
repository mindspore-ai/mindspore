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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_SERIALIZERS_SERIALIZER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_SERIALIZERS_SERIALIZER_H_

#include <vector>
#include <string>
#include <sstream>
#include "coder/allocator/allocator.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_stream_utils.h"

namespace mindspore::lite::micro {

/*
 *  convert array T[] to string
 *  std::ostream &operator<<(std::ostream &, const ::T &) must exist
 *  arr shouldn't be pointer, T* is not valid
 *  example:
 *      int arr[] = {1, 2, 3};
 *      ToString(arr);
 *  the code above would produce:
 *      "{1, 2, 3}"
 */
template <typename T, unsigned int N>
std::string ToString(const T (&arr)[N]) {
  std::stringstream code;
  int n = N;
  while (n > 0 && arr[n - 1] == 0) {
    n--;
  }
  code << "{";
  for (int i = 0; i < n - 1; ++i) {
    code << arr[i] << ", ";
  }
  if (n > 0) {
    code << arr[n - 1];
  }
  code << "}";
  return code.str();
}

class Serializer {
 public:
  Serializer() = default;
  virtual ~Serializer() = default;

  /*
   * Code function call to generated code
   * First parameter is the function name, the rest are the parameters of the function
   * example:
   *    CodeFunction("function", "foo", "bar", "foobar", 42);
   * the code above would produce:
   *    "function("foo", "bar", "foobar", 42);\n"
   */
  template <typename... PARAMETERS>
  void CodeFunction(const std::string &name, PARAMETERS... parameters) {
    code << name << "(";
    GenCode(parameters...);
    code << ");\n";
  }

  /*
   * Code function call to generated code, with checking the return code
   * First parameter is the function name, the rest are the parameters of the function
   * example:
   *    CodeFunctionWithCheck("function", "foo", "bar", "foobar", 42);
   * the code above would produce:
   * """
   *    if(function("foo", "bar", "foobar", 42) != 0) {\n
   *       return -1;
   *    }
   * """
   */
  template <typename... PARAMETERS>
  void CodeFunctionWithCheck(const std::string &name, PARAMETERS... parameters) {
    code << "if(" << name << "(";
    GenCode(parameters...);
    code << ") != RET_OK) {\n";
    code << "  return RET_ERROR;\n";
    code << "}\n";
  }

  /*
   * helper function for coding
   * example:
   *    int bar[] = {1 ,3, 2};
   *    CodeArray("bar", bar, 3);
   * the code above would produce:
   *    "int bar[3] = {1 ,3, 2};\n"
   */
  template <typename T>
  void CodeArray(const std::string &name, T *data, int length, bool is_const = true) {
    std::string type = GetVariableTypeName<T>();
    if (is_const) {
      code << "const " << type << " " << name << "[" << length << "] = {";
    } else {
      code << type << " " << name << "[" << length << "] = {";
    }
    for (int i = 0; i < length - 1; ++i) {
      code << data[i] << ", ";
    }
    if (length > 0) {
      code << data[length - 1];
    }
    code << "};\n";
  }

  template <typename T>
  void CodeMallocExpression(T t, size_t size) {
    if (size == 0) {
      MS_LOG(ERROR) << "CodeMallocExpression size is zero";
      exit(1);
    }
    GenCode(t);
    code << " = malloc(" << size << ");\n";
    code << "if (";
    GenCode(t);
    code << " == NULL) {\n";
    code << "  return RET_ERROR;\n";
    code << "}\n";
  }

  std::streamsize precision(std::streamsize size) {
    std::streamsize old = code.precision(size);
    return old;
  }

  std::string str() const { return code.str(); }

  template <typename T>
  Serializer &operator<<(T t) {
    code << t;
    return *this;
  }

  /*
   * helper function for CodeStruct
   * all parameters should be
   * example:
   * given:
   *    typedef struct Foo {
   *      int array[5];
   *      int *pointer;
   *      int count;
   *    } Foo;
   *    int pointer[] = {1 ,3, 2, 42};
   *    Foo foo = {{1, 2, 3}, pointer, 4};
   *    the CodeStruct should be written as:
   *    CodeStruct(const string &name, const Foo &foo) {
   *      CodeArray("pointer_gen", foo.pointer, foo.count);
   *      CodeBaseStruct("Foo", "foo_gen", ToString(foo.array), "pointer_gen", foo.count);
   *    }
   * the code above would produce:
   *    "int pointer_gen[4] = {1 ,3, 2, 42};\n
   *    const Foo foo_gen = {{1, 2, 3}, pointer_gen, 4};\n"
   */
  template <bool immutable = true, typename... PARAMETERS>
  void CodeBaseStruct(const std::string &type, const std::string &name, PARAMETERS... parameters) {
    if constexpr (immutable) {
      code << "const " << type << " " << name << " = {";
    } else {
      code << type << " " << name << " = {";
    }
    GenCode(parameters...);
    code << "};\n";
  }

 protected:
  std::ostringstream code;

 private:
  /*
   *   function GenCode(Args... args)
   *   Convert all parameters to string, and join connect them with comma ", "
   *   example:
   *      GenCode(true, false, static_cast<int8_t>(12), static_cast<uint8_t>(57), 'c', 5567);
   *   the code above would produce:
   *      "true, false, 12, 57, c, 5567"
   */
  template <typename T, typename... REST>
  void GenCode(T t, REST... args) {
    GenCode(t);
    code << ", ";
    GenCode(args...);
  }
  template <typename T>
  void GenCode(T t) {
    code << t;
  }

  /*
   *  Convert pointer to string when it's in MemoryAllocator (and it should be)
   *  if t is not in the table of MemoryAllocator, it would return empty string ""
   *  then the coder would generate something like
   *    {foo, , bar}
   *  and make the generated code
   *  not compilable rather than generating code like
   *    {foo, 0x7ffed0cd377c, bar}
   *  which would bring the hard coded address to the runtime and make it harder to debug
   *
   *  if t is nullptr, "NULL" would be coded to generated code because some pointer might
   *  be nullptr in some cases and we want to code it.
   *  In this function, passing nullptr **would not** be regarded as a bug or mistake
   */
  template <typename T>
  void GenCode(T *t) {
    if (t == nullptr) {
      code << "NULL";
    } else {
      std::string name = MemoryAllocator::GetInstance()->GetRuntimeAddr(t);
      if (name.empty()) {
        MS_LOG(ERROR) << "pointer is not allocated by the allocator";
        exit(1);
      }
      code << name;
    }
  }

  // std::boolalpha converts bool to string literals {"true", "false"} instead of {1, 0}
  void GenCode(bool t) { code << std::boolalpha << t; }
  void GenCode(int8_t t) { code << std::to_string(t); }
  void GenCode(uint8_t t) { code << std::to_string(t); }
  void GenCode(decltype(nullptr) t) { code << "NULL"; }
  void GenCode(const char *t) { code << t; }
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_SERIALIZERS_SERIALIZER_H_
