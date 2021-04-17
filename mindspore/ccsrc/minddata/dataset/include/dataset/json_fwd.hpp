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

#ifndef INCLUDE_NLOHMANN_JSON_FWD_HPP_
#define INCLUDE_NLOHMANN_JSON_FWD_HPP_

#include <cstdint>  // int64_t, uint64_t
#include <map>      // map
#include <memory>   // allocator
#include <string>   // string
#include <vector>   // vector

/*!
@brief namespace for Niels Lohmann
@see https://github.com/nlohmann
@since version 1.0.0
*/
namespace nlohmann {
/*!
@brief default JSONSerializer template argument

This serializer ignores the template arguments and uses ADL
([argument-dependent lookup](https://en.cppreference.com/w/cpp/language/adl))
for serialization.
*/
template <typename T = void, typename SFINAE = void>
struct adl_serializer;

template <template <typename U, typename V, typename... Args> class ObjectType = std::map,
          template <typename U, typename... Args> class ArrayType = std::vector, class StringType = std::string,
          class BooleanType = bool, class NumberIntegerType = std::int64_t, class NumberUnsignedType = std::uint64_t,
          class NumberFloatType = double, template <typename U> class AllocatorType = std::allocator,
          template <typename T, typename SFINAE = void> class JSONSerializer = adl_serializer>
class basic_json;

/*!
@brief JSON Pointer

A JSON pointer defines a string syntax for identifying a specific value
within a JSON document. It can be used with functions `at` and
`operator[]`. Furthermore, JSON pointers are the base for JSON patches.

@sa [RFC 6901](https://tools.ietf.org/html/rfc6901)

@since version 2.0.0
*/
template <typename BasicJsonType>
class json_pointer;

/*!
@brief default JSON class

This type is the default specialization of the @ref basic_json class which
uses the standard template types.

@since version 1.0.0
*/
using json = basic_json<>;
}  // namespace nlohmann

#endif  // INCLUDE_NLOHMANN_JSON_FWD_HPP_
