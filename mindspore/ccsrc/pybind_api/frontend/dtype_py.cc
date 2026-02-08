/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "ir/dtype.h"
#include "ir/dtype/ref.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "pybind_api/frontend/frontend_api.h"

namespace mindspore {
// Define python wrapper to handle data types.
void RegTyping(py::module *m) {
  auto m_sub = m->def_submodule("typing", "submodule for dtype");
  py::enum_<TypeId>(m_sub, "TypeId");
  (void)m_sub.def("is_subclass", &IsIdentidityOrSubclass, "is equal or subclass");
  (void)m_sub.def("load_type", &TypeIdToType, "load type");
  (void)m_sub.def("dump_type", [](const TypePtr &t) { return t->type_id(); }, py::arg("t").none(false), "dump type");
  (void)m_sub.def("str_to_type", &StringToType, "string to typeptr");
  (void)m_sub.def("type_size_in_bytes", &GetTypeByte, "type size in bytes");
  (void)m_sub.def(
    "type_to_type_id", [](const TypePtr &t) { return GetTypeId(t->type_id()); }, py::arg("t").none(false),
    "convert type to type id enum value");
  (void)m_sub.def(
    "type_id_to_type", [](const int &t) { return TypeIdToType(TypeId(t)); }, "convert type id enum value to type");
  (void)py::class_<Type, std::shared_ptr<Type>>(m_sub, "Type")
    .def("__eq__",
         [](const TypePtr &t1, const py::object &other) {
           if (!py::isinstance<Type>(other)) {
             return false;
           }
           auto t2 = py::cast<TypePtr>(other);
           if (t1 != nullptr && t2 != nullptr) {
             return *t1 == *t2;
           }
           return false;
         })
    .def("__hash__", &Type::hash)
    .def("__str__", &Type::ToString)
    .def("__repr__", &Type::ReprString)
    .def("__deepcopy__", [](const TypePtr &t, py::dict) {
      if (t == nullptr) {
        return static_cast<TypePtr>(nullptr);
      }
      return t->DeepCopy();
    });
  (void)py::class_<Number, Type, std::shared_ptr<Number>>(m_sub, "Number")
    .def(py::init())
    .def_property_readonly("itemsize", &Number::ItemSize)
    .def_property_readonly("is_signed", &Number::IsSigned)
    .def_property_readonly("is_floating_point", &Number::IsFloatingPoint)
    .def_property_readonly("is_complex", &Number::IsComplex)
    .def("to_real", &Number::ToReal)
    .def("to_complex", &Number::ToComplex);
  (void)py::class_<Bool, Number, std::shared_ptr<Bool>>(m_sub, "Bool")
    .def(py::init())
    .def(py::pickle(
      [](const Bool &) {  // __getstate__
        return py::make_tuple();
      },
      [](const py::tuple &) {  // __setstate__
        return std::make_shared<Bool>();
      }));
  (void)py::class_<Int, Number, std::shared_ptr<Int>>(m_sub, "Int")
    .def(py::init())
    .def(py::init<int>(), py::arg("nbits"))
    .def(py::pickle(
      [](const Int &t) {  // __getstate__
        /* Return a tuple that fully encodes the state of the object */
        return py::make_tuple(py::int_(t.nbits()));
      },
      [](const py::tuple &t) {  // __setstate__
        if (t.size() != 1) {
          throw std::runtime_error("Invalid state!");
        }
        /* Create a new C++ instance */
        Int data(t[0].cast<py::int_>());
        return data;
      }));
  (void)py::class_<UInt, Number, std::shared_ptr<UInt>>(m_sub, "UInt")
    .def(py::init())
    .def(py::init<int>(), py::arg("nbits"))
    .def(py::pickle(
      [](const UInt &t) {  // __getstate__
        /* Return a tuple that fully encodes the state of the object */
        return py::make_tuple(py::int_(t.nbits()));
      },
      [](const py::tuple &t) {  // __setstate__
        if (t.size() != 1) {
          throw std::runtime_error("Invalid state!");
        }
        /* Create a new C++ instance */
        UInt data(t[0].cast<py::int_>());
        return data;
      }));
  (void)py::class_<Float, Number, std::shared_ptr<Float>>(m_sub, "Float")
    .def(py::init())
    .def(py::init<int>(), py::arg("nbits"))
    .def(py::pickle(
      [](const Float &t) {  // __getstate__
        /* Return a tuple that fully encodes the state of the object */
        return py::make_tuple(py::int_(t.nbits()));
      },
      [](const py::tuple &t) {  // __setstate__
        if (t.size() != 1) {
          throw std::runtime_error("Invalid state!");
        }
        /* Create a new C++ instance */
        Float data(t[0].cast<py::int_>());
        return data;
      }));
  (void)py::class_<BFloat, Number, std::shared_ptr<BFloat>>(m_sub, "BFloat")
    .def(py::init())
    .def(py::init<int>(), py::arg("nbits"))
    .def(py::pickle(
      [](const BFloat &t) {  // __getstate__
        /* Return a tuple that fully encodes the state of the object */
        return py::make_tuple(py::int_(t.nbits()));
      },
      [](const py::tuple &t) {  // __setstate__
        if (t.size() != 1) {
          throw std::runtime_error("Invalid state!");
        }
        /* Create a new C++ instance */
        BFloat data(t[0].cast<py::int_>());
        return data;
      }));
  (void)py::class_<Complex, Number, std::shared_ptr<Complex>>(m_sub, "Complex")
    .def(py::init())
    .def(py::init<int>(), py::arg("nbits"))
    .def(py::pickle(
      [](const Complex &t) {  // __getstate__
        /* Return a tuple that fully encodes the state of the object */
        return py::make_tuple(py::int_(t.nbits()));
      },
      [](const py::tuple &t) {  // __setstate__
        if (t.size() != 1) {
          throw std::runtime_error("Invalid state!");
        }
        /* Create a new C++ instance */
        Complex data(t[0].cast<py::int_>());
        return data;
      }));
  (void)py::class_<List, Type, std::shared_ptr<List>>(m_sub, "List")
    .def(py::init())
    .def(py::init<std::vector<TypePtr>>(), py::arg("elements"));
  (void)py::class_<Tuple, Type, std::shared_ptr<Tuple>>(m_sub, "Tuple")
    .def(py::init())
    .def(py::init<std::vector<TypePtr>>(), py::arg("elements"));
  (void)py::class_<Dictionary, Type, std::shared_ptr<Dictionary>>(m_sub, "Dict")
    .def(py::init())
    .def(py::init<std::vector<std::pair<ValuePtr, TypePtr>>>(), py::arg("key_values"));
  (void)py::class_<TensorType, Type, std::shared_ptr<TensorType>>(m_sub, "TensorType")
    .def(py::init())
    .def(py::init<TypePtr>(), py::arg("element"))
    .def("element_type", &TensorType::element)
    .def(py::pickle(
      [](const TensorType &t) {  // __getstate__
        auto element_type = t.element();
        if (!element_type) {
          throw std::runtime_error("Can not serialize TensorType with null element_type.");
        }
        /* Return a tuple that fully encodes the state of the object */
        return py::make_tuple(py::int_(static_cast<int>(element_type->type_id())));
      },
      [](const py::tuple &t) {  // __setstate__
        if (t.size() != 1) {
          throw std::runtime_error("Invalid state!");
        }
        /* Create a new C++ instance */
        TensorType data(TypeIdToType(TypeId(static_cast<int>(t[0].cast<py::int_>()))));
        return data;
      }));
  (void)py::class_<RowTensorType, Type, std::shared_ptr<RowTensorType>>(m_sub, "RowTensorType")
    .def(py::init())
    .def_property_readonly("ElementType", &RowTensorType::element, "Get the RowTensorType's element type.");
  (void)py::class_<COOTensorType, Type, std::shared_ptr<COOTensorType>>(m_sub, "COOTensorType")
    .def(py::init())
    .def_property_readonly("ElementType", &COOTensorType::element_type, "Get the COOTensorType's element type.");
  (void)py::class_<CSRTensorType, Type, std::shared_ptr<CSRTensorType>>(m_sub, "CSRTensorType")
    .def(py::init())
    .def_property_readonly("ElementType", &CSRTensorType::element_type, "Get the CSRTensorType's element type.");
  (void)py::class_<UndeterminedType, Type, std::shared_ptr<UndeterminedType>>(m_sub, "UndeterminedType")
    .def(py::init());
  (void)py::class_<Function, Type, std::shared_ptr<Function>>(m_sub, "Function")
    .def(py::init())
    .def(py::init<std::vector<TypePtr>, TypePtr>(), py::arg("args"), py::arg("retval"));
  (void)py::class_<SymbolicKeyType, Type, std::shared_ptr<SymbolicKeyType>>(m_sub, "SymbolicKeyType").def(py::init());
  (void)py::class_<EnvType, Type, std::shared_ptr<EnvType>>(m_sub, "EnvType").def(py::init());
  (void)py::class_<TypeNone, Type, std::shared_ptr<TypeNone>>(m_sub, "TypeNone").def(py::init());
  (void)py::class_<TypeType, Type, std::shared_ptr<TypeType>>(m_sub, "TypeType").def(py::init());
  (void)py::class_<String, Type, std::shared_ptr<String>>(m_sub, "String").def(py::init());
  (void)py::class_<RefKeyType, Type, std::shared_ptr<RefKeyType>>(m_sub, "RefKeyType").def(py::init());
  (void)py::class_<RefType, TensorType, Type, std::shared_ptr<RefType>>(m_sub, "RefType").def(py::init());
  (void)py::class_<TypeAny, Type, std::shared_ptr<TypeAny>>(m_sub, "TypeAny").def(py::init());
  (void)py::class_<Slice, Type, std::shared_ptr<Slice>>(m_sub, "Slice").def(py::init());
  (void)py::class_<TypeEllipsis, Type, std::shared_ptr<TypeEllipsis>>(m_sub, "TypeEllipsis").def(py::init());
  (void)py::class_<MsClassType, Type, std::shared_ptr<MsClassType>>(m_sub, "TypeMsClassType").def(py::init());
  (void)py::class_<TypeNull, Type, std::shared_ptr<TypeNull>>(m_sub, "TypeNull").def(py::init());
  (void)py::class_<Keyword, Type, std::shared_ptr<Keyword>>(m_sub, "Keyword").def(py::init());
  m_sub.attr("kBool") = py::cast(kBool);
  m_sub.attr("kInt4") = py::cast(kInt4);
  m_sub.attr("kInt8") = py::cast(kInt8);
  m_sub.attr("kInt16") = py::cast(kInt16);
  m_sub.attr("kInt32") = py::cast(kInt32);
  m_sub.attr("kInt64") = py::cast(kInt64);
  m_sub.attr("kUInt8") = py::cast(kUInt8);
  m_sub.attr("kUInt16") = py::cast(kUInt16);
  m_sub.attr("kUInt32") = py::cast(kUInt32);
  m_sub.attr("kUInt64") = py::cast(kUInt64);
  m_sub.attr("kFloat16") = py::cast(kFloat16);
  m_sub.attr("kFloat32") = py::cast(kFloat32);
  m_sub.attr("kFloat64") = py::cast(kFloat64);
  m_sub.attr("kFloat8E4M3FN") = py::cast(kFloat8E4M3FN);
  m_sub.attr("kFloat8E5M2") = py::cast(kFloat8E5M2);
  m_sub.attr("kHiFloat8") = py::cast(kHiFloat8);
  m_sub.attr("kBFloat16") = py::cast(kBFloat16);
  m_sub.attr("kInt") = py::cast(kInt);
  m_sub.attr("kUInt") = py::cast(kUInt);
  m_sub.attr("kFloat") = py::cast(kFloat);
  m_sub.attr("kBFloat") = py::cast(kBFloat);
  m_sub.attr("kNumber") = py::cast(kNumber);
  m_sub.attr("kComplex64") = py::cast(kComplex64);
  m_sub.attr("kComplex128") = py::cast(kComplex128);
  m_sub.attr("kString") = py::cast(kString);
  m_sub.attr("kList") = py::cast(kList);
  m_sub.attr("kTuple") = py::cast(kTuple);
  m_sub.attr("kTypeNone") = py::cast(kTypeNone);
  m_sub.attr("kTypeNull") = py::cast(kTypeNull);
  m_sub.attr("kTensorType") = py::cast(kTensorType);
  m_sub.attr("kRowTensorType") = py::cast(kRowTensorType);
  m_sub.attr("kCOOTensorType") = py::cast(kCOOTensorType);
  m_sub.attr("kCSRTensorType") = py::cast(kCSRTensorType);
  m_sub.attr("kTypeEnv") = py::cast(kTypeEnv);
  m_sub.attr("kTypeType") = py::cast(kTypeType);
  m_sub.attr("kRefKeyType") = py::cast(kRefKeyType);
}
}  // namespace mindspore
