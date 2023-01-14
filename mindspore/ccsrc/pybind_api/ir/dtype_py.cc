/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
// Define python wrapper to handle data types.
void RegTyping(py::module *m) {
  auto m_sub = m->def_submodule("typing", "submodule for dtype");
  py::enum_<TypeId>(m_sub, "TypeId");
  (void)m_sub.def("is_subclass", &IsIdentidityOrSubclass, "is equal or subclass");
  (void)m_sub.def("load_type", &TypeIdToType, "load type");
  (void)m_sub.def(
    "dump_type", [](const TypePtr &t) { return t->type_id(); }, "dump type");
  (void)m_sub.def("str_to_type", &StringToType, "string to typeptr");
  (void)m_sub.def("type_size_in_bytes", &GetTypeByte, "type size in bytes");
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
  (void)py::class_<Number, Type, std::shared_ptr<Number>>(m_sub, "Number").def(py::init());
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
        /* Return a tuple that fully encodes the state of the object */
        return py::make_tuple(py::int_(static_cast<int>(t.element()->type_id())));
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
  (void)py::class_<TypeAnything, Type, std::shared_ptr<TypeAnything>>(m_sub, "TypeAnything").def(py::init());
  (void)py::class_<Slice, Type, std::shared_ptr<Slice>>(m_sub, "Slice").def(py::init());
  (void)py::class_<TypeEllipsis, Type, std::shared_ptr<TypeEllipsis>>(m_sub, "TypeEllipsis").def(py::init());
  (void)py::class_<MsClassType, Type, std::shared_ptr<MsClassType>>(m_sub, "TypeMsClassType").def(py::init());
  (void)py::class_<TypeNull, Type, std::shared_ptr<TypeNull>>(m_sub, "TypeNull").def(py::init());
}
}  // namespace mindspore
