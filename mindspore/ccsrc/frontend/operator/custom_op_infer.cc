/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#if !defined(_WIN32) && !defined(_WIN64)
#include <dlfcn.h>
#endif

#include "include/utils/utils.h"
#include "include/frontend/jit/ps/static_analysis/py_infer_convert.h"
#include "primitive/framework_ops.h"
#include "include/frontend/operator/primitive_py.h"
#include "frontend/jit/ps/static_analysis/prim.h"
#include "frontend/jit/ps/static_analysis/prim_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "utils/file_utils.h"
#include "utils/custom_aot_extra.h"
#include "mindspore/ops/infer/custom.h"
#include "infer/ops_func_impl/custom_ext.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kCppInferShapeAttr = "cpp_infer_shape";
constexpr auto kCppInferTypeAttr = "cpp_infer_type";
constexpr auto kFuncName = "func_name";
constexpr auto kAOTFuncType = "aot";
constexpr auto kRegOpName = "reg_op_name";
constexpr auto kDTypeAttr = "dtype";
constexpr auto kMultiOutput = "multi_output";

#define RUN_INFER_FUNC(kernel_name, ...)                                                                               \
  do {                                                                                                                 \
    try {                                                                                                              \
      ret = infer_func(__VA_ARGS__);                                                                                   \
    } catch (const std::exception &e) {                                                                                \
      MS_LOG(EXCEPTION) << "For " << kernel_name << ", operator failed when executing user defined file " << file_path \
                        << "! Error message is " << e.what();                                                          \
    }                                                                                                                  \
  } while (false)

std::vector<AbstractBasePtr> GetInputAbstract(const std::string &op_name,
                                              const std::vector<AbstractBasePtr> &input_args) {
  if (input_args.size() != 1) {
    MS_LOG(INFO) << "Custom op [" << op_name << "] input args size is not equal 1, but is " << input_args.size();
  }
  auto input = input_args[0];
  if (!input->isa<abstract::AbstractTuple>()) {
    MS_LOG(EXCEPTION) << "Custom op [" << op_name
                      << "] input abstract type is not tuple, abstract: " << input->ToString();
  }
  auto tuple_input = input->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_input);
  return tuple_input->elements();
}

bool IsMultiOutput(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  return primitive->HasAttr(kMultiOutput) && GetValue<bool>(primitive->GetAttr(kMultiOutput));
}

#if !defined(_WIN32) && !defined(_WIN64)

void GetInferFilePathAndFuncName(const PrimitivePtr &primitive, std::string *file_path, std::string *func_name) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto func_type = GetValue<std::string>(primitive->GetAttr(kAttrFuncType));
  const auto &exec_info = GetValue<std::string>(primitive->GetAttr(kFuncName));
  if (func_type != kAOTFuncType) {
    MS_LOG(EXCEPTION) << "The custom operator of type '" << func_type
                      << "' does not support shape inference in cpp yet, func name:" << exec_info;
  }

  auto kernel_name = primitive->name();
  if (auto pos = exec_info.find(":"); pos != std::string::npos) {
    auto path = exec_info.substr(0, pos);
    auto real_path = FileUtils::GetRealPath(path.c_str());
    if (!real_path.has_value()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', couldn't find the AOT binary file under path: " << path;
    }
    *file_path = real_path.value();
    *func_name = exec_info.substr(pos + 1);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', user defined function path '" << exec_info << "' is illegal.";
  }
}

void PrepareInferShapeParams(const std::string &kernel_name, const std::vector<AbstractBasePtr> &input_args,
                             std::vector<int> *ndims, std::vector<std::vector<int64_t>> *shape_list) {
  MS_EXCEPTION_IF_NULL(ndims);
  MS_EXCEPTION_IF_NULL(shape_list);
  for (size_t idx = 0; idx < input_args.size(); idx++) {
    const auto &input_arg = input_args[idx];
    MS_EXCEPTION_IF_NULL(input_arg);
    MS_LOG(DEBUG) << "Custom op [" << kernel_name << "], input " << idx << " debug string: " << ToString(input_args);
    MS_VLOG(VL_CUSTOM_OP) << "Custom op [" << kernel_name << "], input " << idx
                          << " debug string: " << ToString(input_args);
    if (input_arg->isa<mindspore::abstract::AbstractSequence>()) {
      const auto &seq_input = input_arg->cast<mindspore::abstract::AbstractSequencePtr>();
      const auto &elements = seq_input->elements();
      for (size_t ele_idx = 0; ele_idx < elements.size(); ele_idx++) {
        if (elements[ele_idx]->GetType()->object_type() != kObjectTypeTensorType) {
          break;
        }
        auto params_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(kernel_name, elements, ele_idx);
        MS_EXCEPTION_IF_NULL(params_shape_ptr);
        const auto &params_shape = params_shape_ptr->shape();
        (void)ndims->emplace_back(SizeToInt(params_shape.size()));
        (void)shape_list->emplace_back(params_shape);
      }
    }
    if (input_arg->GetType()->object_type() != kObjectTypeTensorType) {
      continue;
    }
    auto params_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(kernel_name, input_args, idx);
    MS_EXCEPTION_IF_NULL(params_shape_ptr);
    auto params_shape = params_shape_ptr->shape();
    (void)ndims->emplace_back(SizeToInt(params_shape.size()));
    (void)shape_list->emplace_back(params_shape);
  }
}

void PrePareInferTypeParams(const std::string &kernel_name, const std::vector<AbstractBasePtr> &input_args,
                            std::vector<TypeId> *input_types) {
  MS_EXCEPTION_IF_NULL(input_types);
  for (size_t idx = 0; idx < input_args.size(); ++idx) {
    const auto &input_arg = input_args[idx];
    MS_EXCEPTION_IF_NULL(input_arg);
    MS_LOG(DEBUG) << "Custom op [" << kernel_name << "], input " << idx << " debug string: " << ToString(input_args);
    MS_VLOG(VL_CUSTOM_OP) << "Custom op [" << kernel_name << "], input " << idx
                          << " debug string: " << ToString(input_args);
    if (input_arg->isa<mindspore::abstract::AbstractSequence>()) {
      const auto &seq_input = input_arg->cast<mindspore::abstract::AbstractSequencePtr>();
      const auto &elements = seq_input->elements();
      for (size_t ele_idx = 0; ele_idx < elements.size(); ele_idx++) {
        if (elements[ele_idx]->GetType()->object_type() != kObjectTypeTensorType) {
          break;
        }
        auto type = CheckAndConvertUtils::GetTensorInputType(kernel_name, elements, ele_idx);
        MS_EXCEPTION_IF_NULL(type);
        (void)input_types->emplace_back(type->type_id());
      }
    }
    if (input_arg->GetType()->object_type() != kObjectTypeTensorType) {
      continue;
    }
    auto type = CheckAndConvertUtils::GetTensorInputType(kernel_name, input_args, idx);
    MS_EXCEPTION_IF_NULL(type);
    (void)input_types->emplace_back(type->type_id());
  }
}

BaseShapePtr CustomInferShapeByCPP(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto &kernel_name = GetValue<std::string>(primitive->GetAttr(kRegOpName));
  MS_LOG(DEBUG) << "Start infer type for " << kernel_name;
  MS_VLOG(VL_CUSTOM_OP) << "Start infer type for " << kernel_name;
  std::vector<int64_t *> input_shapes;
  std::vector<int> ndims;
  std::vector<std::vector<int64_t>> shape_list;
  BaseShapePtr res_shape = nullptr;
  PrepareInferShapeParams(kernel_name, input_args, &ndims, &shape_list);
  (void)std::transform(std::begin(shape_list), std::end(shape_list), std::back_inserter(input_shapes),
                       [](auto &v) { return &v[0]; });

  AotExtraImpl attrs;
  attrs.SetKernelPrim(primitive);

  std::string file_path;
  std::string func_name;
  GetInferFilePathAndFuncName(primitive, &file_path, &func_name);
  void *handle = dlopen(file_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', dlopen file under path" << file_path
                      << "throw the error: " << dlerror();
  }

  std::vector<abstract::BaseShapePtr> shape_vector;
  if (IsMultiOutput(primitive)) {
    auto infer_func =
      reinterpret_cast<std::add_pointer<std::vector<std::vector<int64_t>>(int *, int64_t **, AotExtra *)>::type>(
        dlsym(handle, (func_name + "InferShape").c_str()));
    if (infer_func == nullptr) {
      MS_LOG(EXCEPTION) << "Get infer shape functions failed. The custom operator does not support dynamic shape yet,"
                        << " func name:" << func_name
                        << ". Add the cpp version of the infer shape function to support dynamic shape.";
    }
    std::vector<std::vector<int64_t>> ret;
    RUN_INFER_FUNC(kernel_name, &ndims[0], &input_shapes[0], (&attrs));

    std::transform(ret.begin(), ret.end(), std::back_inserter(shape_vector),
                   [](const auto &item) { return std::make_shared<abstract::Shape>(item); });
    res_shape = std::make_shared<abstract::TupleShape>(shape_vector);
  } else {
    auto infer_func = reinterpret_cast<std::add_pointer<std::vector<int64_t>(int *, int64_t **, AotExtra *)>::type>(
      dlsym(handle, (func_name + "InferShape").c_str()));
    if (infer_func == nullptr) {
      MS_LOG(EXCEPTION) << "Get infer shape functions failed. The custom operator does not support dynamic shape yet,"
                        << " func name:" << func_name
                        << ". Add the cpp version of the infer shape function to support dynamic shape.";
    }
    std::vector<int64_t> ret;
    RUN_INFER_FUNC(kernel_name, &ndims[0], &input_shapes[0], (&attrs));
    res_shape = std::make_shared<abstract::Shape>(ret);
  }

  (void)dlclose(handle);
  attrs.DestructKernelData();
  return res_shape;
}

TypePtr CustomInferTypeByCPP(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto &kernel_name = GetValue<std::string>(primitive->GetAttr(kRegOpName));
  MS_LOG(DEBUG) << "Start infer type for " << kernel_name;
  MS_VLOG(VL_CUSTOM_OP) << "Start infer type for " << kernel_name;
  std::vector<TypeId> input_types;
  PrePareInferTypeParams(kernel_name, input_args, &input_types);

  AotExtraImpl attrs;
  attrs.SetKernelPrim(primitive);

  std::string file_path;
  std::string func_name;
  GetInferFilePathAndFuncName(primitive, &file_path, &func_name);
  void *handle = dlopen(file_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', dlopen file under path" << file_path
                      << "throw the error: " << dlerror();
  }
  TypePtr res_type = nullptr;
  if (IsMultiOutput(primitive)) {
    std::vector<TypePtr> dtype_vector;
    std::vector<TypeId> ret;
    auto infer_func = reinterpret_cast<std::add_pointer<std::vector<TypeId>(std::vector<TypeId>, AotExtra *)>::type>(
      dlsym(handle, (func_name + "InferType").c_str()));
    if (infer_func == nullptr) {
      MS_LOG(EXCEPTION) << "Get infer type functions failed. "
                        << " func name:" << func_name << ", Please add infer type function.";
    }

    RUN_INFER_FUNC(kernel_name, input_types, (&attrs));
    std::transform(ret.begin(), ret.end(), std::back_inserter(dtype_vector),
                   [](const auto &item) { return TypeIdToType(item); });
    res_type = std::make_shared<Tuple>(dtype_vector);
  } else {
    TypeId ret;
    auto infer_func = reinterpret_cast<std::add_pointer<TypeId(std::vector<TypeId>, AotExtra *)>::type>(
      dlsym(handle, (func_name + "InferType").c_str()));
    if (infer_func == nullptr) {
      MS_LOG(EXCEPTION) << "Get infer type functions failed. "
                        << " func name:" << func_name << ", Please add infer type function.";
    }
    RUN_INFER_FUNC(kernel_name, input_types, (&attrs));
    res_type = TypeIdToType(ret);
  }

  (void)dlclose(handle);
  attrs.DestructKernelData();
  return res_type;
}

BaseShapePtr CustomInferShapeByPy(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto &op_type = GetValue<std::string>(primitive->GetAttr(kRegOpName));
  MS_LOG(DEBUG) << "Start infer shape for " << op_type;
  MS_VLOG(VL_CUSTOM_OP) << "Start infer shape for " << op_type;
  if (!primitive->isa<PrimitivePy>()) {
    MS_LOG(EXCEPTION) << "The prim is not a PrimitivePy. Prim name: " << primitive->name();
  }
  // cppcheck-suppress unreadVariable
  py::gil_scoped_acquire acquire;
  auto prim_py = dyn_cast<PrimitivePy>(primitive);
  auto py_args = PreparePyInputs(input_args);
  auto out_dict = prim_py->RunInfer(py_args);
  if (IsMultiOutput(primitive)) {
    std::vector<abstract::BaseShapePtr> shape_vector;
    auto shape_obj = out_dict[ATTR_SHAPE];
    auto shape_tuple = shape_obj.cast<py::tuple>();
    for (auto &shape : shape_tuple) {
      (void)shape_vector.emplace_back(std::make_shared<abstract::Shape>(shape.cast<ShapeVector>()));
    }
    return std::make_shared<abstract::TupleShape>(shape_vector);
  } else {
    auto shape_obj = out_dict[ATTR_SHAPE];
    return std::make_shared<abstract::Shape>(shape_obj.cast<ShapeVector>());
  }
}

TypePtr CustomInferTypeByPy(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto &op_type = GetValue<std::string>(primitive->GetAttr(kRegOpName));
  MS_LOG(DEBUG) << "Start infer type for " << op_type;
  MS_VLOG(VL_CUSTOM_OP) << "Start infer type for " << op_type;
  if (!primitive->isa<PrimitivePy>()) {
    MS_LOG(EXCEPTION) << "The prim is not a PrimitivePy. Prim name: " << primitive->name();
  }
  // cppcheck-suppress unreadVariable
  py::gil_scoped_acquire acquire;
  auto prim_py = dyn_cast<PrimitivePy>(primitive);
  auto py_args = PreparePyInputs(input_args);
  auto output = prim_py->RunInfer(py_args);
  auto type_obj = output[ATTR_DTYPE];
  if (IsMultiOutput(primitive)) {
    std::vector<TypePtr> dtype_vector;
    auto py_tuple_dtype = type_obj.cast<py::tuple>();
    for (auto &item : py_tuple_dtype) {
      (void)dtype_vector.emplace_back(item.cast<TypePtr>());
    }
    return std::make_shared<Tuple>(dtype_vector);
  } else {
    auto res_dtype = type_obj.cast<TypePtr>();
    if (res_dtype == nullptr) {
      MS_LOG(EXCEPTION) << "The infer dtype should be a type object, but get infer dtype in: " << output;
    }
    return res_dtype;
  }
}

#endif

BaseShapePtr CustomInferShapeImpl(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
#if !defined(_WIN32) && !defined(_WIN64)
  if (primitive->HasAttr(kCppInferShapeAttr)) {
    return CustomInferShapeByCPP(primitive, input_args);
  } else {
    return CustomInferShapeByPy(primitive, input_args);
  }
#else
  MS_LOG(EXCEPTION) << "Custom Operators of type AOT doesn't support Windows currently";
  return mindspore::abstract::kNoShape;
#endif
}

TypePtr CustomInferTypeImpl(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
#if !defined(_WIN32) && !defined(_WIN64)
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->HasAttr(kCppInferTypeAttr)) {
    return CustomInferTypeByCPP(primitive, input_args);
  } else {
    return CustomInferTypeByPy(primitive, input_args);
  }
#else
  MS_LOG(EXCEPTION) << "Custom Operators of type AOT doesn't support Windows currently";
#endif
}

}  // namespace

#define REGISTER_PRIMITIVE_OP_CPP_INFER_IMPL(name, primitive, OP_INFER_ClASS, is_impl_infer_value) \
  const auto helper_op_infer_##name = abstract::RegisterStandardPrimitiveEvalHelper(               \
    abstract::GetPrimitiveInferMapPtr(), primitive, std::make_shared<OP_INFER_ClASS>(), is_impl_infer_value)

class AGCustomInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CustomInferShapeImpl(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return CustomInferTypeImpl(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr & /* engine */, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    if (!primitive->HasAttr(kCppInferShapeAttr) && !primitive->HasAttr(kCppInferTypeAttr)) {
      if (!primitive->isa<PrimitivePy>()) {
        MS_LOG(EXCEPTION) << "The prim is not a PrimitivePy. Prim name: " << primitive->name();
      }
      // cppcheck-suppress unreadVariable
      py::gil_scoped_acquire acquire;
      auto prim_py = dyn_cast<PrimitivePy>(primitive);
      auto py_args = PreparePyInputs(input_args);
      auto output = prim_py->RunInfer(py_args);
      return abstract::PyInferRes2Abstract(prim_py, output);
    }
    auto shape = InferShape(primitive, input_args);
    auto type = InferType(primitive, input_args);
    auto res = MakeAbstract(shape, type);
    return res;
  }
};

// infet shape for pyboost
BaseShapePtr CustomFuncImplInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto &op_type = GetValue<std::string>(primitive->GetAttr(kRegOpName));
  const auto &inputs = GetInputAbstract(op_type, input_args);
  auto infer_shape = CustomInferShapeImpl(primitive, inputs);
  if (!IsMultiOutput(primitive)) {
    std::vector<abstract::BaseShapePtr> shape_vector{infer_shape};
    return std::make_shared<abstract::TupleShape>(shape_vector);
  }
  return infer_shape;
}

// infer type for pyboost
TypePtr CustomFuncImplInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto &op_type = GetValue<std::string>(primitive->GetAttr(kRegOpName));
  const auto &inputs = GetInputAbstract(op_type, input_args);
  auto infer_type = CustomInferTypeImpl(primitive, inputs);
  if (!IsMultiOutput(primitive)) {
    std::vector<TypePtr> type_vector{infer_type};
    return std::make_shared<Tuple>(type_vector);
  }
  return infer_type;
}

struct CustomRegFunc {
  CustomRegFunc() noexcept {
    CustomExtFuncImpl::set_infer_shape_call_func(CustomFuncImplInferShape);
    CustomExtFuncImpl::set_infer_type_call_func(CustomFuncImplInferType);
  }
} g_reg_func;

REGISTER_PRIMITIVE_OP_CPP_INFER_IMPL(Custom, prim::kPrimCustom, AGCustomInfer, false);
}  // namespace ops
}  // namespace mindspore
