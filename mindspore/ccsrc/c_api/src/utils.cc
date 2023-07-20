/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "c_api/src/utils.h"
#include "c_api/src/helper.h"
#include "frontend/operator/ops_front_infer_function.h"
#include "backend/operator/ops_backend_infer_function.h"

void ConvertConstScalarInputToTensor(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (!input_node->isa<ValueNodeImpl>()) {
    return;
  }
  auto value_node = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<ScalarImpl>()) {
    return;
  }
  TensorPtr tensor_ptr = ScalarToTensor(value->cast<ScalarPtr>());
  if (tensor_ptr == nullptr) {
    MS_LOG(WARNING) << "Create tensor of" << input_node->DebugString() << "failed";
    return;
  }
  value_node->set_value(tensor_ptr);
  value_node->set_abstract(tensor_ptr->ToAbstract());
}

std::vector<TensorPtr> ConvertOutputToTensor(const mindspore::BaseRef &output) {
  std::vector<TensorPtr> ref_outputs{};
  if (mindspore::utils::isa<mindspore::VectorRef>(output)) {
    auto vec_ref = mindspore::utils::cast<mindspore::VectorRef>(output);
    for (const auto &item : vec_ref) {
      // for multiple outputs, ascend will return a VectorRef of VectorRef.
      const std::vector<TensorPtr> &item_out = ConvertOutputToTensor(item);
      (void)ref_outputs.insert(ref_outputs.end(), item_out.begin(), item_out.end());
    }
  } else if (mindspore::utils::isa<TensorPtr>(output)) {
    auto tensor = std::dynamic_pointer_cast<TensorImpl>(output.copy());
    tensor->data_sync();
    ref_outputs.push_back(tensor);
  } else if (mindspore::utils::isa<ScalarPtr>(output)) {
    auto value = mindspore::utils::cast<ScalarPtr>(output);
    auto tensor = ScalarToTensor(value->cast<ScalarPtr>());
    ref_outputs.push_back(tensor);
  } else {
    MS_LOG(ERROR) << "Convert output to tensor failed, unrecognized output type: " << output.ToString();
  }
  return ref_outputs;
}

STATUS OpSetAttrs(ResMgrHandle res_mgr, const PrimitivePtr &prim, const char *const *attr_names, ValueHandle attrs[],
                  size_t attr_num) {
  AttrMap attr_map{};
  for (size_t i = 0; i < attr_num; ++i) {
    if (attr_names[i] == nullptr) {
      MS_LOG(ERROR) << "Input array [attr_names] has nullptr element, index: " << i;
      return RET_NULL_PTR;
    }
    auto value = GetSrcPtr<ValuePtr>(res_mgr, attrs[i]);
    if (value == nullptr) {
      MS_LOG(ERROR) << "Get attribute's source pointer failed, attribute index: " << i;
      return RET_NULL_PTR;
    }
    std::string name(attr_names[i]);
    auto iter = kOpAttrNameAdaptMap.find(name);
    if (iter != kOpAttrNameAdaptMap.end()) {
      attr_map[iter->second] = value;
    }
    attr_map[name] = value;
  }
  (void)prim->SetAttrs(attr_map);
  return RET_OK;
}

std::vector<TensorPtr> ConvertOutputToTensor(const ValuePtr &output) {
  std::vector<TensorPtr> tensor_outputs{};
  if (output->isa<ValueSequenceImpl>()) {
    auto value_sequeue = output->cast<ValueSequencePtr>();
    for (const auto &item : value_sequeue->value()) {
      // for multiple outputs, ascend will return a tuple of ValuePtr.
      const std::vector<TensorPtr> &item_out = ConvertOutputToTensor(item);
      (void)tensor_outputs.insert(tensor_outputs.end(), item_out.begin(), item_out.end());
    }
  } else if (output->isa<TensorImpl>()) {
    auto tensor = output->cast<TensorPtr>();
    tensor->data_sync();
    tensor_outputs.push_back(tensor);
  } else if (output->isa<ScalarImpl>()) {
    auto tensor = ScalarToTensor(output->cast<ScalarPtr>());
    tensor_outputs.push_back(tensor);
  } else {
    MS_LOG(ERROR) << "Convert output to tensor failed, unrecognized output type: " << output->type_name();
  }
  return tensor_outputs;
}

std::vector<BaseShapePtr> BuildShape(int64_t **out_shapes, size_t *out_dims, size_t out_num) {
  MS_EXCEPTION_IF_NULL(out_shapes);
  MS_EXCEPTION_IF_NULL(out_dims);
  std::vector<BaseShapePtr> shape_list;
  if (out_num == 1) {
    int64_t *shape = out_shapes[0];
    ShapeVector shape_vec(shape, shape + out_dims[0]);
    auto infer_shape = std::make_shared<Shape>(shape_vec);
    (void)shape_list.emplace_back(infer_shape);
  } else {
    for (size_t i = 0; i < out_num; i++) {
      int64_t *shape = out_shapes[i];
      ShapeVector shape_vec(shape, shape + out_dims[i]);
      auto each_shape = std::make_shared<Shape>(shape_vec);
      (void)shape_list.emplace_back(each_shape);
    }
  }
  return shape_list;
}

std::vector<TypePtr> BuildType(const DataTypeC *out_dtypes, size_t out_num) {
  MS_EXCEPTION_IF_NULL(out_dtypes);
  std::vector<TypePtr> type_list;
  if (out_num == 1) {
    DataTypeC dtype = out_dtypes[0];
    auto cxx_type = mindspore::TypeId(dtype);
    auto infer_type = mindspore::TypeIdToType(cxx_type);
    (void)type_list.emplace_back(infer_type);
  } else {
    for (size_t i = 0; i < out_num; i++) {
      DataTypeC dtype = out_dtypes[i];
      auto cxx_type = mindspore::TypeId(dtype);
      auto type_val = mindspore::TypeIdToType(cxx_type);
      (void)type_list.emplace_back(type_val);
    }
  }
  return type_list;
}

AbstractBasePtr BuildAbstract(std::vector<BaseShapePtr> shapes, std::vector<TypePtr> types) {
  MS_EXCEPTION_IF_CHECK_FAIL(!shapes.empty(), "The size of shapes is empty!");
  MS_EXCEPTION_IF_CHECK_FAIL(!types.empty(), "The size of types is empty!");
  MS_EXCEPTION_IF_CHECK_FAIL(shapes.size() == types.size(), "The size of shapes and types must be equal!");
  if (shapes.size() == 1) {
    auto shape = shapes[0]->cast<ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape);
    auto type = types[0];
    MS_EXCEPTION_IF_NULL(type);
    auto shape_vec = shape->shape();
    // if the size of shape list is empty, return an scalar abstract
    if (shape_vec.empty() && (!type->isa<mindspore::TensorType>())) {
      AbstractScalarPtr abs_scalar = std::make_shared<AbstractScalarImpl>(mindspore::kValueAny, type);
      return abs_scalar;
    }
    return MakeAbstractTensor(shape, type);
  } else {
    mindspore::abstract::AbstractBasePtrList ptr_list;
    for (size_t i = 0; i < shapes.size(); ++i) {
      auto shape = shapes[i];
      MS_EXCEPTION_IF_NULL(shape);
      auto type = types[i];
      MS_EXCEPTION_IF_NULL(type);
      auto tensor_abs = BuildAbstract({shape}, {type});
      (void)ptr_list.emplace_back(tensor_abs);
    }
    return std::make_shared<AbstractTupleImpl>(ptr_list);
  }
}

AbstractBasePtr GetAbstract(const TypePtr &type_ptr, const int64_t shape[], size_t shape_size, bool is_param) {
  if (shape == nullptr) {
    if (shape_size == 0) {
      if (is_param) {
        ShapeVector shape_vec{};
        return std::make_shared<AbstractTensorImpl>(type_ptr, shape_vec);
      }
      return std::make_shared<AbstractScalarImpl>(type_ptr);
    } else {
      MS_LOG(ERROR) << "Input Handle [shape_size] should >= 0.";
      return nullptr;
    }
  }
  if (shape[0] == 0 && shape_size == 1) {
    ShapeVector shape_vec{};
    return std::make_shared<AbstractTensorImpl>(type_ptr, shape_vec);
  }
  ShapeVector shape_vec(shape, shape + shape_size);
  return std::make_shared<AbstractTensorImpl>(type_ptr, shape_vec);
}

AbstractBasePtr OpInferShapeAndType(const PrimitivePtr &prim, const mindspore::AbstractBasePtrList &args_abs_list) {
  MS_EXCEPTION_IF_NULL(prim);
  auto front_eval_impl = mindspore::abstract::GetFrontendPrimitiveInferImpl(prim);
  if (front_eval_impl.has_value()) {
    auto infer = front_eval_impl.value();
    MS_EXCEPTION_IF_CHECK_FAIL(infer.IsImplInferShapeAndType(), "There is no infer-abstract implement!");
    auto abs = infer.InferShapeAndType(nullptr, prim, args_abs_list);
    return abs;
  }
  auto back_eval_impl = mindspore::abstract::GetBackendPrimitiveInferImpl(prim);
  if (back_eval_impl.has_value()) {
    auto infer = back_eval_impl.value();
    MS_EXCEPTION_IF_CHECK_FAIL(infer.IsImplInferShapeAndType(), "There is no infer-abstract implement!");
    auto abs = infer.InferShapeAndType(nullptr, prim, args_abs_list);
    return abs;
  }
  MS_LOG(EXCEPTION) << "Get infer function failed, the operator has not infer shape of infer type function yet, "
                       "primitive name:"
                    << prim->name() << " primitive type:" << prim->type_name();
}

STATUS CheckCustomOpInfo(const CustomOpInfo &info) {
  MS_ERROR_IF_FALSE_W_RET_N_LOG(info.func_name != nullptr, RET_ERROR, "The func_name of custom op must be specified!");
  MS_ERROR_IF_FALSE_W_RET_N_LOG(info.func_type != nullptr, RET_ERROR, "The func_type of custom op must be specified!");
  MS_ERROR_IF_FALSE_W_RET_N_LOG(info.target != nullptr, RET_ERROR, "The target of custom op must be specified!");
  MS_ERROR_IF_FALSE_W_RET_N_LOG(info.input_names != nullptr, RET_ERROR,
                                "The input_names of custom op must be specified!");
  MS_ERROR_IF_FALSE_W_RET_N_LOG(info.output_names != nullptr, RET_ERROR,
                                "The output_names of custom op must be specified!");
  MS_ERROR_IF_FALSE_W_RET_N_LOG(info.input_num > 0, RET_ERROR, "The input_num of custom op must be a positive value!");
  MS_ERROR_IF_FALSE_W_RET_N_LOG(info.output_num > 0, RET_ERROR,
                                "The output_num of custom op must be a positive value!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.attr_num < 0, RET_ERROR, "The attr_num of custom op must be non-negative!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.dtype_infer_func == nullptr && info.output_dtypes == nullptr, RET_ERROR,
                               "Either dtype infer function or output shape must be specified!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.dtype_infer_func != nullptr && info.output_dtypes != nullptr, RET_ERROR,
                               "Only one should be specified between dtype infer function and output shape!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.shape_infer_func == nullptr && info.output_shapes == nullptr, RET_ERROR,
                               "Either shape infer function or output shape must be specified!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.shape_infer_func != nullptr && info.output_shapes != nullptr, RET_ERROR,
                               "Only one should be specified between shape infer function and output shape!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.output_shapes != nullptr && info.output_dims == nullptr, RET_ERROR,
                               "Output dims must be specified if output_shapes are given!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.attr_num == 0 && (info.attr_names != nullptr || info.attr_values != nullptr),
                               RET_ERROR, "The attr_name and attr_values must be nullptr if attr_num is 0!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.attr_num != 0 && (info.attr_names == nullptr || info.attr_values == nullptr),
                               RET_ERROR, "The attr_name and attr_values must be specified if attr_num is non-zero!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.dtype_formats != nullptr && info.dtype_formats_num == 0, RET_ERROR,
                               "The dtype_formats_num of custom op must be none-zero if dtype_formats is specified!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.dtype_formats == nullptr && info.dtype_formats_num != 0, RET_ERROR,
                               "The dtype_formats_num of custom op must be zero if dtype_formats is not specified!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(std::string(info.func_name).find(".so:") == std::string::npos, RET_ERROR,
                               "so file path and function name must be provided in func_name!");
  return RET_OK;
}

nlohmann::json ConvertOpInfoToJson(const CustomOpInfo &info) {
  nlohmann::json obj;
  obj["attr"] = {};
  std::string target = info.target;
  obj["target"] = target;
  obj["op_name"] = "Custom" + std::string(info.func_name);
  obj["fusion_tyoe"] = "OPAQUE";
  if (info.dtype_formats != nullptr) {
    std::vector<std::vector<std::string>> dtype_formats;
    for (size_t i = 0; i < info.dtype_formats_num; i++) {
      for (size_t j = 0; j < info.input_num + info.output_num; j++) {
        auto iter = kDTypeFmtEnumToStrMap.find(info.dtype_formats[i][j]);
        if (iter == kDTypeFmtEnumToStrMap.end()) {
          MS_LOG(ERROR) << "Unsupported DTypeFormat: " << info.dtype_formats[i][j];
          return {};
        }
        dtype_formats.push_back(iter->second);
      }
    }
    obj["dtype_format"] = {dtype_formats};
  }
  std::vector<nlohmann::json> js_inputs;
  for (size_t i = 0; i < info.input_num; i++) {
    nlohmann::json js_input;
    js_input["index"] = i;
    js_input["name"] = std::string(info.input_names[i]);
    js_input["paramType"] = "required";
    js_inputs.push_back(js_input);
  }
  obj["inputs"] = js_inputs;
  std::vector<nlohmann::json> js_outputs;
  for (size_t i = 0; i < info.output_num; i++) {
    nlohmann::json js_output;
    js_output["index"] = i;
    js_output["name"] = std::string(info.output_names[i]);
    js_output["paramType"] = "required";
    js_outputs.push_back(js_output);
  }
  obj["outputs"] = js_outputs;
  auto aot_imply_type = target == "Ascend" ? "BiSheng" : target;
  const std::map<std::string, std::string> func_type_to_imply_type = {
    {"hybrid", "AKG"},  {"akg", "AKG"},    {"tbe", "TBE"},         {"aicpu", "AICPU"},
    {"pyfunc", target}, {"julia", target}, {"aot", aot_imply_type}};
  auto iter = func_type_to_imply_type.find(std::string(info.func_type));
  if (iter == func_type_to_imply_type.end()) {
    MS_LOG(ERROR) << "Unsupported function type: " << std::string(info.func_type);
    return {};
  }
  auto imply_type = iter->second;
  obj["imply_type"] = imply_type;
  return obj;
}

size_t GetMaxMallocSize() {
  size_t max_malloc_size = 0;
#if defined(_MSC_VER) || defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  max_malloc_size = static_cast<size_t>(status.ullTotalPhys);
#else
  max_malloc_size = static_cast<size_t>(sysconf(_SC_PHYS_PAGES)) * static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
  return max_malloc_size;
}
