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

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include <unistd.h>
#endif
#include <utility>
#include <vector>
#include <Python.h>

#include "pybind_api/pynative/tensor/storage/storage_py.h"
#include "include/utils/exception.h"
#include "pybind_api/pynative/pynative_api.h"
#include "include/utils/pynative/storage_py.h"
#include "device_address/device_address_maker.h"
#include "utils/ms_utils_secure.h"
#include "include/utils/pyobj_manager.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

namespace py = pybind11;

namespace mindspore {
static Py_ssize_t StoragePy_Length(StoragePy *self) {
  HANDLE_MS_EXCEPTION
  return static_cast<Py_ssize_t>(StoragePy_Unpack(self).NBytes());
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static PyObject *StoragePy_Getitem(StoragePy *self, PyObject *index) {
  HANDLE_MS_EXCEPTION
  MS_LOG(EXCEPTION) << "The function __getitem__ is not implemented for Storage!";
  HANDLE_MS_EXCEPTION_END
}

static int StoragePy_Setitem(StoragePy *self, PyObject *index, PyObject *value) {
  HANDLE_MS_EXCEPTION
  MS_LOG(EXCEPTION) << "The function __setitem__ is not implemented for Storage!";
  HANDLE_MS_EXCEPTION_RET_FAIL_END
}

static void StoragePy_PyDealloc(PyObject *obj) {
  const auto &storage = StoragePy_Unpack(obj);
  storage.~Storage();
  Py_TYPE(obj)->tp_free(obj);
}

static PyMappingMethods StoragePy_mappingmethods = {(lenfunc)StoragePy_Length, (binaryfunc)StoragePy_Getitem,
                                                    (objobjargproc)StoragePy_Setitem};

PyTypeObject StoragePyType = {
  PyVarObject_HEAD_INIT(NULL, 0) "StoragePy", /* tp_name */
  sizeof(StoragePy),                          /* tp_basicsize */
  0,                                          /* tp_itemsize */
  StoragePy_PyDealloc,                        /* tp_dealloc */
  0,                                          /* tp_vectorcall_offset */
  nullptr,                                    /* tp_getattr */
  nullptr,                                    /* tp_setattr */
  nullptr,                                    /* tp_reserved */
  nullptr,                                    /* tp_repr */
  nullptr,                                    /* tp_as_number */
  nullptr,                                    /* tp_as_sequence */
  &StoragePy_mappingmethods,                  /* tp_as_mapping */
  nullptr,                                    /* tp_hash  */
  nullptr,                                    /* tp_call */
  nullptr,                                    /* tp_str */
  nullptr,                                    /* tp_getattro */
  nullptr,                                    /* tp_setattro */
  nullptr,                                    /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
  nullptr,                                    /* tp_doc */
  nullptr,                                    /* tp_traverse */
  nullptr,                                    /* tp_clear */
  nullptr,                                    /* tp_richcompare */
  0,                                          /* tp_weaklistoffset */
  nullptr,                                    /* tp_iter */
  nullptr,                                    /* tp_iternext */
  nullptr,
  /* will be assigned in init */ /* tp_methods */
  nullptr,
  /* will be assigned in init */ /* tp_members */
  nullptr,                       /* tp_getset */
  nullptr,                       /* tp_base */
  nullptr,                       /* tp_dict */
  nullptr,                       /* tp_descr_get */
  nullptr,                       /* tp_descr_set */
  0,                             /* tp_dictoffset */
  nullptr,                       /* tp_init */
  nullptr,                       /* tp_alloc */
  nullptr,                       /* tp_new */
};

void StoragePy_assertNotNull(StoragePy *storage) {
  if (StoragePy_Unpack(storage).get_storage_base() == nullptr) {
    MS_LOG(EXCEPTION) << "Got a null Storage";
  }
}

void StoragePy_assertNotNull(PyObject *obj) { StoragePy_assertNotNull(reinterpret_cast<StoragePy *>(obj)); }

static PyObject *StoragePy_Copy_(PyObject *self, PyObject *args, PyObject *kwargs) {
  HANDLE_MS_EXCEPTION
  StoragePy_assertNotNull(self);
  auto self_ = StoragePy_Unpack(self);
  PyObject *py_type;
  static const char *kwlist[] = {"src", "non_blocking", nullptr};
  int non_blocking = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", const_cast<char **>(kwlist), &py_type, &non_blocking)) {
    return nullptr;
  }

  auto src = StoragePy_Unpack(py_type);
  if (self_.NBytes() != src.NBytes()) {
    MS_LOG(EXCEPTION) << "Size does not match, self was " << self_.NBytes() << " bytes but src was " << src.NBytes()
                      << " bytes";
  }

  // ToDo: use tensor.copy_ to implement storage copy_
  self_.InplaceCopy(src, non_blocking);
  Py_INCREF(self);
  return self;

  HANDLE_MS_EXCEPTION_END
}

static PyObject *StoragePy_ElementSize(PyObject *self, PyObject *noargs) {
  HANDLE_MS_EXCEPTION
  StoragePy_assertNotNull(self);
  return PyLong_FromLongLong(sizeof(uint8_t));
  HANDLE_MS_EXCEPTION_END
}

static PyObject *StoragePy_Resize_(PyObject *self, PyObject *number_arg) {
  HANDLE_MS_EXCEPTION
  StoragePy_assertNotNull(self);
  const auto &storage = StoragePy_Unpack(self);

  if (!(PyLong_CheckExact(number_arg) && !PyBool_Check(number_arg))) {
    MS_LOG(EXCEPTION) << "resize_ expects an int, but got " << Py_TYPE(number_arg)->tp_name;
  }

  int64_t newsize = PyLong_AsLong(number_arg);
  auto device_type = storage.device();
  if (device_type == "Ascend" || device_type == "CPU") {
    storage.get_mutable_storage_base()->InplaceReSize(newsize);
  } else if (device_type == "GPU") {
    MS_LOG(EXCEPTION) << "Current Storage only support NPU, but got GPU!";
  }
  Py_INCREF(self);
  return self;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *StoragePy_NBytes(PyObject *self, PyObject *noargs) {
  HANDLE_MS_EXCEPTION
  StoragePy_assertNotNull(self);
  return py::cast(StoragePy_Unpack(self).NBytes()).release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *StoragePy_DataPtr(PyObject *self, PyObject *noargs) {
  HANDLE_MS_EXCEPTION
  auto self_ = StoragePy_Unpack(self);
  PyObject *dataPtrResult = PyLong_FromVoidPtr(reinterpret_cast<void *>(self_.DataPtr()));
  return dataPtrResult;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *StoragePy_shareFd(PyObject *self, PyObject *noargs) {
  HANDLE_MS_EXCEPTION
  StoragePy_assertNotNull(self);
  auto self_ = StoragePy_Unpack(self);
  auto device_type = self_.device();
  if (device_type != "CPU") {
    MS_LOG(EXCEPTION) << "_share_fd_: only available on CPU.";
  }
  // Do sync here in case that the sender process launch kernels and put the tensor in queue,
  // then the receiver process get the tensor with wrong data from queue because the kernels have not fineshed.
  runtime::Pipeline::Get().WaitForward();
  device::DeviceContextManager::GetInstance().SyncAllStreams();
  int64_t nbytes = self_.NBytes();
  TypeId type_id = self_.GetTypeId();
  ShapeVector shape_vector;

  if (self_.GetMapAllocator()) {
    // pass
    MS_LOG(INFO) << "StoragePy_shareFd map_allocator exist.";
  } else {
    std::string filename = NewShareMemoryHandle();
    auto map_allocator_new = std::make_unique<MapAllocator>(filename, true, -1, nbytes);
    void *base_ptr_ = map_allocator_new->Alloc(nbytes);
    MS_LOG(INFO) << "StoragePy_shareFd type_id:" << type_id << ", nbytes:" << nbytes << ", filename:" << filename
                 << ", original DataPtr:" << self_.DataPtr() << ", map_allocator ptr:" << base_ptr_;
    {
      // copy data to share memory may take long time, so release gil
      pybind11::gil_scoped_release release_gil;
      // H2H copy
      auto memcpy_ret = common::huge_memcpy(reinterpret_cast<uint8_t *>(base_ptr_), static_cast<size_t>(nbytes),
                                            reinterpret_cast<uint8_t *>(self_.DataPtr()), static_cast<size_t>(nbytes));
      if (memcpy_ret != EOK) {
        MS_LOG(EXCEPTION) << "memcpy failed!";
      }
    }
    auto device_address = DeviceAddressMaker(base_ptr_, type_id, shape_vector)
                            .set_maker(GetDeviceAddressMaker(device::DeviceType::kCPU))
                            .make_device_address();
    std::dynamic_pointer_cast<device::DeviceAddress>(device_address)->set_map_allocator(std::move(map_allocator_new));
    self_.SetDevicePointer(std::dynamic_pointer_cast<device::DeviceAddress>(device_address)->device_pointer());
  }
  PyObject *tuple_ret = PyTuple_New(3);
  PyTuple_SetItem(tuple_ret, 0, PyLong_FromLong(self_.GetMapAllocator()->fd()));
  PyTuple_SetItem(tuple_ret, 1, PyLong_FromLongLong(nbytes));
  PyTuple_SetItem(tuple_ret, 2, PyLong_FromLong(type_id));

  return tuple_ret;
  HANDLE_MS_EXCEPTION_END
}

static PyObject *StoragePy_newSharedFd(PyObject *_unused, PyObject *args) {
  HANDLE_MS_EXCEPTION

  Py_ssize_t args_element_count = 3;
  if (PyTuple_GET_SIZE(args) != args_element_count) {
    MS_LOG(EXCEPTION) << "The args are invalid, expect 3 items in the tuple.";
  }
  PyObject *_tmp_fd = PyTuple_GET_ITEM(args, 0);
  PyObject *_size = PyTuple_GET_ITEM(args, 1);
  PyObject *_type_id = PyTuple_GET_ITEM(args, 2);

  int tmp_fd = (int)PyLong_AsLongLong(_tmp_fd);
  int64_t size = (int64_t)PyLong_AsLongLong(_size);
  TypeId type_id = TypeId((int)PyLong_AsLong(_type_id));
  size_t element_size = UnitSizeInBytes(type_id);
  if (element_size == 0) {
    MS_LOG(EXCEPTION) << "element_size can't be zero. TypeId:" << type_id;
    return nullptr;
  }
  int64_t element_count = static_cast<int64_t>(size / static_cast<int64_t>(element_size));
  ShapeVector shape_vec;
  shape_vec.push_back(element_count);
  MS_LOG(INFO) << "StoragePy_newSharedFd tmp_fd:" << tmp_fd << ", size:" << size << ", type_id:" << type_id
               << ", shape_vec:" << shape_vec;

  // duplicate the file descriptor, this is defensive operation to avoid that the original fd is released elsewhere
  int fd = dup(tmp_fd);
  if (fd == -1) {
    MS_LOG(EXCEPTION) << "could not duplicate a shared memory file descriptor.";
    return nullptr;
  }

  auto map_allocator = std::make_unique<MapAllocator>("", false, fd, size);
  void *base_ptr_ = map_allocator->Alloc(size);
  auto device_address = DeviceAddressMaker(base_ptr_, type_id, shape_vec)
                          .set_maker(GetDeviceAddressMaker(device::DeviceType::kCPU))
                          .make_device_address();
  std::dynamic_pointer_cast<device::DeviceAddress>(device_address)->set_map_allocator(std::move(map_allocator));
  auto storage_base = std::make_shared<StorageBase>(device_address, type_id);
  Storage storage = Storage(storage_base);

  return CreateStoragePyObj(storage);
  HANDLE_MS_EXCEPTION_END
}

static PyObject *StoragePy_isShared(PyObject *self, PyObject *noargs) {
  HANDLE_MS_EXCEPTION
  StoragePy_assertNotNull(self);
  auto self_ = StoragePy_Unpack(self);
  auto device_type = self_.device();
  if (device_type == "Ascend") {
    MS_LOG(WARNING) << "Storage.is_shared() will always return true for NPU tensor. However, reduce_tensor method "
                       "in mindspore.multiprocessing module does not support NPU tensor currently, so inter-process "
                       "communication(IPC) is not available for NPU tensor.";
    Py_RETURN_TRUE;
  }
  if (self_.GetMapAllocator()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  HANDLE_MS_EXCEPTION_END
}

static PyObject *StoragePy_device(PyObject *self, PyObject *_unused) {
  HANDLE_MS_EXCEPTION
  StoragePy_assertNotNull(self);
  const auto &storage = StoragePy_Unpack(self);
  auto device_type = storage.device();
  return py::cast(device_type).release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyObject *StoragePy_dtype(PyObject *self, PyObject *_unused) {
  HANDLE_MS_EXCEPTION
  StoragePy_assertNotNull(self);
  const auto &storage = StoragePy_Unpack(self);
  auto type_id = storage.GetTypeId();
  TypePtr type_ptr = TypeIdToType(type_id);
  return py::cast(type_ptr).release().ptr();
  HANDLE_MS_EXCEPTION_END
}

static PyGetSetDef StoragePy_properties[] = {{"device", (getter)StoragePy_device, nullptr, nullptr, nullptr},
                                             {"dtype", (getter)StoragePy_dtype, nullptr, nullptr, nullptr},
                                             {nullptr}};

static PyMethodDef StoragePy_methods[] = {
  {"copy_", reinterpret_cast<PyCFunction>(StoragePy_Copy_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"element_size", StoragePy_ElementSize, METH_NOARGS, nullptr},
  {"resize_", StoragePy_Resize_, METH_O, nullptr},
  {"nbytes", StoragePy_NBytes, METH_NOARGS, nullptr},
  {"size", StoragePy_NBytes, METH_NOARGS, nullptr},
  {"data_ptr", StoragePy_DataPtr, METH_NOARGS, nullptr},
  {"_share_fd_cpu_", StoragePy_shareFd, METH_NOARGS, nullptr},
  {"_new_shared_fd_cpu", StoragePy_newSharedFd, METH_VARARGS | METH_STATIC, nullptr},
  {"is_shared", StoragePy_isShared, METH_NOARGS, nullptr},
  {nullptr}};

void RegStorage(py::module *m) {
  static std::vector<PyMethodDef> methods;
  size_t i = 0;
  while (true) {
    methods.push_back(StoragePy_methods[i]);
    if (!StoragePy_methods[i].ml_name) {
      break;
    }
    i++;
  }

  StoragePyType.tp_methods = methods.data();
  StoragePyType.tp_getset = StoragePy_properties;
  if (PyType_Ready(&StoragePyType) < 0) {
    return;
  }
  Py_INCREF(&StoragePyType);
  SetStoragePyType(&StoragePyType);
  m->add_object("StoragePy", reinterpret_cast<PyObject *>(&StoragePyType));
}

PyObject *CreateStorageObj(const Storage &storage) {
  PyTypeObject *type = &StoragePyType;
  PyObject *obj = type->tp_alloc(type, 0);
  auto s = reinterpret_cast<StoragePy *>(obj);
  new (&s->cdata) Storage();
  s->cdata = std::move(storage);
  return obj;
}

PyObject *CreateStoragePyObj(const Storage &storage) {
  PyTypeObject *type = PyObjManager::Get().GetUntypedStorageClass();
  PyObject *obj = type->tp_alloc(type, 0);
  auto s = reinterpret_cast<StoragePy *>(obj);
  new (&s->cdata) Storage();
  s->cdata = std::move(storage);
  return obj;
}

}  // namespace mindspore
