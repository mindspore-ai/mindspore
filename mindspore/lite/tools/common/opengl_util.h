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

#ifndef LITE_OPENGL_UTIL_H
#define LITE_OPENGL_UTIL_H

#ifdef ENABLE_OPENGL_TEXTURE

#include <map>
#include <vector>
#include <utility>
#include "EGL/egl.h"
#include "GLES3/gl3.h"
#include "GLES3/gl32.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"

#define OPEN_GL_CHECK_ERROR

#ifdef OPEN_GL_CHECK_ERROR
#define OPENGL_CHECK_ERROR                                  \
  {                                                         \
    GLenum error = glGetError();                            \
    if (GL_NO_ERROR != error) {                             \
      MS_LOG(ERROR) << "ERR CHECK Fail, error = " << error; \
      return 0;                                             \
    }                                                       \
    MS_ASSERT(GL_NO_ERROR == error);                        \
  }
#else
#define OPENGL_CHECK_ERROR
#endif

namespace mindspore {
namespace OpenGL {
#define BIND_INDEX_0 0
#define BIND_INDEX_1 1
#define BIND_INDEX_2 2
#define BIND_INDEX_3 3
#define BIND_INDEX_4 4

class OpenGLRuntime {
 public:
  OpenGLRuntime();
  virtual ~OpenGLRuntime();

  bool Init();

  static GLuint LoadShader(GLenum shaderType, const char *pSource);
  static GLuint CreateComputeProgram(const char *pComputeSource);
  GLuint GLCreateSSBO(GLsizeiptr size, void *hostData = nullptr, GLenum type = GL_SHADER_STORAGE_BUFFER,
                      GLenum usage = GL_DYNAMIC_DRAW);
  bool CopyDeviceSSBOToHost(GLuint ssboBufferID, void *hostData, GLsizeiptr size);
  bool CopyHostToDeviceSSBO(void *hostData, GLuint ssboBufferID, GLsizeiptr size);

  GLuint GLCreateTexture(int w, int h, int c, GLenum textrueFormat = GL_RGBA32F, GLenum target = GL_TEXTURE_2D);
  bool CopyDeviceTextureToSSBO(GLuint textureID, GLuint ssboBufferID);
  bool CopyDeviceSSBOToTexture(GLuint ssboBufferID, GLuint textureID);

  // GLuint CopyHostToDeviceTexture(void *hostData, int width, int height);
  GLuint CopyHostToDeviceTexture(void *hostData, int width, int height, int channel);
  void *CopyDeviceTextureToHost(GLuint textureID);

  void PrintImage2DData(float *data, int w, int h, int c = 4);

 private:
  std::map<GLuint, std::pair<GLsizeiptr, GLenum>> m_ssbo_pool_;
  std::map<GLuint, std::pair<std::vector<int>, std::vector<GLenum>>> m_texture_pool_;
  EGLContext m_context_;
  EGLDisplay m_display_;
  EGLSurface m_surface_;
};
}  // namespace OpenGL
}  // namespace mindspore

#endif  // ENABLE_OPENGL_TEXTURE
#endif  // LITE_OPENGL_UTIL_H
