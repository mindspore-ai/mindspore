/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.mindspore.lite;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class Model {
    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private long modelPtr;

    public Model() {
        this.modelPtr = 0;
    }

    public boolean loadModel(String modelPath) {
        this.modelPtr = loadModelByPath(modelPath);
        return this.modelPtr != 0;
    }

    public void free() {
        this.free(this.modelPtr);
        this.modelPtr = 0;
    }

    public void freeBuffer() {
        this.freeBuffer(this.modelPtr);
    }

    protected long getModelPtr() {
        return modelPtr;
    }

    private native long loadModel(MappedByteBuffer buffer);

    private native long loadModelByPath(String modelPath);

    private native void free(long modelPtr);

    private native void freeBuffer(long modelPtr);
}
