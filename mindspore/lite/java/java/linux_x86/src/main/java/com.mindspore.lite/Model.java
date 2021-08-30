/*
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

package com.mindspore.lite;

import java.nio.MappedByteBuffer;

/**
 * Model defines model in MindSpore Lite for managing graph.
 *
 * @since v1.0
 */
public class Model {
    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private long modelPtr;

    /**
     * Model construct function
     */
    public Model() {
        this.modelPtr = 0;
    }

    /**
     * Load the MindSpore Lite model from path.
     *
     * @param modelPath Model file path.
     * @return Whether the load is successful.
     */
    public boolean loadModel(String modelPath) {
        this.modelPtr = loadModelByPath(modelPath);
        return this.modelPtr != 0;
    }

    /**
     * Free all temporary memory in MindSpore Lite Model.
     */
    public void free() {
        this.free(this.modelPtr);
        this.modelPtr = 0;
    }

    /**
     * Free MetaGraph in MindSpore Lite Model to reduce memory usage during inference.
     */
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
