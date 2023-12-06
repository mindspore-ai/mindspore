/*
 * Copyright 2023 Huawei Technologies Co., Ltd
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

package com.mindspore.micro;

import java.util.logging.Level;
import java.util.logging.Logger;


public class MSContext {
    private static final Logger LOGGER = Logger.getLogger(MSContext.class.toString());
    private long msContextPtr;
    private static final long EMPTY_CONTEXT_PTR_VALUE = 0;

    /**
     * Construct function.
     */
    public MSContext() {
        this.msContextPtr = EMPTY_CONTEXT_PTR_VALUE;
    }

    /**
     * Init Context.
     *
     * @return init status.
     */
    public boolean init() {
        this.msContextPtr = createDefaultMSContext();
        return this.msContextPtr != EMPTY_CONTEXT_PTR_VALUE;
    }

    /**
     * Free context.
     */
    public void free() {
        if (isInitialized()) {
            this.free(this.msContextPtr);
            this.msContextPtr = EMPTY_CONTEXT_PTR_VALUE;
        } else {
            LOGGER.log(Level.SEVERE, "[Micro Context free] Pointer from java is nullptr.\n");
        }
    }

    /**
     * Get context pointer.
     *
     * @return context pointer.
     */
    public long getMSContextPtr() {
        return msContextPtr;
    }

    /**
     * Check weather the msContextPtr has been initialized or not.
     *
     * @return true: initialized; false: not initialized.
     */
    private boolean isInitialized() {
        return this.msContextPtr != EMPTY_CONTEXT_PTR_VALUE;
    }

    private native long createDefaultMSContext();

    private native void free(long msContextPtr);
}
