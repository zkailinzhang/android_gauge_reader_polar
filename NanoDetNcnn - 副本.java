// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.nanodetncnn;

import android.content.res.AssetManager;
import android.view.Surface;

public class NanoDetNcnn
{  //public native boolean Init(AssetManager mgr);
    public class Obj
    {
        public float x;
        public float y;
        public float w;
        public float h;
        public String label;
        public float prob;
        public float value;

    }

    public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
    public native boolean openCamera(int facing,float min, float max);

    public native float resultRecall();

    public native boolean closeCamera();
    //public native boolean openCamera(int facing,float min,float max);
    //public native boolean closeCamera(float value);
    public native boolean setOutputWindow(Surface surface);

    public static NanoDetNcnnCall call;

    public native void setCall(float value);

    static {
        System.loadLibrary("nanodetncnn");
    }

    public static void setValue(float value) {
        if (call != null) {
            call.result(value);
        }
    }

    public interface NanoDetNcnnCall {
        void result(float value);
    }

}
