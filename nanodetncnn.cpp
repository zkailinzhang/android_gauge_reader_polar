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

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "nanodet.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <jni.h>

#include <iostream>

#include <string>
#include <vector>
#include <list>
#include <deque>

// ncnn
#include "layer.h"
#include "net.h"
#include "benchmark.h"
#include "mat.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

//void create_vm(JavaVM** jvm, JNIEnv** env)
//{
//    JavaVMInitArgs args;
//    JavaVMOption options[1];
//    args.version = JNI_VERSION_1_6;
//    args.nOptions = 1;
//    options[0].optionString = "-Djava.class.path=./";
//    args.options = options;
//    args.ignoreUnrecognized = JNI_FALSE;
//    JNI_CreateJavaVM(jvm, env, &args);
//}


static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}




static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

static NanoDet* g_nanodet = 0;
static ncnn::Mutex lock;

static std::deque<float> result_value_list;
static float min_value = 0;
static float max_value = 0;
static bool RECALL=false;
static float result_mid = 0;


class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    //这个rgb引用，即使不做画操作，手机也是显示。

    // nanodet
    {
        ncnn::MutexLockGuard g(lock);

        if (g_nanodet) {

            //检测结果放在这里了，没用static
            std::vector <Object> objects;
            //检测仪表
            g_nanodet->detect(rgb, objects);

            for (size_t i = 0; i < objects.size(); i++) {
                const Object &obj = objects[i];

                if (obj.label != 0)
                    continue;

                if (obj.prob < 0.7)
                    continue;

                if (obj.w / obj.h < 0.8 && obj.w / obj.h > 1.2)
                    continue;

//                if (obj.x / rgb.cols < 0.08 || obj.x / rgb.cols > 0.5)
//                   continue;

               // if (obj.y / rgb.rows < 0.1 || obj.y / rgb.rows > 0.7)
                   // continue;
                //读数  结果存到类里， 还是在这里，三次for
                //float value = g_nanodet->detectvalue(rgb, obj,min_value,max_value);
                float value = g_nanodet->polardetect(rgb, obj,min_value,max_value);
                //float value = g_nanodet->detectthree(rgb, obj,min_value,max_value);
                //float value = g_nanodet->detectfour(rgb, obj,min_value,max_value);
                if (value ==float(10086.111f) || value==0.0 ||
                            isnan(value) == true)
                    continue;
                result_value_list.push_back(value);

                __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%f  value", value);


                if (result_value_list.size() == 3) {
                    //是一张图读数三次，还是三张图 三次， 三取中，

                    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%f  ,%f ,%f value", result_value_list[0],result_value_list[1],result_value_list[2]);

                    std::sort(result_value_list.begin(),result_value_list.end());
                    //画仪表  其他时候显示 ‘’，读数结果保存三次，才会娴熟读数，
                    //g_nanodet->draw(rgb, objects，result_value_list[1]);
                    if(result_value_list[1]!=NULL)
                        g_nanodet->draw(rgb, obj,result_value_list[1]);

                    RECALL=true;
                    result_mid = result_value_list[1];
                    //还是清空
                    //result_value_list.pop_front();
                    //result_value_list.clear();

                    result_value_list = std::deque<float>();

                }
                else{
                    g_nanodet->draw(rgb, obj);

                }
            }

        }
        else
        {
            draw_unsupported(rgb);
        }
    }

    draw_fps(rgb);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_nanodet;
        g_nanodet = 0;
    }

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL
Java_com_tencent_nanodetncnn_NanoDetNcnn_loadModel(JNIEnv *env, jobject thiz, jobject assetManager,
                                                   jint modelid, jint cpugpu) {
    if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1) {
        return JNI_FALSE;
    }

    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char *modeltypes[] =
            {
                    "m",
                    "m-416",
                    "g",
                    "ELite0_320",
                    "ELite1_416",
                    "ELite2_512",
                    "RepVGG-A0_416"
            };

    const int target_sizes[] =
            {
                    320,
                    416,
                    416,
                    320,
                    416,
                    512,
                    416
            };

    const float mean_vals[][3] =
            {
                    {103.53f, 116.28f, 123.675f},
                    {103.53f, 116.28f, 123.675f},
                    {103.53f, 116.28f, 123.675f},
                    {127.f,   127.f,   127.f},
                    {127.f,   127.f,   127.f},
                    {127.f,   127.f,   127.f},
                    {103.53f, 116.28f, 123.675f}
            };

    const float norm_vals[][3] =
            {
                    {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
                    {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
                    {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
                    {1.f / 128.f,   1.f / 128.f,  1.f / 128.f},
                    {1.f / 128.f,   1.f / 128.f,  1.f / 128.f},
                    {1.f / 128.f,   1.f / 128.f,  1.f / 128.f},
                    {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f}
            };

    const char *modeltype = modeltypes[(int) modelid];
    int target_size = target_sizes[(int) modelid];
    bool use_gpu = (int) cpugpu == 1;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        if (use_gpu && ncnn::get_gpu_count() == 0) {
            // no gpu
            delete g_nanodet;
            g_nanodet = 0;
        } else {
            if (!g_nanodet)
                g_nanodet = new NanoDet;
            //g_nanodet->nanodet.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);

            g_nanodet->load(mgr, modeltype, target_size, mean_vals[(int) modelid],
                            norm_vals[(int) modelid], use_gpu);
        }
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL
Java_com_tencent_nanodetncnn_NanoDetNcnn_openCamera(JNIEnv *env, jobject thiz, jint facing) {
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    //Jobject

    g_camera->open((int) facing);
    //g_camera->open((int) facing,min_value,max_value);

    return JNI_TRUE;
}






// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL
Java_com_tencent_nanodetncnn_NanoDetNcnn_closeCamera(JNIEnv *env, jobject thiz) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL
Java_com_tencent_nanodetncnn_NanoDetNcnn_setOutputWindow(JNIEnv *env, jobject thiz,
                                                         jobject surface) {
    ANativeWindow *win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

JNIEXPORT void JNICALL Java_com_tencent_nanodetncnn_NanoDetNcnn_setCall(JNIEnv* env, jfloat value)
{
    jclass dpclazz = env->FindClass("com/tencent/nanodetncnn/NanoDetNcnn");
    if(dpclazz==0){
        return;
    }
    jmethodID method1 = env->GetStaticMethodID(dpclazz,"setValue","(F)V");
    if(method1==0){
        return;
    }
    env->CallStaticVoidMethod(dpclazz,method1,value);
}

/**
JNIEXPORT void JNICALL Java_com_shqcjd_yuhuantechnologymobile_NanoDetNcnn_setCall(JNIEnv * env)
{
    jclass dpclazz = (*env)->FindClass(env,"com/shqcjd/yuhuantechnologymobile/NanoDetNcnn");
    if(dpclazz==0){
        return;
    }
    jmethodID method1 = (*env)->GetStaticMethodID(env,dpclazz,"setvalue","(F)V");
    if(method1==0){
        return;
    }
    jfloat value = 10086.11;
    (*env)->CallStaticVoidMethod(env,dpclazz,method1,value);

}


JNIEXPORT void JNICALL Java_cn_itcast_ndkcallback_DataProvider_callmethod1
        (JNIEnv * env, jobject obj){
    //在c代码里面调用java代码里面的方法
    // java 反射
    //1 . 找到java代码的 class文件
    //    jclass      (*FindClass)(JNIEnv*, const char*);
    jclass dpclazz = (*env)->FindClass(env,"cn/itcast/ndkcallback/DataProvider");
    if(dpclazz==0){
        LOGI("find class error");
        return;
    }
    LOGI("find class ");

    //2 寻找class里面的方法
    //   jmethodID   (*GetMethodID)(JNIEnv*, jclass, const char*, const char*);
    jmethodID method1 = (*env)->GetMethodID(env,dpclazz,"helloFromJava","()V");
    if(method1==0){
        LOGI("find method1 error");
        return;
    }
    LOGI("find method1 ");
    //3 .调用这个方法
    //    void        (*CallVoidMethod)(JNIEnv*, jobject, jmethodID, ...);
    (*env)->CallVoidMethod(env,obj,method1);
}
**/

}