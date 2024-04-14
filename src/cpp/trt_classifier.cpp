#include "opencv2/imgproc.hpp"
#include "trt_model.hpp"
#include "utils.hpp" 
#include "trt_logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "imagenet_labels.hpp"
#include "trt_classifier.hpp"
#include "trt_preprocess.hpp"
#include "utils.hpp"
#include <numeric>

using namespace std;
using namespace nvinfer1;

namespace model{

namespace classifier {

std::vector<float> softmax(const std::vector<float>& inputs) {
    //->计算所有输入的指数
    std::vector<float> expValues(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        expValues[i] = std::exp(inputs[i]);
    }

    //->计算指数值的和
    float sumOfExpValues = std::accumulate(expValues.begin(), expValues.end(), 0.0);

    //->计算softmax值
    std::vector<float> softmaxValues(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        softmaxValues[i] = expValues[i] / sumOfExpValues;
    }

    return softmaxValues;
}


void Classifier::setup(void const* data, size_t size) {     //->设置input/output bindings, 分配host/device的memory等
    m_runtime    = shared_ptr<IRuntime>(createInferRuntime(*m_logger), destroy_trt_ptr<IRuntime>);
    m_engine     = shared_ptr<ICudaEngine>(m_runtime->deserializeCudaEngine(data, size), destroy_trt_ptr<ICudaEngine>);
    m_context    = shared_ptr<IExecutionContext>(m_engine->createExecutionContext(), destroy_trt_ptr<IExecutionContext>);
    m_inputDims  = m_context->getBindingDimensions(0);
    m_outputDims = m_context->getBindingDimensions(1);      //->图像分类模型多为一输入一输出

    CUDA_CHECK(cudaStreamCreate(&m_stream));                //->创建CUDA流
    
    m_inputSize  = m_params->img.h * m_params->img.w * m_params->img.c * sizeof(float);
    m_outputSize = m_params->num_cls * sizeof(float);
    m_imgArea    = m_params->img.h * m_params->img.w;

    CUDA_CHECK(cudaMallocHost(&m_inputMemoryHost, m_inputSize));                //->输入分配Host pinned memory
    CUDA_CHECK(cudaMalloc(&m_inputMemoryDevice, m_inputSize));                  //->输入分配Device memory
    m_DeviceBindings.emplace_back(m_inputMemoryDevice);                         //->添加到binding

    CUDA_CHECK(cudaMallocHost(&m_outputMemoryHost, m_outputSize));              //->输出分配Host pinned memory
    CUDA_CHECK(cudaMalloc(&m_outputMemoryDevice, m_outputSize));                //->输出分配Device memory
    m_DeviceBindings.emplace_back(m_outputMemoryDevice);                        //->添加到binding
}

void Classifier::reset_task(){}     //->分类此函数为空实现，但是需要重写虚函数

bool Classifier::preprocess_cpu() {
    float mean[] = {0.406, 0.456, 0.485};   //->图像分类任务需要使用mean和std
    float std[]  = {0.225, 0.224, 0.229};

    cv::Mat input_image;                    //->读取数据
    input_image = cv::imread(m_imagePath);
    if (input_image.data == nullptr) {
        LOGE("ERROR: Image file not founded! Program terminated"); 
        return false;
    }

    m_timer->start_cpu();                   //->CPU测速

    cv::resize(input_image, input_image,    //->resize(默认是双线性插值)
               cv::Size(m_params->img.w, m_params->img.h), 0, 0, cv::INTER_LINEAR);

    //**************  host端进行normalization和BGR2RGB, NHWC->NCHW  ****************//
    int index;
    int offset_ch0 = m_imgArea * 0;
    int offset_ch1 = m_imgArea * 1;
    int offset_ch2 = m_imgArea * 2;
    for (int i = 0; i < m_inputDims.d[2]; i++) {
        for (int j = 0; j < m_inputDims.d[3]; j++) {
            index = i * m_inputDims.d[3] * m_inputDims.d[1] + j * m_inputDims.d[1];
            ((float*)m_inputMemoryHost)[offset_ch2++] = (input_image.data[index + 0] / 255.0f - mean[0]) / std[0];
            ((float*)m_inputMemoryHost)[offset_ch1++] = (input_image.data[index + 1] / 255.0f - mean[1]) / std[1];
            ((float*)m_inputMemoryHost)[offset_ch0++] = (input_image.data[index + 2] / 255.0f - mean[2]) / std[2];
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(m_inputMemoryDevice, m_inputMemoryHost, m_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));

    m_timer->stop_cpu();        //->CPU测速结束
    m_timer->duration_cpu<timer::Timer::ms>("preprocess(CPU)");

    return true;
}

bool Classifier::preprocess_gpu() {
    float mean[] = {0.406, 0.456, 0.485};     //->图像分类任务需要使用mean和std
    float std[]  = {0.225, 0.224, 0.229};

    cv::Mat input_image;                      //->读取数据
    input_image = cv::imread(m_imagePath);
    if (input_image.data == nullptr) {
        LOGE("ERROR: file not founded! Program terminated"); return false;
    }

    m_timer->start_gpu();                      //->CPU测速
    
    preprocess::preprocess_resize_gpu(input_image, (float*)m_inputMemoryDevice, //->GPU进行双线性插值
                                   m_params->img.h, m_params->img.w, 
                                   mean, std, preprocess::tactics::GPU_BILINEAR);

    m_timer->stop_gpu();                       //->CPU测速结束
    m_timer->duration_gpu("preprocess(GPU)");
    return true;
}


bool Classifier::postprocess_cpu() {
    m_timer->start_cpu();               //->CPU测速

    //->数据从device拷贝到host
    int output_size    = m_params->num_cls * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(m_outputMemoryHost, m_outputMemoryDevice, output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    ImageNetLabels labels;              //->分类出对应标签
    int pos = max_element((float*)m_outputMemoryHost, (float*)m_outputMemoryHost + m_params->num_cls) - (float*)m_outputMemoryHost;
    std::vector<float> result(m_params->num_cls);          //->创建vector用来保存模型输出的结果
    memcpy(result.data(), m_outputMemoryHost, output_size); //->数据拷贝
   
    std::vector<float> result_softmax = softmax(result);
    float confidence = result_softmax[pos] * 100;

    m_timer->stop_cpu();                //->CPU测速结束
    m_timer->duration_cpu<timer::Timer::ms>("postprocess(CPU)");

    LOG("Result:     %s", labels.imagenet_labelstring(pos).c_str());   
    LOG("Confidence  %.3f%%\n", confidence);   
    return true;
}


bool Classifier::postprocess_gpu() {
    return postprocess_cpu();           //->任务简单，在CPU进行后处理

}

shared_ptr<Classifier> make_classifier(
    std::string onnx_path, logger::Level level, model::Params params)
{
    return make_shared<Classifier>(onnx_path, level, params);
}

}; //->namespace classifier

}; //->namespace model
