#ifndef __TRT_CLASSIFIER_HPP__
#define __TRT_CLASSIFIER_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "trt_logger.hpp"
#include "trt_model.hpp"

namespace model{

namespace classifier {
class Classifier : public Model{

public:
    // 这个构造函数实际上调用的是父类的Model的构造函数
    Classifier(std::string onnx_path, logger::Level level, Params params) : //->调用父类model构造函数
        Model(onnx_path, level, params) {};                                 //->调用父类构造函数

public:
    virtual void setup(void const* data, std::size_t size) override;//->重写虚函数，setup负责分配host/device的memory, bindings, 以及创建推理所需要的上下文      
    virtual void reset_task() override;         //->重写虚函数，图像分类此函数为空
    virtual bool preprocess_cpu() override;     //->重写虚函数，CPU前处理
    virtual bool preprocess_gpu() override;     //->重写虚函数，GPU前处理
    virtual bool postprocess_cpu() override;    //->重写虚函数，CPU后处理
    virtual bool postprocess_gpu() override;    //->重写虚函数，GPU后处理

private:
    float m_confidence;
    std::string m_label;
    int m_inputSize; 
    int m_imgArea;
    int m_outputSize;

    nvinfer1::Dims m_inputDims;         //->输入维度信息
    nvinfer1::Dims m_outputDims;        //->输出维度信息

    void* m_inputMemoryHost;            //->Host输入 memory空间
    void* m_inputMemoryDevice;          //->Device输入 memory空间
    void* m_outputMemoryHost;           //->Host输出 memory空间
    void* m_outputMemoryDevice;         //->Device输出 memory空间
};

std::shared_ptr<Classifier> make_classifier(        //->创建图像分类器实例
    std::string onnx_path, logger::Level level, Params params);

}; //->namespace classifier
}; //->namespace model

#endif //->__TRT_CLASSIFIER_HPP__
