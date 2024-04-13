#ifndef __TRT_DETECTOR_HPP__
#define __TRT_DETECTOR_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "trt_logger.hpp"
#include "trt_model.hpp"

namespace model{

namespace detector {

enum model {
    YOLOV5,
    YOLOV8
};

struct bbox {                   //->保存检测框信息
    float x0, x1, y0, y1;
    float confidence;
    bool  flg_remove;
    int   label;
    
    bbox() = default;
    bbox(float x0, float y0, float x1, float y1, float conf, int label) : 
        x0(x0), y0(y0), x1(x1), y1(y1), 
        confidence(conf), flg_remove(false), 
        label(label){};
};

class Detector : public Model{

public:
    Detector(std::string onnx_path, logger::Level level, Params params) : //->调用父类model构造函数
        Model(onnx_path, level, params) {};                               //->调用父类构造函数

public:
    virtual void setup(void const* data, std::size_t size) override;//->重写虚函数，setup负责分配host/device的memory, bindings, 以及创建推理所需要的上下文
    virtual void reset_task() override;         //->重写虚函数，清除检测框
    virtual bool preprocess_cpu() override;     //->重写虚函数，CPU前处理
    virtual bool preprocess_gpu() override;     //->重写虚函数，GPU前处理
    virtual bool postprocess_cpu() override;    //->重写虚函数，CPU后处理
    virtual bool postprocess_gpu() override;    //->重写虚函数，GPU后处理

private:
    std::vector<bbox> m_bboxes;    //->保存检测框
    int m_inputSize;               //->输入维度
    int m_imgArea;                 //->图像高×宽
    int m_outputSize;

    int m_DecodeNumSize;           //->输出1维度（检测框数量）
    int m_DecodeBoxesSize;         //->输出2维度（检测框坐标）
    int m_DecodeScoresSize;        //->输出3维度（检测框置信度）
    int m_DecodeClassesSize;       //->输出4维度（检测框类别）

    nvinfer1::Dims m_inputDims;            //->输入维度信息
    nvinfer1::Dims m_DecodeNumDims;        //->输出1的维度信息
    nvinfer1::Dims m_DecodeBoxesDims;      //->输出2的维度信息
    nvinfer1::Dims m_DecodeScoresDims;     //->输出3的维度信息
    nvinfer1::Dims m_DecodeClassesDims;    //->输出4的维度信息

    void* m_inputMemoryHost;               //->Host输入memory空间
    void* m_inputMemoryDevice;             //->Device输入memory空间
    void* m_outputMemoryNumHost;           //->Host输出1 memory空间
    void* m_outputMemoryNumDevice;         //->Device输出1 memory空间
    void* m_outputMemoryBoxesHost;         //->Host输出2 memory空间
    void* m_outputMemoryBoxesDevice;       //->Device输出2 memory空间
    void* m_outputMemoryScoresHost;        //->Host输出3 memory空间
    void* m_outputMemoryScoresDevice;      //->Device输出3 memory空间
    void* m_outputMemoryClassesHost;       //->Host输出4 memory空间
    void* m_outputMemoryClassesDevice;     //->Device输出4 memory空间
};


std::shared_ptr<Detector> make_detector(        //->创建目标检测器实例
    std::string onnx_path, logger::Level level, Params params);

}; //->namespace detector
}; //->namespace model

#endif //->__TRT_DETECTOR_HPP__
