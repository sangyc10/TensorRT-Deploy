/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __YOLOV5_PLUGINS__
#define __YOLOV5_PLUGINS__

#include <cassert>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <vector>
#include "NvInferPlugin.h"

using namespace nvinfer1;

#define CUDA_CHECK(status)                                                                                        \
    {                                                                                                             \
        if (status != 0)                                                                                          \
        {                                                                                                         \
            std::cout << "CUDA failure: " << cudaGetErrorString(status) << " in file " << __FILE__ << " at line " \
                      << __LINE__ << std::endl;                                                                   \
            abort();                                                                                              \
        }                                                                                                         \
    }


namespace {     //->定义插件版本和插件名
    const char *YOLOLAYER_PLUGIN_VERSION{"1"};
    const char *YOLOLAYER_PLUGIN_NAME{"YoloLayer_TRT"};
} //->namespace

//->Plugin类是插件类，用来写插件的具体实现
class YoloLayer : public IPluginV2DynamicExt {
public:
    YoloLayer() = delete;
    YoloLayer(const void *data, size_t length);         //->clone以及反序列化的时候用的构造函数
    YoloLayer(                                          //->parse时候用的构造函数
        const uint &maxStride, const uint &numClasses,
        const std::vector<float> &anchors, const float &scoreThreshold);

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int         getNbOutputs() const noexcept override;
    size_t      getSerializationSize() const noexcept override;
    const char* getPluginNamespace() const noexcept override;
    DataType    getOutputDataType(int index, const DataType *inputType, int nbInputs) const noexcept override;
    DimsExprs   getOutputDimensions(int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder) noexcept override;
    size_t      getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const noexcept override;
    int         initialize() noexcept override;
    void        terminate() noexcept override;
    void        serialize(void *buffer) const noexcept override;  
    void        destroy() noexcept override;
    int32_t     enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
    bool        supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) noexcept override;
    void        configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs, const DynamicPluginTensorDesc *out, int nbOutputs) noexcept override;
    void        setPluginNamespace(const char *pluginNamespace) noexcept override;
    void        attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator *gpuAllocator) noexcept override;
    void        detachFromContext() noexcept override;
    
private:
    std::string m_Namespace{""};
    int m_NetWidth{0};
    int m_NetHeight{0};
    int m_MaxStride{0};
    int m_NumClasses{0};
    std::vector<float> m_Anchors;
    std::vector<DimsHW> m_FeatureSpatialSize;
    float m_ScoreThreshold{0};
    uint64_t m_OutputSize{0};
};

//->PluginCreator类是插件工厂类，用来根据需求创建插件，插件创建类需要继承IPluginCreator，需要重写虚函数
class YoloLayerPluginCreator : public IPluginCreator
{
public:
    YoloLayerPluginCreator() noexcept;  //->初始化mFC以及mPluginAttributes

    ~YoloLayerPluginCreator() noexcept;

    const char*                  getPluginName() const noexcept override; 
    const char*                  getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    const char*                  getPluginNamespace() const noexcept override;
    IPluginV2DynamicExt*         createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override;
    IPluginV2DynamicExt*         deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept;
    void                         setPluginNamespace(const char *pluginNamespace) noexcept;

private:
    static PluginFieldCollection mFC;                  //->负责将onnx中的参数传递给Plugin
    static std::vector<PluginField> mPluginAttributes; //->保存从onnx中获取的参数
    std::string mNamespace;
};

#endif //->__YOLOV5_PLUGINS__
