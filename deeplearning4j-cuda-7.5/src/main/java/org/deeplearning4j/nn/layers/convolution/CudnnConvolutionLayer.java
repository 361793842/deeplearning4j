/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.nn.layers.convolution;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.util.Dropout;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;

import static org.bytedeco.javacpp.cuda.*;
import static org.bytedeco.javacpp.cudnn.*;

/**
 * cuDNN-based convolution layer
 *
 * @author saudet
 */
public class CudnnConvolutionLayer extends ConvolutionLayer {
    FloatPointer alpha = new FloatPointer(1.0f);
    FloatPointer beta  = new FloatPointer(0.0f);

    cudnnContext cudnnHandle = new cudnnContext() {
        { deallocator(new Pointer.Deallocator() {
            @Override public void deallocate() { destroyHandles(); }
        });
    }};
    cudnnTensorStruct srcTensorDesc = new cudnnTensorStruct(),
                      dstTensorDesc = new cudnnTensorStruct(),
                      biasTensorDesc = new cudnnTensorStruct(),
                      deltaTensorDesc = new cudnnTensorStruct();
    cudnnFilterStruct filterDesc = new cudnnFilterStruct();
    cudnnConvolutionStruct convDesc = new cudnnConvolutionStruct();
    cudnnActivationStruct activationDesc = new cudnnActivationStruct();

    int dataType = Nd4j.dataType() == DataBuffer.Type.DOUBLE ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    int tensorFormat = CUDNN_TENSOR_NCHW;

    public CudnnConvolutionLayer(NeuralNetConfiguration conf) {
        super(conf);
        createHandles();
    }

    public CudnnConvolutionLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
        createHandles();
    }

    static void checkCuda(int error) {
        if (error != cudaSuccess) {
            throw new RuntimeException("CUDA error = " + error);
        }
    }

    static void checkCudnn(int status) {
        if (status != CUDNN_STATUS_SUCCESS) {
            throw new RuntimeException("cuDNN status = " + status);
        }
    }

    void createHandles() {
        checkCudnn(cudnnCreate(cudnnHandle));
        checkCudnn(cudnnCreateTensorDescriptor(srcTensorDesc));
        checkCudnn(cudnnCreateTensorDescriptor(dstTensorDesc));
        checkCudnn(cudnnCreateTensorDescriptor(biasTensorDesc));
        checkCudnn(cudnnCreateTensorDescriptor(deltaTensorDesc));
        checkCudnn(cudnnCreateFilterDescriptor(filterDesc));
        checkCudnn(cudnnCreateConvolutionDescriptor(convDesc));
        checkCudnn(cudnnCreateActivationDescriptor(activationDesc));
    }

    void destroyHandles() {
        checkCudnn(cudnnDestroyActivationDescriptor(activationDesc));
        checkCudnn(cudnnDestroyConvolutionDescriptor(convDesc));
        checkCudnn(cudnnDestroyFilterDescriptor(filterDesc));
        checkCudnn(cudnnDestroyTensorDescriptor(srcTensorDesc));
        checkCudnn(cudnnDestroyTensorDescriptor(dstTensorDesc));
        checkCudnn(cudnnDestroyTensorDescriptor(biasTensorDesc));
        checkCudnn(cudnnDestroyTensorDescriptor(deltaTensorDesc));
        checkCudnn(cudnnDestroy(cudnnHandle));
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        INDArray weights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);

        int miniBatch = input.size(0);
        int inH = input.size(2);
        int inW = input.size(3);

        int outDepth = weights.size(0);
        int inDepth = weights.size(1);
        int kH = weights.size(2);
        int kW = weights.size(3);

        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] pad = layerConf().getPadding();

        int outH = Convolution.outSize(inH, kernel[0], strides[0], pad[0],false);
        int outW = Convolution.outSize(inW, kernel[1], strides[1], pad[1], false);


        INDArray biasGradView = gradientViews.get(ConvolutionParamInitializer.BIAS_KEY);
        INDArray weightGradView = gradientViews.get(ConvolutionParamInitializer.WEIGHT_KEY);    //4d, c order. Shape: [outDepth,inDepth,kH,kW]
        INDArray delta;
        String afn = conf.getLayer().getActivationFunction();

        if("identity".equals(afn)){
            delta = epsilon;    //avoid doing .muli with 1s
        } else {
            INDArray sigmaPrimeZ = preOutput(true);
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(
                    afn, sigmaPrimeZ, conf.getExtraArgs()).derivative());
            delta = sigmaPrimeZ.muli(epsilon);  //Current shape: [miniBatch,outD,outH,outW]
        }

        if (!Shape.strideDescendingCAscendingF(delta)) {
            // apparently not supported by cuDNN
            delta = delta.dup();
        }

        int[] srcStride = input.stride();
        int[] deltaStride = delta.stride();
        int[] algo = new int[1];
        checkCudnn(cudnnSetTensor4dDescriptorEx(srcTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                srcStride[0], srcStride[1], srcStride[2], srcStride[3]));
        checkCudnn(cudnnSetTensor4dDescriptorEx(deltaTensorDesc, dataType, miniBatch, outDepth, outH, outW,
                deltaStride[0], deltaStride[1], deltaStride[2], deltaStride[3]));
        checkCudnn(cudnnSetConvolution2dDescriptor(convDesc, pad[0], pad[1], strides[0], strides[1], 1, 1, CUDNN_CROSS_CORRELATION));
        checkCudnn(cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorFormat, outDepth, inDepth, kH, kW));
        checkCudnn(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle, srcTensorDesc, deltaTensorDesc, convDesc,
                 filterDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, algo));

        INDArray epsNext = Nd4j.create(new int[]{miniBatch,inDepth,inH,inW},'c');
        int[] dstStride = epsNext.stride();

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(input, weights, weightGradView, biasGradView, delta, epsNext);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer filterData = allocator.getPointer(weights, context);
        Pointer filterGradData = allocator.getPointer(weightGradView, context);
        Pointer biasGradData = allocator.getPointer(biasGradView, context);
        Pointer deltaData = allocator.getPointer(delta, context);
        Pointer dstData = allocator.getPointer(epsNext, context);

        SizeTPointer sizeInBytes = new SizeTPointer(1);
        Pointer workSpace = null;
        checkCudnn(cudnnSetTensor4dDescriptorEx(dstTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                dstStride[0], dstStride[1], dstStride[2], dstStride[3]));
        checkCudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, srcTensorDesc,
                deltaTensorDesc, convDesc, filterDesc, algo[0], sizeInBytes));
        long sizeInBytes1 = sizeInBytes.get(0);
        checkCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, filterDesc,
                deltaTensorDesc, convDesc, dstTensorDesc, algo[0], sizeInBytes));
        long sizeInBytes2 = sizeInBytes.get(0);
        if (sizeInBytes1 > 0 || sizeInBytes2 > 0) {
            checkCuda(cudaMalloc(workSpace = new Pointer(), Math.max(sizeInBytes1, sizeInBytes2)));
        }

        checkCudnn(cudnnSetTensor4dDescriptor(biasTensorDesc, tensorFormat, dataType, 1, outDepth, 1, 1));
        checkCudnn(cudnnConvolutionBackwardBias(cudnnHandle, alpha, deltaTensorDesc, deltaData, beta, biasTensorDesc, biasGradData));
        checkCudnn(cudnnConvolutionBackwardFilter(cudnnHandle, alpha, srcTensorDesc, srcData, deltaTensorDesc, deltaData,
                convDesc, algo[0], workSpace, sizeInBytes1, beta, filterDesc, filterGradData));
        checkCudnn(cudnnConvolutionBackwardData(cudnnHandle, alpha, filterDesc, filterData, deltaTensorDesc, deltaData, convDesc,
                algo[0], workSpace, sizeInBytes2, beta, dstTensorDesc, dstData));

        if (workSpace != null) {
            checkCuda(cudaFree(workSpace));
        }

        allocator.registerAction(context, input, weights, weightGradView, biasGradView, delta, epsNext);

        Gradient retGradient = new DefaultGradient();
        retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, biasGradView);
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, weightGradView, 'c');

        return new Pair<>(retGradient,epsNext);
    }

    public INDArray preOutput(boolean training) {
        INDArray weights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);
        INDArray bias = getParam(ConvolutionParamInitializer.BIAS_KEY);
        if(conf.isUseDropConnect() && training) {
            if (conf.getLayer().getDropOut() > 0) {
                weights = Dropout.applyDropConnect(this, ConvolutionParamInitializer.WEIGHT_KEY);
            }
        }

        int miniBatch = input.size(0);
        int inH = input.size(2);
        int inW = input.size(3);

        int outDepth = weights.size(0);
        int inDepth = weights.size(1);
        int kH = weights.size(2);
        int kW = weights.size(3);

        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] pad = layerConf().getPadding();

        int outH = Convolution.outSize(inH, kernel[0], strides[0], pad[0],false);
        int outW = Convolution.outSize(inW, kernel[1], strides[1], pad[1], false);

        int[] srcStride = input.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(srcTensorDesc, dataType, miniBatch, inDepth, inH, inW,
                srcStride[0], srcStride[1], srcStride[2], srcStride[3]));
        checkCudnn(cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorFormat, outDepth, inDepth, kH, kW));
        checkCudnn(cudnnSetConvolution2dDescriptor(convDesc, pad[0], pad[1], strides[0], strides[1], 1, 1, CUDNN_CROSS_CORRELATION));

        // find dimension of convolution output
        int[] algo = new int[1], n = new int[1], c = new int[1], h = new int[1], w = new int[1];
        checkCudnn(cudnnGetConvolution2dForwardOutputDim(convDesc, srcTensorDesc, filterDesc, n, c, h, w));
        INDArray z = Nd4j.createUninitialized(new int[]{n[0],c[0],h[0],w[0]},'c');
        int[] dstStride = z.stride();
        checkCudnn(cudnnSetTensor4dDescriptorEx(dstTensorDesc, dataType, n[0], c[0], h[0], w[0],
                dstStride[0], dstStride[1], dstStride[2], dstStride[3]));
        checkCudnn(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, srcTensorDesc, filterDesc, convDesc,
                dstTensorDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algo));

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(input, weights, bias, z);
        Pointer srcData = allocator.getPointer(input, context);
        Pointer filterData = allocator.getPointer(weights, context);
        Pointer biasData = allocator.getPointer(bias, context);
        Pointer dstData = allocator.getPointer(z, context);

        SizeTPointer sizeInBytes = new SizeTPointer(1);
        Pointer workSpace = null;
        checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, srcTensorDesc,
                filterDesc, convDesc, dstTensorDesc, algo[0], sizeInBytes));
        if (sizeInBytes.get(0) > 0) {
            checkCuda(cudaMalloc(workSpace = new Pointer(), sizeInBytes.get(0)));
        }
        checkCudnn(cudnnConvolutionForward(cudnnHandle, alpha, srcTensorDesc, srcData,
                filterDesc, filterData, convDesc, algo[0], workSpace, sizeInBytes.get(0),
                beta, dstTensorDesc, dstData));

        checkCudnn(cudnnSetTensor4dDescriptor(biasTensorDesc, tensorFormat, dataType, 1, c[0], 1, 1));
        checkCudnn(cudnnAddTensor(cudnnHandle, alpha, biasTensorDesc, biasData, alpha, dstTensorDesc, dstData));

        if (workSpace != null) {
            checkCuda(cudaFree(workSpace));
        }

        allocator.registerAction(context, input, weights, bias, z);

        return z;
    }

    @Override
    public INDArray activate(boolean training) {
        if(input == null)
            throw new IllegalArgumentException("No null input allowed");
        applyDropOutIfNecessary(training);

        INDArray z = preOutput(training);
        INDArray activation = z;

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(z);
        Pointer dstData = allocator.getPointer(z, context);

        switch (conf.getLayer().getActivationFunction()) {
            case "identity":
                break;
            case "sigmoid":
                checkCudnn(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0));
                checkCudnn(cudnnActivationForward(cudnnHandle, activationDesc, alpha, dstTensorDesc, dstData, beta, dstTensorDesc, dstData));
                break;
            case "relu":
                checkCudnn(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));
                checkCudnn(cudnnActivationForward(cudnnHandle, activationDesc, alpha, dstTensorDesc, dstData, beta, dstTensorDesc, dstData));
                break;
            case "tanh":
                checkCudnn(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0));
                checkCudnn(cudnnActivationForward(cudnnHandle, activationDesc, alpha, dstTensorDesc, dstData, beta, dstTensorDesc, dstData));
                break;
            case "softmax":
                checkCudnn(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
                        CUDNN_SOFTMAX_MODE_CHANNEL, alpha, dstTensorDesc, dstData, beta, dstTensorDesc, dstData));
                break;
            case "logsoftmax":
                checkCudnn(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_LOG,
                        CUDNN_SOFTMAX_MODE_CHANNEL, alpha, dstTensorDesc, dstData, beta, dstTensorDesc, dstData));
                break;
            default:
                activation = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), z));
        }

        allocator.registerAction(context, z);

        return activation;
    }

}
