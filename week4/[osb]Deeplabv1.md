# [DeepLab V1](https://arxiv.org/pdf/1412.7062)
* DCNN(Deep Convolutional Neural Network)을 기반으로 한 Semantic Image Segmentation 모델.

## Introduction
* DCNN은 다양한 비전 영역에 대해서 뛰어난 성능을 보여줌.
* DCNN읜 한계
  * Signal Downsampling(신호 다운샘플링): 일반적으로 Pooling연산을 통해 Feature Map의 크기를 줄인다. 이는 해상도를 낮춰 픽셀 단위의 정확한 위치 정보를 잃게 되어 Semantic Segmentation의 정확성이 떨어짐.
  * Spatial Invariance(공간적 불변성): DCNN은 일반적으로 이미지의 전체의 특징을 잡는데 유리하여 분류와 같은 작업에서는 유용하지만, 픽셀단위의 분할과 같은 작업에서는 분리함. 특히 필터가 이미지의 위치에 따라 적용되지 않아, 객체의 위치와 모양 변화에 민감하지 않음.

* atrous convolution을 통해 Signal DownSampling 문제를 해결하고, Conditional Random Field(CRF)를 통해 Spatial Invariance 문제를 해결함.

## Related Work
* 기본적으로 Fully Convolutional Network(FCN)의 한계를 극복하고자 함.

### FCN 구조    
<img src='https://www.researchgate.net/publication/327521314/figure/fig1/AS:11431281212337231@1702589880856/Fully-convolutional-neural-network-architecture-FCN-8.tif' height = '250'>  

  * FCN의 구조를 보면 이미지를 Down-sampling을 진행하고 마지막에 Up-sampling을 수행하여 입력 이미지와 같은 크기의 Segmentation map을 출력.
  * 기본적으로 Down-sampling이 진행되고 이를 Up-sampling을 진행했기 때문에 경계선 정보와 같은 정보 손실이 발생.
  * FCN은 Downsampling과정이 3번 발생하므로 Latent vector에서는 1개의 픽셀이 8x8의 Receptive Field(수용 영역)를 가지지만 DeepLab은 이보다 더 큰 Receptive Field를 가져 더 큰 객체나 복잡한 장면에 대한 이해력이 높다.

### DeepLab architecture 
* FCN과 마찬가지로 VGG16을 기반으로 구조를 개선.  
![alt text](imgs/Hole_algorithm.png)
* Hole 알고리즘(atrous convolution)
  * 필터 크기는 그대로 유지하고 입력의 간격(Atrous rate)을 2 또는 4로 주어 Receptive field를 넓힘(해상도는 그대로 유지)
    * 더 큰 문맥 정보를 캡쳐할 수 있도록 도와줌
    * 일반적인 Pooling이 없어도 더 밀집된 feature map을 생성.
    * Down_samping이 없으므로 정보 손실이 없어짐.
    * 기존 FCN의 방법보다 해상도(Resolution)을 8배 늘어남.
    * PASCAL VOC에 대해 Fine-tuning이 약 10시간 걸리지만, FCN은 몇일 걸림.
  
* Receptive field
  * 일반적으로 Classification을 위한 CNN model들은 큰 Receptive field를 지님
    * VGG16의 경우 `224x224(zero padding)`과 `404x404(Convolutional)`
  * FCN의 경우 fully-connected layer 층에 7x7x4096의 큰 공간 크기를 가진 필터로 인해 `computational bottleneck`이 발생
    * 큰 필터로 인해 많은 계산이 필요로되며, 처리 속도가 느려짐.
    * 이를 4x4 또는 3x3의 크기로 `subsampling`하여 Receptive field와 계산량을 줄여서 해결.
      * Receptive field가 `128×128(zero padding)`과 `308×308(Convolutional)`로 감소.
      * 계산 시간이 2~3배 감소.
    * 막대한 성능 감소 없이 채널수를 4096에서 1024로 줄여 계산 시간과 메모리 사용량을 감소시킴.

* Conditional Random Field(CRF)  
![alt text](imgs/result1.png)
  * 기본적으로 Classification을 위한 CNN model들(DCNN)은 이미지에서 객체의 존재와 대략적인 위치를 신뢰성 있게 예측할 수 있지만, 정확한 윤곽을 찾는것은 어려움.
    * CNN의 여러 layer의 정보들을 활용(ex. FCN)하여 객체 경게 추정 정확도 향상.
    * super-pixel representation로 low-level segmentation method로 정확도 향상.
    


* Deeplab architecture  
<img src='imgs/deeplab_archi.png' height = '400'>  
[이미지출처](https://towardsdatascience.com/witnessing-the-progression-in-semantic-segmentation-deeplab-series-from-v1-to-v3-4f1dd0899e6e)

## Results
