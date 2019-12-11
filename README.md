# DeepLearning

Deep Learning Study Project

### TensorFlow [![Sources](https://img.shields.io/badge/출처-TensorFlow-yellow)](https://www.tensorflow.org/)

- TensorFlow는 기계 학습과 딥러닝을 위해 Google에서 만든 E2E Opensource Platform이다.
- Tools, Library, Community Resource로 구성되어 첨단 기술을 구현할 수 있고, 개발자들은 ML이 접목된 application을 손쉽게 빌드 및 배포할 수 있다.

---

### Keras [![Sources](https://img.shields.io/badge/출처-Keras-yellow)](https://www.tensorflow.org/guide/keras?hl=ko)

- tf.keras는 딥러닝 엔진인 TensorFlow2.0의 딥러닝 모델 설계와 훈련을 위한 고수준(high-level) API로 정의
- 몇 가지 model-building APIs(Sequential, Functional, and Subclassing)을 제공하여 프로젝트에 적합한 추상화 수준을 선택할 수 있다

[![Sources](https://img.shields.io/badge/출처-TensorFlow-yellow)](https://medium.com/tensorflow/whats-coming-in-tensorflow-2-0-d3663832e9b8)

![tensorflow](images/tensorflow_v.2.0_architecture.png)

- `Model`은 `Network(네트워크)-Objective Function(목표함수)-Optimizer(최적화기)`로 구성되어 있다. `model=sequential()`
- `Compile`은 네트워크가 학습할 준비가 되었을 때, 이 구성요소들을 묶어 주는 역할을 한다. `model.compile`

- `Application`은 사전 교육된 가중치와 함께 사용할 수 있는 심층 학습 모델로 예측, 형상 추출 및 미세 조정에 사용할 수 있다. 모델을 인스턴스화할 때 가중치가 자동으로 다운로드된다. (저장 위치 `~/.keras/models/.`) [![Sources](https://img.shields.io/badge/출처-Applications-yellow)](https://keras.io/applications/)
	- Xception (88MB, 126 Layers)
	- VGG16 (528MB, 23)
	- VGG19(549MB, 26)
	- ResNet50(99MB, 168)
	- InceptionV3(92MB, 159)
	- InceptionResNetV2(215MB, 572)
	- MobileNet(17MB, 88)
	- DenseNet121(33MB, 121)
	- DenseNet169(57MB, 169)
	- DenseNet201(80MB, 201)
	- NASNet...

- `Objective Function(목표함수)`에는 mean_squared_error, categorical_crossentropy, binary_crossentropy
등이 있다. [![Sources](https://img.shields.io/badge/출처-Lossfunction-yellow)](https://keras.io/losses/)
	- loss function (or objective function, or optimization score function)은 모델을 컴파일 하기 위해 요구되는 2개의 파라미터 중 하나이다. `model.compile(loss='mean_squared_error', optimizer='sgd')`

- `Optimizer`는 모델을 컴파일 하기 위해 요구되는 또 하나의 파라미터로 Network을 Update할 수 있다.
- Network을 얼마간의 오차로 Update 하도록 알려 주는 역할은 Objective Function가 수행한다.

```js
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

- `AutoKeras`는 자동 시스템 학습(AutoML)을 위한 오픈 소스 소프트웨어 라이브러리이다. [![Sources](https://img.shields.io/badge/출처-AutoKeras-yellow)](https://autokeras.com/)
- AutoKeras는 아키텍처 및 심층 학습 모델의 하이퍼 프레임을 `자동`으로 검색하는 기능을 제공한다.
- AutoKeras1.0이 곧 출시 예정이다. (현재 AutoKeras는 Python3.6과만 호환)


---

### Pytorch

---

### fast.ai