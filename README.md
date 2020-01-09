# DeepLearning

Deep Learning Study Project

### TensorFlow [![Sources](https://img.shields.io/badge/출처-TensorFlow-yellow)](https://www.tensorflow.org/)

- TensorFlow는 기계 학습과 딥러닝을 위해 Google에서 만든 E2E Opensource Platform이다.
- Tools, Library, Community Resource로 구성되어 첨단 기술을 구현할 수 있고, 개발자들은 ML이 접목된 application을 손쉽게 빌드 및 배포할 수 있다.

#### Installation

- 다른 프로그램에 영향을 미치지 않도록 Anaconda 환경에서 설치한다.
	- 설치파일=Anaconda3-5.2.0-Windows-x86_64.exe (파이썬 3.6)
	- numpy 버전=1.15.4
	- tensorflow 버전=1.12.0
	- keras 버전=2.2.4

```bash
$conda update -n base conda # conda 자체를 업데이트
$conda update --all # 설치된 파이썬 패키지를 모두 최신 버전으로 업데이트
```

- tensorflow 이름을 갖는 conda 환경한다.

```bash
$ conda create -n tensorflow python=3.7
```

- 환경을 활성화시키고 그 안에서 pip를 이용하여 텐서플로우를 설치

```bash
$ source activate tensorflow # Linux 환경
$ virtualenv tensorflow # Windows 환경

(tensorflow)$ pip install tensorflow # tensorflow 설치
(tensorflow)$ jupyter notebook --port=8888

(tensorflow)$ source deactivate # Linux 환경
(tensorflow)$ virtualenv deactivate # Windows 환경
```

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


#### 패션 MNIST 데이터셋 Import [![Sources](https://img.shields.io/badge/출처-TensorflowGuide-yellow)](https://www.tensorflow.org/tutorials/keras/classification?hl=ko)

- 10개의 범주(category)와 70,000개의 흑백 이미지로 구성된 패션 MNIST 데이터셋을 사용
- 네트워크를 훈련하는데 60,000개의 이미지를 사용하여, 네트워크가 얼마나 정확하게 이미지를 분류하는지 10,000개의 이미지로 평가


#### [Data Set 준비]

- 필요한 훈련set, 검증set, Test Set을 준비 : MNIST은 28×28 크기의 0~9사이의 숫자 이미지와 이에 해당하는 레이블(Label)로 구성된 데이터베이스
- 

```js
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```
- load_data() 함수를 호출하면 네 개의 넘파이(NumPy) 배열이 반환
	- `train_images`와 `train_labels` 배열은 모델 학습에 사용되는 훈련 세트
	- `test_images`와 `test_labels` 배열은 모델 테스트에 사용되는 테스트 세트

- 나중에 이미지를 출력할 때 사용하기 위해 별도의 변수를 만들어 저장

```js
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

#### Data 탐색

```js
train_images.shape // 훈련 세트에 60,000개의 이미지가 있으며, 각 이미지는 28x28 픽셀로 표현
len(train_labels) // 훈련 세트에는 60,000개의 레이블
train_labels // 각 레이블은 0과 9사이의 정수
test_images.shape // 테스트 세트에는 10,000개의 이미지가 있으며 28x28 픽셀로 표현
len(test_labels) // 테스트 세트는 10,000개의 이미지에 대한 레이블을 가지고 있음
````

#### 데이터 전처리

- 네트워크를 훈련하기 전에 데이터를 전처리해야 하며, 훈련 세트에 있는 첫 번째 이미지를 보면 픽셀 값의 범위가 0~255 사이라는 것을 알 수 있다.

```js
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```

- 신경망 모델에 주입하기 전에 이 값의 범위를 0~1 사이로 조정하기 위해 255로 나누어야 힌다. 훈련 세트와 테스트 세트를 동일한 방식으로 전처리하는 것이 중요하다.

```js
train_images = train_images / 255.0
test_images = test_images / 255.0
```

- 훈련 세트에서 처음 25개 이미지와 그 아래 클래스 이름을 출력. 데이터 포맷이 올바른지 확인하고 네트워크 구성과 훈련할 준비

```js
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

#### 모델 구성

- 신경망 모델을 만들려면 모델의 층을 구성한 다음 모델을 컴파일 해야 한다.
- 신경망의 기본 구성 요소는 층(layer)으로 주입된 데이터에서 표현을 추출한다. tf.keras.layers.Dense와 같은 층들의 가중치(parameter)는 훈련하는 동안 학습된다.

```js
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

- Batch Size
- Epoch

---

### Pytorch

---

### fast.ai