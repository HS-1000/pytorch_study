import torch

# 1. (4,4) shape를 가지고 모든 값이 1이며 데이터 타입은 int32인 배열을 만드세요
tensor_ = torch.ones(4, 4, dtype=torch.int32)
# print(tensor_)
"""
ones 함수는 1로 초기화된 tensor를 생성 합니다.
비슷한 함수로 0으로 초기화하는 zeros 함수도 있습니다.
"""

# 2. 위의 데이터에서 1값이 아닌 3으로 바꿔보세요.
for i in range(len(tensor_)):
	for j in range(len(tensor_[i])):
		tensor_[i][j] = 3

# print(tensor_)
"""
tensor_ 의 각row에 접근하고 그 row를 반복해서 값을 3으로 저장합니다.
"""

# 3. 임의의 텐서 a, b를 만들고 element-wise sum을 진행하세요
a = torch.rand(2, 2)
b = torch.rand(2, 2)
sum_ = a + b
# print("a : \n", a, end="\n\n")
# print("b : \n", b, end="\n\n")
# print("sum_ : \n", sum_)
"""
rand 함수로 임의 2*2 배열 a, b를 만들었습니다.
tensor 간의 + 연산을 사용하거나 torch.add(a, b) 
함수로 element-wise sum 진행이 가능합니다. 
"""

# 4. (2,2) shape를 가진 역행렬이 존재하는 임의의 행렬을 텐서로 만들고, 
# 그 행렬의 역행렬을 구하는 코드를 작성해 보세요,
tensor_ = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
inv_tensor = torch.inverse(tensor_)
# print(tensor_ @ inv_tensor)
"""
inverse 함수로 역행렬을 계산하고
python @으로 항등행렬을 출력 확인합니다.
"""

# 5. x = torch.randn(3,4)를 선언하고 x의 원소들 중 0 이상인 
# 경우 True를 반납하는 논리형식의 텐서를 만들어 보세요.
x = torch.randn(3, 4)
bool_x = torch.empty(3, 4, dtype=torch.bool)
bool_x = x > 0
# print("x : \n", x, end="\n\n")
# print("bool_x : \n", bool_x)
"""
x > 0 으로 모든 값에 대해 0보다 큰지 논리연산을 진행합니다.
+, - 같은 연산도 가능합니다. x + 1 모든값에 + 1 연산을 합니다.
"""

# 6. torch.randn(1000)에서 가장 큰 값의 인덱스를 찾는 코드를 작성하세요.
tensor_ = torch.randn(1000)
max_index = -1
max_val = 0 
for i in range(len(tensor_)):
	if tensor_[i] > max_val:
		max_index = i
		max_val = tensor_[i]
# print(f"MAX: {max_val}\tINDEX: {max_index}")
"""
tensor_ 의 모든값을 확인하면서 기존 최댓값보다 크면 
인덱스와 값을 업데이트 합니다.
"""

# 7. w = torch.randn(3,4) v = torch.randn(4,5)일 때 matrix multiplication
w = torch.randn(3,4) 
v = torch.randn(4,5)
# print("w : \n", w, end="\n\n")
# print("v : \n", v, end="\n\n")
# print("matrix multiplication : \n", w @ v)
"""
@ 연산은 행렬곱연산을 진행합니다.
"""

# 8. torch.cat에 대해서 조사하고 활용한 예시 코드 작성
tensor_1 = torch.zeros(2, 2)
tensor_2 = torch.ones(2, 2)
vstack = torch.cat([tensor_1, tensor_2], dim=0)
hstack = torch.cat([tensor_1, tensor_2], dim=1)
# print("vstack shape : ", vstack.shape)
# print("hstack shape : ", hstack.shape)
"""
cat함수는 tensor을 병합하는 함수인데 dim인자로 방향을 결정합니다.
dim=0의 경우 가로로 확장하고, dim=1의 경우 세로로 확장합니다. 
"""

# 9. torch.view와 torch.reshape의 차이점에 대해서 조사하고 torch.permute와
# torch.transpose의 차이점에 대해서 조사하세요.

"""
view, reshape 모두 tensor의 shape를 변경하는 함수 입니다.
두 함수의 차이는 contiguos속성인 tensor에 사용 가능한지 입니다.
reshape함수가 원본을 반환하는지 복사본을 반환하는지 차이가 있지만
reshape가 항상 복사본을 반환하지 않아서 큰 차이는 아닌거 같습니다.
contiguos는 tensor의 값들의 메모리 순서의 방향에따라 결정된다.
transpose는 tensor의 두차원을 교환하는데 (3, 4) shape의 tensor
를 tensor.transpose(0, 1) 으로 (4, 3)으로 변환한다. 이 과정에서
값의 저장순서가 메모리의 순서와 달라지고 이런경우 contiguos가 
아니지만 tensor.contiguous() 으로 contiguous 가능합니다.

permute 함수는 transpose 와 비슷하게 차원을 교환하는데 transpose
는 두차원만 교환 가능하지만 permute는 3개 이상의 차원의 순서를 교환
가능합니다.

permute와 view는 비슷하게 shape를 변화시키지만  view는 값들의 순서를
유지하고 permute 차원을 변경해서 예를들어 단순한 2차원 배열을 permute
를 사용해서 변환하면 행과 열이 변합니다. 따라서 사실상 서로 다른 연산
이므로 주의가 필요합니다.
"""

# 10. 파이토치 내부 함수 (torch.nn)을 사용하지 않고 시그모이드 함수를 구현 한 뒤, 시그모
# 이드 함수 미분값을 원래 함수와 함께 시각화(미분하는 함수를 만들어서 사용)

from math import e
import matplotlib.pyplot as plt

sigmoid = lambda x: (1/(1 + e**(-x)))
sigmoid_diff = lambda x: 1/(2 + e**x + e**(-x))

input_ = torch.tensor([i/100 for i in range(-500, 500)])
sigmoid_val = sigmoid(input_)
sigmoid_diff_val = sigmoid_diff(input_)

# plt.figure()
# plt.plot(input_, sigmoid_val, color="blue")
# plt.plot(input_, sigmoid_diff_val, color="orange")
# plt.show()
"""
math 라이브러리에서 자연상수를 가져와 sigmoid와 그 미분함수를 정의
했습니다. 그리고 -5부터 5까지 0.01 단위로 이루어진 tensor을 만들어
함수의 입력, 출력을 구성했습니다.
"""

# 12. 파이토치 내부 함수 (torch.nn)을 사용하고 시그모이드 함수를 구현 한 뒤, 시그모이드
# 함수 미분값을 원래 함수와 함께 시각화(미분은 backward를 사용해서 진행)
x = torch.tensor([i/100 for i in range(-500, 500)], requires_grad=True)
sigmoid = torch.nn.Sigmoid()
y = sigmoid(x)
v = torch.ones_like(x)
y.backward(v)
x_list = [i/100 for i in range(-500, 500)]
plt.figure()
plt.plot(x_list, y.detach().numpy(), color="blue")
plt.plot(x_list, x.grad, color="orange")
plt.show()












