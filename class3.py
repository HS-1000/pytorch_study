import re

# 입력 문자열
text = """

Train Loss: 2.2967
Train Accuracy: 19.44%
Val Loss: 2.2855
Val Accuracy: 19.08%
Running Time: 28.9576

Train Loss: 2.2112
Train Accuracy: 30.67%
Val Loss: 1.9006
Val Accuracy: 46.68%
Running Time: 28.9510

Train Loss: 1.0760
Train Accuracy: 66.55%
Val Loss: 0.6133
Val Accuracy: 80.74%
Running Time: 28.9633

Train Loss: 0.4856
Train Accuracy: 85.70%
Val Loss: 0.4013
Val Accuracy: 87.94%
Running Time: 28.9205

Train Loss: 0.3811
Train Accuracy: 88.92%
Val Loss: 0.3410
Val Accuracy: 89.78%
Running Time: 29.0078

Train Loss: 0.3239
Train Accuracy: 90.65%
Val Loss: 0.2965
Val Accuracy: 91.42%
Running Time: 28.9593

Train Loss: 0.2781
Train Accuracy: 91.98%
Val Loss: 0.2539
Val Accuracy: 92.59%
Running Time: 28.9205

Train Loss: 0.2404
Train Accuracy: 93.07%
Val Loss: 0.2312
Val Accuracy: 93.30%
Running Time: 28.9894

Train Loss: 0.2111
Train Accuracy: 93.96%
Val Loss: 0.2042
Val Accuracy: 94.11%
Running Time: 28.9955

Train Loss: 0.1865
Train Accuracy: 94.62%
Val Loss: 0.1792
Val Accuracy: 94.78%
Running Time: 29.0097

"""
lines = text.split("\n")

count = 0
for line in lines:
    if line.find(": ") > 0:
        count += 1
        if count % 5:
            index = line.find(": ")
            print(line[index+2:], end = "\t")
        else:
            print("")
