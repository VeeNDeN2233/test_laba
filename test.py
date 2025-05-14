from random import uniform

def mse(outputs, targets):
    error = 0
    for i, output in enumerate(outputs):
        error += (output - targets[i]) ** 2
    return error / len(outputs)

class LinearRegression:
    def __init__(self, features_num):
        # Меньший диапазон для начальных весов
        self.weights = [uniform(-0.5, 0.5) for _ in range(features_num + 1)]

    def forward(self, input_features):
        output = 0
        for i, feature in enumerate(input_features):
            output += self.weights[i] * feature
        output += self.weights[-1]
        return output

    def train(self, inp, output, target, samples_num, lr):
        for j in range(len(self.weights) - 1):
            self.weights[j] -= lr * (2 / samples_num) * (output - target) * inp[j]
        self.weights[-1] -= lr * (2 / samples_num) * (output - target)

    def fit(self, inputs, targets, epochs=100, lr=0.1):
        # Нормализация входных данных
        max_values = [max(col) for col in zip(*inputs)]
        normalized_inputs = [[value / max_val for value, max_val in zip(inp, max_values)] for inp in inputs]

        for epoch in range(epochs):
            outputs = []
            for i, inp in enumerate(normalized_inputs):
                output = self.forward(inp)
                outputs.append(output)
                self.train(inp, output, targets[i], len(normalized_inputs), lr)
            print(f"epoch: {epoch}, error: {mse(outputs, targets)}")

if __name__ == '__main__':
    inputs = [
        [65000, 22, 43, 108], [62000, 21, 54, 131], [73000, 25, 62, 150],
        [84000, 27, 71, 137], [72000, 25, 61, 189], [81000, 27, 85, 105],
        [55000, 19, 35, 133], [73000, 25, 57, 121], [70000, 24, 55, 153],
        [45000, 17, 29, 137], [21000, 10, 12, 115], [26000, 10, 14, 115],
        [64000, 22, 55, 147], [89000, 30, 77, 155], [66000, 22, 45, 147],
        [68000, 23, 57, 175], [78000, 27, 55, 173], [40000, 15, 21, 133],
        [17000, 7, 11, 134], [25000, 11, 21, 125], [26400, 9, 54, 114],
        [21500, 8, 31, 120], [40600, 15, 77, 170], [43000, 17, 92, 190]
    ]

    targets = [55, 67, 78, 86, 67, 90, 45, 67, 66, 43, 23, 23, 56, 90, 56, 67, 68, 32, 12, 34, 55, 45, 85, 97]

    lr_model = LinearRegression(features_num=4)
    lr_model.fit(inputs, targets, epochs=150, lr=0.1)

    print(lr_model.weights)