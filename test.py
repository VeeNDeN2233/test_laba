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
        [65000, 43], [62000, 54], [73000, 62], [84000, 71], [72000, 61], [81000, 85],
        [55000, 35], [73000, 57], [70000, 55], [45000, 29], [21000, 12], [26000, 14],
        [64000, 55], [89000, 77], [66000, 45], [68000, 57], [78000, 55], [40000, 21],
        [17000, 11], [25000, 21], [26400, 54], [21500, 31], [40600, 77], [43000, 92]
    ]

    targets = [55, 67, 78, 86, 67, 90, 45, 67, 66, 43, 23, 23, 56, 90, 56, 67, 68, 32, 12, 34, 55, 45, 85, 97]

    lr_model = LinearRegression(features_num=2)
    lr_model.fit(inputs, targets, epochs=100, lr=0.1)

    print(lr_model.weights)