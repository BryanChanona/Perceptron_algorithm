import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Perceptron import BatchPerceptron


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, 0:2].to_numpy(dtype=float)
    y = df.iloc[:, 2].to_numpy(dtype=float)
    return X, y


def visualize_weights(weight_history, lr):
    plt.figure(figsize=(8, 5))
    for i in range(weight_history.shape[1]):
        plt.plot(weight_history[:, i], label=f"W{i}")
    plt.title(f"Evolución de pesos (λ = {lr})")
    plt.xlabel("Épocas")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_errors(results, epochs):
    plt.figure(figsize=(10, 6))
    for lr, errors in results:
        plt.plot(range(epochs), errors, label=f"λ = {lr}")
    plt.xlabel("Épocas")
    plt.ylabel("Error global (L2)")
    plt.title("Comparación de errores para distintos λ")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():

    X, y = load_dataset("students.csv")

    epochs = 500
    learning_rates = [1e-6, 0.25, 0.5, 0.75, 1]

    all_results = []
    summary = []

    for lr in learning_rates:

        model = BatchPerceptron(learning_rate=lr, epochs=epochs)
        model.fit(X, y)

        errors = model.get_error_history()
        weights = model.get_weight_history()

        all_results.append((lr, errors))

        summary.append({
            "lambda": lr,
            **{f"W{i}_init": weights[0][i] for i in range(weights.shape[1])},
            **{f"W{i}_final": weights[-1][i] for i in range(weights.shape[1])}
        })

        if lr == 1e-6:
            visualize_weights(weights, lr)

    visualize_errors(all_results, epochs)

    pd.DataFrame(summary).to_csv("weights_summary.csv", index=False)
    print("Resumen guardado.")


if __name__ == "__main__":
    main()
