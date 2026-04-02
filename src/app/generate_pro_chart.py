import numpy as np
import matplotlib.pyplot as plt
import os


def generate_pro_chart(equity_curve, benchmark_curve=None, name="portfolio"):

    equity = np.array(equity_curve)

    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1

    os.makedirs("reports/charts", exist_ok=True)
    path = f"reports/charts/{name}.png"

    plt.figure(figsize=(10, 6))

    # =========================
    # EQUITY + BENCHMARK
    # =========================

    plt.subplot(2, 1, 1)

    plt.plot(equity, label="System")

    if benchmark_curve is not None:
        plt.plot(benchmark_curve, linestyle="--", label="S&P500")

    plt.title("Equity vs Benchmark")
    plt.legend()
    plt.grid(True)

    # =========================
    # DRAWDOWN
    # =========================

    plt.subplot(2, 1, 2)
    plt.plot(drawdown)
    plt.title("Drawdown")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path
def generate_pro_chart_fast():

    import numpy as np
    import matplotlib.pyplot as plt
    import os

    portfolio_curve = np.load("reports/data/portfolio_curve.npy")
    bench_curve = np.load("reports/data/bench_curve.npy")

    peak = np.maximum.accumulate(portfolio_curve)
    drawdown = portfolio_curve / peak - 1

    os.makedirs("reports/charts", exist_ok=True)
    path = "reports/charts/portfolio.png"

    plt.figure(figsize=(10, 6))

    # EQUITY + BENCHMARK
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_curve, label="System")
    plt.plot(bench_curve, linestyle="--", label="S&P500")
    plt.legend()
    plt.grid(True)

    # DRAWDOWN
    plt.subplot(2, 1, 2)
    plt.plot(drawdown)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path
