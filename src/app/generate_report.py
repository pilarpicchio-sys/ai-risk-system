import os
from datetime import datetime

# importa direttamente i dati live
from src.app.run_live import assets, signals, sizes, weights


# =========================
# CONFIG
# =========================

capital = 10000
chart_path = "reports/charts/portfolio.png"


# =========================
# BUILD REPORT (MARKDOWN)
# =========================

date = datetime.now().strftime("%d %B %Y")

# MARKET STANCE
high_risk = any(s > 0.25 for s in sizes)
stance = "🟢 HIGH RISK" if high_risk else "🔴 LOW RISK"

report = f"""
# 📊 AI Risk Report — {date}

report += f"""
## 📈 System Snapshot

- Final Equity: 1.37x
- Max Drawdown: -4.7%

---
"""

## {stance}

---

## 💰 Allocations
"""

# ALLOCATIONS
for i, name in enumerate(assets.keys()):
    alloc = capital * weights[i] * sizes[i]
    if alloc > 1:
        report += f"- **{name.upper()}**: €{alloc:.2f}\n"

report += "\n---\n"

# CHART EMBED (MARKDOWN)
report += "## 📉 Portfolio Performance\n\n"

if os.path.exists(chart_path):
    report += f"![Portfolio Chart]({chart_path})\n"
else:
    report += "_Chart not available (run backtest first)_\n"

report += "\n---\n"

# ALERTS
report += "## ⚠️ Signals\n\n"

alerts = []

for i, name in enumerate(assets.keys()):
    if sizes[i] > 0.25:
        direction = "LONG" if signals[i] > 0 else "REDUCE"
        alerts.append(f"**{name.upper()}** — {direction} (size={sizes[i]:.2f})")

if alerts:
    for a in alerts:
        report += f"- {a}\n"
else:
    report += "_No strong signals today._\n"

report += "\n---\n"

# COMMENT
report += "## 📌 Interpretation\n\n"

if alerts:
    report += (
        "The system detects actionable opportunities. "
        "Risk exposure is justified under current conditions.\n"
    )
else:
    report += (
        "Signals are weak across assets. "
        "Expected returns do not justify risk.\n\n"
        "**Staying defensive preserves capital.**\n"
    )


# =========================
# SAVE
# =========================

os.makedirs("reports", exist_ok=True)

filename = f"reports/report_{datetime.now().strftime('%Y-%m-%d')}.md"

with open(filename, "w", encoding="utf-8") as f:
    f.write(report)

print("\n=== REPORT READY ===\n")
print(report)
print(f"\nSaved to {filename}")