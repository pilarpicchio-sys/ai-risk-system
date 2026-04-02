import sys
import os
import subprocess

# fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
from datetime import datetime


import json

with open("reports/data/live_signals.json") as f:
    data = json.load(f)

assets = {name: None for name in data["assets"]}
signals = data["signals"]
sizes = data["sizes"]
weights = data["weights"]

# =========================
# CONFIG
# =========================

capital = 10000
chart_path = "reports/charts/portfolio.png"

st.set_page_config(page_title="AI Risk System", layout="centered")

# =========================
# ENSURE CHART (CACHED)
# =========================

@st.cache_data(show_spinner=False)
def ensure_chart():
    return chart_path

# =========================
# HEADER
# =========================

st.title("Know when to take risk. Know when to stay out.")
st.caption("A daily system that tells you how much capital should be exposed to the market.")

date = datetime.now().strftime("%d %B %Y")
st.caption(date)

# =========================
# DECISION
# =========================

high_risk = any(s > 0.25 for s in sizes)

if high_risk:
    st.success("🟢 HIGH RISK")
    st.write("Opportunities detected.\n\nRisk exposure is justified.")
else:
    st.error("🔴 LOW RISK")
    st.write("No meaningful edge detected.\n\nStay defensive.")

# =========================
# EXPOSURE + CASH
# =========================

total_exposure = sum(weights[i] * sizes[i] for i in range(len(sizes)))
cash = 1 - total_exposure

col1, col2 = st.columns(2)

with col1:
    st.metric("Invested", f"{total_exposure*100:.1f}%")

with col2:
    st.metric("Cash", f"{cash*100:.1f}%")

st.caption(
    "This is not a trade signal.\n"
    "It shows how much capital is justified to keep invested under current conditions."
)

# =========================
# ALLOCATIONS
# =========================

st.markdown("## 💰 Allocations")
st.caption("Suggested allocation of capital based on current risk conditions.")

for i, name in enumerate(assets.keys()):
    pct = weights[i] * sizes[i]
    alloc = capital * pct

    if pct > 0.001:
        st.write(f"**{name.upper()}** → {pct*100:.1f}% (€{alloc:.0f})")

# =========================
# CHART
# =========================

st.markdown("## 📉 Portfolio")

chart_file = ensure_chart()

if os.path.exists(chart_file):
    st.image(chart_file)
else:
    st.error("Chart not available")



st.caption(
    "This chart shows how the system would have performed over time.\n\n"
    "It is based on a historical simulation with dynamic risk allocation.\n\n"
    "It does not predict future returns.\n"
    "It shows how risk is managed across market conditions."
)

st.caption("Past performance is not indicative of future results.")

# =========================
# SIGNALS
# =========================

st.markdown("## ⚠️ Signals")

found = False

for i, name in enumerate(assets.keys()):
    if sizes[i] > 0.25:
        found = True
        direction = "LONG" if signals[i] > 0 else "REDUCE"
        st.write(f"**{name.upper()}** → {direction} (size={sizes[i]:.2f})")

if not found:
    st.write("No actionable signals.\n\nThe best decision today is to do nothing.")

# =========================
# WHY
# =========================

st.markdown("## 📌 Why")

if high_risk:
    st.write(
        "Multiple signals align across assets.\n\n"
        "Risk/reward is favorable.\n\n"
        "Increasing exposure is justified."
    )
else:
    st.write(
        "Signals are weak across assets.\n\n"
        "Expected returns do not justify taking risk.\n\n"
        "**Preserving capital is the optimal decision.**"
    )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "This system does not predict markets.\n"
    "It controls risk.\n\n"
    "Most losses come from being exposed at the wrong time."
)