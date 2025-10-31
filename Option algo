"""
Streamlit app: Orderbook / Market-Maker vs Human vs Adversary simulation
Educational & detection-focused — intended for research, compliance, or teaching.

Save this file as `streamlit_orderbook_sim.py` and run with:
    streamlit run streamlit_orderbook_sim.py

Features:
- Interactive controls for market parameters (fair price, quotes, sizes, timings)
- Runs a discrete-time simulation of quotes and trades (simple matching engine)
- Shows timeline table, summary metrics, detection alerts
- Plots bid/ask evolution and trade markers
- CSV download of timeline for offline analysis

DISCLAIMER: This app is for educational/detection purposes only. Do not use it to design or execute manipulative trading strategies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="Orderbook Simulation (Educational)", layout="wide")

st.title("Orderbook Simulation — Market-Maker, Human, and Adversary (Educational)")
st.caption("Use this tool to explore how suspicious quote patterns (quote-flicker, bait-and-withdraw) show up in an order timeline.")

# Sidebar controls
st.sidebar.header("Simulation Controls")
fair_price = st.sidebar.number_input("Fair price of option", value=40.0, min_value=0.1, step=0.5)
T = st.sidebar.slider("Time steps (discrete ticks)", min_value=10, max_value=500, value=60, step=10)
seed = st.sidebar.number_input("Random seed (for reproducibility)", value=42, step=1)

# Market-maker baseline quotes
mm_bid = st.sidebar.number_input("MM starting bid", value=20.0, step=0.5)
mm_ask = st.sidebar.number_input("MM starting ask", value=100.0, step=0.5)
mm_size = st.sidebar.number_input("MM visible size", value=10, step=1)

# Human trader parameters
human_order_price = st.sidebar.number_input("Human limit buy price (example)", value=21.0, step=0.5)
human_order_time = st.sidebar.slider("Time step when human posts limit buy", min_value=0, max_value=T-1, value=5)
human_order_size = st.sidebar.number_input("Human order size", value=2, step=1)

# Adversary behaviour
adv_trigger_time = st.sidebar.slider("Adversary trigger time (when deceptive quote appears)", min_value=0, max_value=T-1, value=15)
adv_deep_ask_pct = st.sidebar.slider("Adversary deep ask (% of fair)", min_value=20, max_value=120, value=80)
adv_withdraw_pct = st.sidebar.slider("Adversary withdraw ask (% of fair) after bait", min_value=100, max_value=200, value=120)
adv_size = st.sidebar.number_input("Adversary visible size", value=5, step=1)

# Additional toggles
run_monte_carlo = st.sidebar.checkbox("Run multiple Monte-Carlo runs (estimate alert frequency)", value=False)
mc_runs = st.sidebar.number_input("Monte-Carlo runs", value=50, min_value=1, max_value=1000, step=1)

st.sidebar.markdown("---")
if st.sidebar.button("Run simulation"):
    st.session_state.run = True

# Ensure a run flag
if "run" not in st.session_state:
    st.session_state.run = False

if st.session_state.run:
    random.seed(int(seed))
    np.random.seed(int(seed))

    def run_single_sim(T, fair_price, mm_bid, mm_ask, mm_size,
                       human_order_price, human_order_time, human_order_size,
                       adv_trigger_time, adv_deep_ask_pct, adv_withdraw_pct, adv_size,
                       seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        events = []

        def post_quote(actor, bid, ask, size, t):
            events.append({"time": t, "actor": actor, "type": "quote", "bid": float(bid), "ask": float(ask), "size": int(size)})

        def trade(executor, price, size, t, counterparty):
            events.append({"time": t, "actor": executor, "type": "trade", "price": float(price), "size": int(size), "counterparty": counterparty})

        # Initial state
        mm_inventory = 0
        mm_cash = 0.0
        human_inventory = 0
        human_cash = 0.0
        adv_inventory = 0
        adv_cash = 0.0

        human_order_posted = False

        # Adversary target quotes
        adv_low_ask = fair_price * (adv_deep_ask_pct / 100.0)
        adv_withdraw_ask = fair_price * (adv_withdraw_pct / 100.0)
        adv_high_bid = fair_price * 1.2

        for t in range(T):
            # MM posts
            post_quote("MM", mm_bid, mm_ask, mm_size, t)

            # Adv posts either bait or normal wide quote
            if t == adv_trigger_time:
                # Bait: attractive ask (low) while putting a high bid to appear active
                post_quote("Adv", adv_high_bid, adv_low_ask, adv_size, t)
            elif t == adv_trigger_time + 1:
                # Withdrawal step: Adv posts very wide ask (simulates withdrawal)
                post_quote("Adv", mm_bid - 1, adv_withdraw_ask, 1, t)
            else:
                # Normal wide quotes
                post_quote("Adv", mm_bid - 1, mm_ask + 1, max(1, adv_size // 2), t)

            # Human posts their limit buy at specified time
            if t == human_order_time and not human_order_posted:
                events.append({"time": t, "actor": "Human", "type": "limit_buy", "price": float(human_order_price), "size": int(human_order_size)})
                human_order_posted = True

            # Matching logic: collect quotes at this time
            current_quotes = [e for e in events if e["time"] == t and e["type"] == "quote"]
            best_ask = min(q["ask"] for q in current_quotes)
            best_ask_actor = min(current_quotes, key=lambda q: q["ask"])["actor"]
            best_bid = max(q["bid"] for q in current_quotes)
            best_bid_actor = max(current_quotes, key=lambda q: q["bid"])["actor"]

            # Execute human limit buy if possible
            if human_order_posted and human_inventory == 0:
                if best_ask <= human_order_price:
                    trade("Human", best_ask, human_order_size, t, best_ask_actor)
                    human_inventory += human_order_size
                    human_cash -= best_ask * human_order_size
                    # Update counterparty
                    if best_ask_actor == "MM":
                        mm_inventory -= human_order_size
                        mm_cash += best_ask * human_order_size
                    elif best_ask_actor == "Adv":
                        adv_inventory -= human_order_size
                        adv_cash += best_ask * human_order_size
                    human_order_posted = False

            # Optionally simulate a later market buy by human to lift best ask (discovery)
            if t == int(T * 0.4):
                # human places market buy of same size as earlier
                trade_price = best_ask
                trade("Human", trade_price, human_order_size, t, best_ask_actor)
                human_inventory += human_order_size
                human_cash -= trade_price * human_order_size
                if best_ask_actor == "MM":
                    mm_inventory -= human_order_size
                    mm_cash += trade_price * human_order_size
                elif best_ask_actor == "Adv":
                    adv_inventory -= human_order_size
                    adv_cash += trade_price * human_order_size

            # MM simple inventory management (benign)
            if mm_inventory < -5:
                mm_bid += 2
                mm_ask = max(mm_bid + 5, mm_ask)
            elif mm_inventory > 5:
                mm_ask -= 2
                mm_bid = min(mm_ask - 5, mm_bid)

            # Random small drift to MM quotes for realism
            mm_bid += random.choice([-0.5, 0, 0.5])
            mm_ask += random.choice([-0.5, 0, 0.5])

        # Build dataframe
        df = pd.DataFrame(events)
        df = df.sort_values(["time", "type"], ascending=[True, True]).reset_index(drop=True)

        # Simple detection heuristics
        alerts = []
        quotes_by_actor_time = df[df["type"] == "quote"].copy()
        for actor in quotes_by_actor_time["actor"].unique():
            actor_quotes = quotes_by_actor_time[quotes_by_actor_time["actor"] == actor].sort_values("time")
            actor_quotes = actor_quotes.reset_index(drop=True)
            actor_quotes["ask_change"] = actor_quotes["ask"].diff().abs().fillna(0)
            if (actor_quotes["ask_change"] > 0.2 * fair_price).any():
                alerts.append(f"Large quote movement detected for {actor} (ask change >20% fair price).")

        # bait-and-withdraw detection
        for actor in quotes_by_actor_time["actor"].unique():
            actor_quotes = quotes_by_actor_time[quotes_by_actor_time["actor"] == actor].sort_values("time").reset_index(drop=True)
            for i in range(len(actor_quotes) - 1):
                a1 = actor_quotes.loc[i, "ask"]
                a2 = actor_quotes.loc[i + 1, "ask"]
                if (a1 <= fair_price * 0.8) and (a2 >= fair_price * 1.1):
                    alerts.append(f"Possible bait-and-withdraw pattern by {actor}: deep ask then withdrawal within 1 step.")

        # human P&L
        # compute mark-to-market using fair_price
        human_inventory = sum([e.get('size', 0) for e in events if e.get('actor') == 'Human' and e.get('type') == 'trade'])
        human_cash = sum([-e.get('price', 0) * e.get('size', 0) for e in events if e.get('actor') == 'Human' and e.get('type') == 'trade'])
        human_mtm = human_inventory * fair_price + human_cash

        summary = {
            'human_inventory': human_inventory,
            'human_cash': human_cash,
            'human_mtm': human_mtm,
            'mm_inventory': mm_inventory,
            'mm_cash': mm_cash,
            'adv_inventory': adv_inventory,
            'adv_cash': adv_cash,
            'alerts': alerts
        }

        return df, summary

    # Run single sim
    df, summary = run_single_sim(T, fair_price, mm_bid, mm_ask, mm_size,
                                 human_order_price, human_order_time, human_order_size,
                                 adv_trigger_time, adv_deep_ask_pct, adv_withdraw_pct, adv_size,
                                 seed=int(seed))

    st.subheader("Simulation timeline")
    st.dataframe(df)

    # Summary
    st.subheader("Summary metrics")
    st.write(f"Human inventory: **{summary['human_inventory']}**")
    st.write(f"Human cash (net paid): **{summary['human_cash']:.2f}**")
    st.write(f"Human mark-to-market (at fair {fair_price}): **{summary['human_mtm']:.2f}**")
    if summary['alerts']:
        st.error("Alerts detected:")
        for a in summary['alerts']:
            st.write("- ", a)
    else:
        st.success("No alerts detected in this run.")

    # Visualizations: extract time-series of best bid/ask
    quotes = df[df['type'] == 'quote'].copy()
    if not quotes.empty:
        # For each time step, compute best bid and best ask
        bests = []
        for t in sorted(quotes['time'].unique()):
            q_t = quotes[quotes['time'] == t]
            bests.append({
                'time': t,
                'best_bid': q_t['bid'].max(),
                'best_ask': q_t['ask'].min()
            })
        bests_df = pd.DataFrame(bests).set_index('time')

        st.subheader("Best bid / Best ask over time")
        st.line_chart(bests_df)

    # Plot trade points on a simple table
    trades = df[df['type'] == 'trade'].copy()
    if not trades.empty:
        st.subheader("Trades")
        st.dataframe(trades)

    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button(label="Download timeline CSV", data=csv, file_name=f"orderbook_timeline_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv", mime='text/csv')

    # Optionally run Monte Carlo to estimate how often alerts occur
    if run_monte_carlo:
        st.subheader("Monte-Carlo: alert frequency (approx)")
        mc_alert_counts = 0
        mc_results = []
        progress = st.progress(0)
        for i in range(int(mc_runs)):
            seed_i = int(seed) + i + 1
            df_i, summary_i = run_single_sim(T, fair_price, mm_bid, mm_ask, mm_size,
                                             human_order_price, human_order_time, human_order_size,
                                             adv_trigger_time, adv_deep_ask_pct, adv_withdraw_pct, adv_size,
                                             seed=seed_i)
            has_alert = len(summary_i['alerts']) > 0
            mc_results.append(1 if has_alert else 0)
            mc_alert_counts += 1 if has_alert else 0
            progress.progress((i + 1) / int(mc_runs))
        freq = mc_alert_counts / int(mc_runs)
        st.write(f"Alerts occurred in **{mc_alert_counts}** of **{int(mc_runs)}** runs — frequency **{freq:.2%}**")
        mc_df = pd.DataFrame({'run': list(range(1, int(mc_runs) + 1)), 'alert': mc_results})
        st.dataframe(mc_df)

    # Helpful next steps
    st.markdown("---")
    st.markdown("**Next steps you can ask for:**\n- A `requirements.txt` for this app.\n- A `README.md` explaining how to run and the educational disclaimer.\n- Add more detection metrics (time-weighted spreads, sequence scoring).\n- Convert to a multi-file app and add unit tests.")

else:
    st.info("Configure simulation parameters in the sidebar and click 'Run simulation'.")

# Footer
st.markdown("---")
st.caption("Educational simulation for detection of suspicious quote behaviour. Not intended for trading or market manipulation.")
