import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
from matplotlib.backends.backend_pdf import PdfPages


# -------------------------
# Helper Functions
# -------------------------
def prepare_data(df):
    """Coerce all non-cname/year columns to numeric, then fill NaNs with column means."""
    for col in df.columns:
        if col in ['cname', 'year']:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
    return df


def optimize_synthetic_control(donor_data, treated_data):
    """Optimize weights W s.t. W>=0, sum(W)=1 to match treated_data from donor_data."""
    scaler = StandardScaler()
    X_donor = scaler.fit_transform(donor_data)
    X_treated = scaler.transform(treated_data.values.reshape(1, -1))[0]

    n = X_donor.shape[0]

    def objective(w):
        synth = w @ X_donor
        return np.sum((X_treated - synth) ** 2)

    w0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},)

    try:
        res = minimize(objective, w0, method='SLSQP',
                       bounds=bounds, constraints=constraints)
        if not res.success:
            raise ValueError(res.message)
        w = res.x
    except Exception:
        # fallback to uniform weights
        w = np.ones(n) / n

    w[w < 0.01] = 0
    if w.sum() > 0:
        w /= w.sum()
    return w


# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="Synthetic Control Simulator", layout="wide")
    st.title("Synthetic Control Simulator (Multi-Country)")

    with st.expander("📋 Instructions (Click to Expand)", expanded=True):
        st.markdown("""
        **How to Use This App:**

        1. **Upload Data**: CSV with `cname`, `year`, and numeric metrics.  
        2. **Select Outcome** and **Predictors**.  
        3. **Pick Treated Country(s)**.  
        4. **Adjust Predictors** via sliders per country.  
        5. **Run Simulation**: View balance, outcome comparison, KDE, donor table, and download a PDF.
        """)

    if 'adjusted_vals' not in st.session_state:
        st.session_state.adjusted_vals = {}

    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if not uploaded:
        return

    df = pd.read_csv(uploaded)
    df = prepare_data(df)
    if 'cname' not in df.columns:
        st.error("CSV must contain a 'cname' column.")
        return

    # Outcome & predictor selection
    numeric_cols = [c for c in df.columns if c not in ['cname', 'year']]
    selected_outcome = st.selectbox("Select Outcome Variable", options=numeric_cols)
    predictors = [c for c in numeric_cols if c != selected_outcome]
    selected_predictors = st.multiselect("Select Predictor Variables",
                                         options=predictors,
                                         default=predictors)

    # Treated country picker
    countries = df['cname'].unique().tolist()
    selected_countries = st.multiselect("Select Treated Country(s)",
                                        options=countries,
                                        default=[countries[0]])

    # Dynamic sliders
    for country in selected_countries:
        st.subheader(f"Adjustments for {country}")
        sub = df[df['cname'] == country]
        means = sub[selected_predictors].mean()
        country_adj = {}
        for col in selected_predictors:
            mn, mx = float(sub[col].min()), float(sub[col].max())
            dv = float(means[col])
            if mn == mx:
                eps = abs(dv) * 0.01 if dv != 0 else 0.01
                mn, mx = dv - eps, dv + eps
            key = f"{country}_{col}"
            if key not in st.session_state.adjusted_vals:
                st.session_state.adjusted_vals[key] = dv
            val = st.slider(f"{col} for {country}",
                            min_value=mn,
                            max_value=mx,
                            value=st.session_state.adjusted_vals[key],
                            key=key)
            st.session_state.adjusted_vals[key] = val
            country_adj[col] = val
        st.session_state.adjusted_vals[country] = country_adj

    # Run simulation
    if st.button("Run Simulation"):
        all_results = {}
        for country in selected_countries:
            treated_df = df[df['cname'] == country]
            tr_series = pd.Series(st.session_state.adjusted_vals[country])

            donor_preds = df.groupby('cname')[selected_predictors].mean()
            donor_outs = df.groupby('cname')[selected_outcome].mean()
            pool = donor_preds.drop(selected_countries, errors='ignore')

            w = optimize_synthetic_control(pool, tr_series)
            synth = float(w @ donor_outs.loc[pool.index].values)
            actual = treated_df[selected_outcome].mean()

            # Build balance DataFrame
            synth_vals = w @ pool.values
            balance_df = pd.DataFrame({
                'Actual': tr_series.values,
                'Synthetic': synth_vals
            }, index=pool.columns)
            balance_df['Difference'] = balance_df['Actual'] - balance_df['Synthetic']

            # Donor table + summary rows
            donor_df = pd.DataFrame({
                'Country': pool.index,
                'Weight (%)': w * 100,
                selected_outcome: donor_outs.loc[pool.index].values,
                'Contribution': w * donor_outs.loc[pool.index].values
            }).sort_values('Weight (%)', ascending=False)
            summary = pd.DataFrame([
                {'Country': 'Synthetic', 'Weight (%)': np.nan, selected_outcome: synth, 'Contribution': np.nan},
                {'Country': 'Actual', 'Weight (%)': np.nan, selected_outcome: actual, 'Contribution': np.nan},
            ])
            full_table = pd.concat([donor_df, summary], ignore_index=True)

            all_results[country] = {
                'actual': actual,
                'synth': synth,
                'balance': balance_df,
                'table': full_table,
                'weights': w
            }

        # Display
        st.header("Simulation Results")
        for country, res in all_results.items():
            with st.expander(country, expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Actual", f"{res['actual']:.2f}")
                    st.metric("Synthetic", f"{res['synth']:.2f}")
                    st.metric("Difference", f"{res['synth'] - res['actual']:.2f}")
                    st.write("### Donor Table")
                    st.dataframe(res['table'].style.format({
                        'Weight (%)': '{:.2f}',
                        selected_outcome: '{:.2f}',
                        'Contribution': '{:.2f}'
                    }))
                with c2:
                    # enlarge combined plot
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
                    # Predictor balance
                    res['balance']['Difference'].plot.barh(ax=ax1)
                    ax1.set_title("Predictor Balance")
                    ax1.set_xlabel("Actual − Synthetic")
                    # Outcome comparison
                    ax2.bar(['Actual', 'Synthetic'],
                            [res['actual'], res['synth']],
                            color=['red', 'blue'])
                    ax2.set_title("Outcome Comparison")
                    ax2.set_ylabel(selected_outcome)
                    # KDE plot
                    sns.kdeplot(
                        data=df[df['cname'].isin(res['table']['Country'])][selected_outcome],
                        fill=True, ax=ax3
                    )
                    ax3.axvline(res['actual'], color='red', linestyle='--', label='Actual')
                    ax3.axvline(res['synth'], color='blue', linestyle='--', label='Synthetic')
                    ax3.set_title("Donor Outcome KDE")
                    ax3.legend()
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

        # PDF report
        buffer = io.BytesIO()
        with PdfPages(buffer) as pdf:
            for country, res in all_results.items():
                # Page 1: enlarged graphs
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
                res['balance']['Difference'].plot.barh(ax=ax1)
                ax1.set_title("Predictor Balance")
                ax2.bar(['Actual', 'Synthetic'], [res['actual'], res['synth']],
                        color=['red', 'blue'])
                ax2.set_title("Outcome Comparison")
                sns.kdeplot(
                    data=df[df['cname'].isin(res['table']['Country'])][selected_outcome],
                    fill=True, ax=ax3
                )
                ax3.axvline(res['actual'], color='red', linestyle='--')
                ax3.axvline(res['synth'], color='blue', linestyle='--')
                ax3.set_title("Donor Outcome KDE")
                plt.tight_layout()
                pdf.savefig(fig);
                plt.close(fig)

                # Page 2: donor table
                fig2, ax4 = plt.subplots(figsize=(10, res['table'].shape[0] * 0.3 + 1))
                ax4.axis('off')
                tbl = ax4.table(
                    cellText=res['table'].values,
                    colLabels=res['table'].columns,
                    cellLoc='center', loc='upper left'
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8)
                tbl.scale(1, 1.5)
                pdf.savefig(fig2);
                plt.close(fig2)

        buffer.seek(0)
        st.download_button(
            "📥 Download Full Report as PDF",
            data=buffer,
            file_name="synthetic_control_report.pdf",
            mime="application/pdf"
        )


if __name__ == "__main__":
    main()
