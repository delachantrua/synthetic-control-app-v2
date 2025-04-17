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
    """
    Coerce all non‐cname/year columns to numeric, then fill NaNs
    with the mean of that numeric column only.
    """
    for col in df.columns:
        if col in ('cname', 'year'):
            continue
        # convert to numeric, unparseable → NaN
        numeric = pd.to_numeric(df[col], errors='coerce')
        # compute mean of numeric values
        mean_val = numeric.mean()
        # fill NaNs with that mean
        df[col] = numeric.fillna(mean_val)
    return df

def optimize_synthetic_control(donor_data, treated_data):
    # (unchanged from before)
    donor_data = donor_data.select_dtypes(include=[np.number])
    n_donors, n_preds = donor_data.shape
    if n_donors == 0 or n_preds != len(treated_data):
        return np.ones(n_donors) / max(n_donors, 1)
    try:
        scaler  = StandardScaler()
        Xd      = scaler.fit_transform(donor_data)
        Xt      = scaler.transform(treated_data.values.reshape(1,-1))[0]
    except:
        return np.ones(n_donors) / n_donors
    w0     = np.ones(n_donors)/n_donors
    bounds = [(0,1)]*n_donors
    cons   = ({'type':'eq','fun': lambda w: w.sum()-1},)
    def obj(w): return np.sum((Xt - w @ Xd)**2)
    try:
        sol = minimize(obj, w0, method='SLSQP',
                       bounds=bounds, constraints=cons)
        w = sol.x if sol.success else w0
    except:
        w = w0
    w[w<0.01] = 0
    return w/w.sum() if w.sum()>0 else w

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="Synthetic Control", layout="wide")
    st.title("Synthetic Control Simulator")

    with st.expander("📋 Instructions", expanded=True):
        st.markdown("""
        1. Upload a CSV with `cname`, `year`, plus only numeric columns.  
        2. Pick one outcome and one or more predictors.  
        3. Select your treated country(ies).  
        4. Tweak predictors via sliders.  
        5. Run to view balance plots, outcome vs synthetic, KDE, donor table, and download PDF.
        """)

    if 'vals' not in st.session_state:
        st.session_state.vals = {}

    uploaded = st.file_uploader("CSV file", type="csv")
    if not uploaded:
        return

    df = prepare_data(pd.read_csv(uploaded))
    if 'cname' not in df.columns:
        st.error("CSV must include a 'cname' column.")
        return

    cols    = [c for c in df.columns if c not in ('cname','year')]
    outcome = st.selectbox("Outcome variable", cols)
    preds   = st.multiselect("Predictors", [c for c in cols if c != outcome],
                              default=[c for c in cols if c != outcome])
    countries = df['cname'].unique().tolist()
    treated   = st.multiselect("Treated country(ies)", countries,
                                default=[countries[0]])

    for c in treated:
        st.subheader(c)
        sub   = df[df['cname']==c]
        means = sub[preds].mean()
        adj   = {}
        for p in preds:
            mn, mx = float(sub[p].min()), float(sub[p].max())
            dv = float(means[p])
            if mn == mx:
                eps = abs(dv)*0.01 or 0.01
                mn, mx = dv-eps, dv+eps
            key = f"{c}_{p}"
            if key not in st.session_state.vals:
                st.session_state.vals[key] = dv
            val = st.slider(p, mn, mx,
                            st.session_state.vals[key],
                            key=key)
            st.session_state.vals[key] = val
            adj[p] = val
        st.session_state.vals[c] = adj

    if st.button("Run Simulation"):
        results = {}
        for c in treated:
            tr   = pd.Series(st.session_state.vals[c])
            dpre = df.groupby('cname')[preds].mean()
            dout = df.groupby('cname')[outcome].mean()
            pool = dpre.drop(treated, errors='ignore')

            w     = optimize_synthetic_control(pool, tr)
            synth = float(w @ dout.loc[pool.index].values)
            actual= float(df[df['cname']==c][outcome].mean())

            bal_vals = w @ pool.values
            bal_df   = pd.DataFrame({'Actual':tr.values,'Synthetic':bal_vals},
                                     index=pool.columns)
            bal_df['Diff'] = bal_df['Actual'] - bal_df['Synthetic']

            dt = pd.DataFrame({
                'Country':      pool.index,
                'Weight (%)':   w*100,
                outcome:        dout.loc[pool.index].values,
                'Contribution': w * dout.loc[pool.index].values
            }).sort_values('Weight (%)', ascending=False)
            summ = pd.DataFrame([
                {'Country':'Synthetic','Weight (%)':np.nan,
                 outcome:synth,'Contribution':np.nan},
                {'Country':'Actual',   'Weight (%)':np.nan,
                 outcome:actual,'Contribution':np.nan}
            ])
            full_table = pd.concat([dt, summ], ignore_index=True)

            results[c] = {
                'actual': actual,
                'synth':  synth,
                'balance': bal_df,
                'table':   full_table
            }

        st.header("Results")
        for c,res in results.items():
            with st.expander(c, expanded=True):
                col1,col2 = st.columns(2)
                with col1:
                    st.metric("Actual",    f"{res['actual']:.2f}")
                    st.metric("Synthetic", f"{res['synth']:.2f}")
                    st.metric("Difference",f"{res['synth']-res['actual']:.2f}")
                    st.dataframe(
                        res['table'].style.format({
                            'Weight (%)':'{:.2f}',
                            outcome:'{:.2f}',
                            'Contribution':'{:.2f}'
                        })
                    )
                with col2:
                    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(24,8))
                    res['balance']['Diff'].plot.barh(ax=ax1)
                    ax1.set_title("Predictor Balance")
                    ax1.set_xlabel("Actual − Synthetic")
                    ax2.bar(['Actual','Synth'],[res['actual'],res['synth']],
                            color=['red','blue'])
                    ax2.set_title("Outcome Comparison")
                    ax2.set_ylabel(outcome)
                    sns.kdeplot(
                        data=df[df['cname'].isin(res['table']['Country'])][outcome],
                        fill=True, ax=ax3
                    )
                    ax3.axvline(res['actual'], color='red', linestyle='--')
                    ax3.axvline(res['synth'],  color='blue', linestyle='--')
                    ax3.set_title("Donor Outcome KDE")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            for c,res in results.items():
                fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(24,8))
                res['balance']['Diff'].plot.barh(ax=ax1)
                ax1.set_title("Predictor Balance")
                ax2.bar(['Actual','Synth'],[res['actual'],res['synth']],
                        color=['red','blue'])
                ax2.set_title("Outcome Comparison")
                sns.kdeplot(
                    data=df[df['cname'].isin(res['table']['Country'])][outcome],
                    fill=True, ax=ax3
                )
                ax3.axvline(res['actual'], color='red', linestyle='--')
                ax3.axvline(res['synth'],  color='blue', linestyle='--')
                ax3.set_title("Donor Outcome KDE")
                plt.tight_layout()
                pdf.savefig(fig); plt.close(fig)

                fig2, ax4 = plt.subplots(
                    figsize=(10, res['table'].shape[0]*0.3+1))
                ax4.axis('off')
                tbl = ax4.table(
                    cellText=res['table'].values,
                    colLabels=res['table'].columns,
                    loc='upper left'
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8)
                tbl.scale(1,1.5)
                pdf.savefig(fig2); plt.close(fig2)

        buf.seek(0)
        st.download_button("📥 Download PDF report",
                           data=buf,
                           file_name="synthetic_control_report.pdf",
                           mime="application/pdf")

if __name__ == "__main__":
    main()
