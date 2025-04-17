import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import zipfile

# -------------------------
# Helper Functions
# -------------------------
@st.cache_data
def prepare_data(df):
    """
    Fill missing values in predictor and outcome columns using column means.
    Excludes 'cname' and 'year' columns from filling.
    """
    cols_to_fill = [col for col in df.columns if col not in ['cname', 'year']]
    for col in cols_to_fill:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
    return df

def optimize_synthetic_control(donor_data, treated_data):
    """
    Scale donor and treated predictor data and optimize weights for synthetic control.
    Returns weights constrained between 0 and 1, summing to 1.
    """
    scaler = StandardScaler()
    X_donor = scaler.fit_transform(donor_data)
    X_treated = scaler.transform(treated_data.values.reshape(1, -1))
    n = X_donor.shape[0]

    def objective(weights):
        synthetic = np.dot(weights, X_donor)
        return np.sum((X_treated[0] - synthetic) ** 2)

    w0 = np.ones(n) / n
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, 1)] * n
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        st.warning("Optimization failed: " + result.message)
    weights = result.x
    weights[weights < 0.01] = 0  # Set small weights to 0
    weights /= np.sum(weights)    # Normalize to sum to 1
    return weights

def check_balance(donor_predictors, treated_predictors, weights, predictors_list):
    """
    Compute synthetic predictors and return a balance DataFrame comparing actual vs. synthetic.
    """
    synthetic = np.dot(weights, donor_predictors[predictors_list].values)
    balance_df = pd.DataFrame({
        'Predictor': predictors_list,
        'Actual Treated': treated_predictors.values,
        'Synthetic': synthetic,
        'Difference': treated_predictors.values - synthetic
    })
    return balance_df

def generate_report_zip(results_df, fig_weights, fig_balance, fig_outcome):
    """
    Generate a ZIP archive containing results as CSV and plots as PNGs.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        zip_file.writestr("results.csv", csv_buffer.getvalue())
        for name, fig in zip(["weights", "balance", "outcome"], [fig_weights, fig_balance, fig_outcome]):
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            zip_file.writestr(f"{name}.png", buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.title("Synthetic Control Simulator")
    st.markdown("""
    **Instructions:**

    1. Upload a CSV file with columns: `cname`, `year`, an outcome variable, and predictor variables.
    2. Select the treated country.
    3. Select the outcome variable.
    4. Select predictor variables.
    5. Adjust predictor values for the treated country.
    6. Click **Run Simulation** to see the results.
    """)

    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = prepare_data(df)

        # Validate required column
        if 'cname' not in df.columns:
            st.error("CSV must contain a 'cname' column.")
            return

        # Select outcome
        candidate_outcomes = [col for col in df.columns if col not in ['cname', 'year']]
        selected_outcome = st.selectbox("Select Outcome", options=candidate_outcomes)

        # Select predictors
        candidate_predictors = [
            col for col in df.columns
            if col not in ['cname', 'year', selected_outcome]
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        selected_predictors = st.multiselect("Select Predictors", options=candidate_predictors, default=candidate_predictors)

        # Adjust treated country predictors
        st.markdown("### Adjust Treated Country Predictors")
        countries = df['cname'].unique()
        treated_country = st.selectbox("Select Treated Country", countries)

        treated_df = df[df['cname'] == treated_country]
        treated_vals = treated_df[selected_predictors].mean()
        adjusted_vals = {}
        for col in selected_predictors:
            default_val = float(treated_vals[col])
            min_val = float(treated_df[col].min())
            max_val = float(treated_df[col].max())
            adjusted_vals[col] = st.slider(f"Adjust {col}", min_value=min_val, max_value=max_val, value=default_val)

        # Run simulation
        if st.button("Run Simulation"):
            donor_predictors = df.groupby('cname')[selected_predictors].mean()
            donor_outcomes = df.groupby('cname')[selected_outcome].mean()
            treated_predictors = pd.Series(adjusted_vals)
            donor_pool = donor_predictors.drop(treated_country, errors='ignore')

            weights = optimize_synthetic_control(donor_pool, treated_predictors)
            synthetic_outcome = np.dot(weights, donor_outcomes.loc[donor_pool.index])

            # Display results
            st.markdown("### Results")
            st.write(f"**Synthetic outcome:** {synthetic_outcome:.2f}")
            st.write(f"**Actual outcome:** {treated_df[selected_outcome].mean():.2f}")

            # Predictor balance
            balance_df = check_balance(donor_pool, treated_predictors, weights, selected_predictors)
            st.markdown("#### Predictor Balance")
            st.dataframe(balance_df)

            # Plot donor weights
            fig1, ax1 = plt.subplots()
            ax1.bar(donor_pool.index, weights * 100)
            ax1.set_title("Donor Weights (%)")
            plt.xticks(rotation=45)
            st.pyplot(fig1)

            # Plot predictor balance
            fig2, ax2 = plt.subplots()
            ax2.barh(balance_df['Predictor'], balance_df['Difference'])
            ax2.set_title("Predictor Balance")
            st.pyplot(fig2)

            # Plot outcome comparison
            fig3, ax3 = plt.subplots()
            ax3.bar(['Synthetic', 'Actual'], [synthetic_outcome, treated_df[selected_outcome].mean()])
            ax3.set_title("Outcome Comparison")
            st.pyplot(fig3)

            # Prepare results for download
            results_all = donor_predictors.copy()
            results_all['Weight (%)'] = np.nan
            for country, w in zip(donor_pool.index, weights):
                results_all.loc[country, 'Weight (%)'] = w * 100
            summary = pd.DataFrame({
                'Country': [f"Synthetic {treated_country}", f"Actual {treated_country}"],
                'Outcome': [synthetic_outcome, treated_df[selected_outcome].mean()]
            })
            results_all = pd.concat([results_all.reset_index().rename(columns={'index': 'Country'}), summary], ignore_index=True)

            # Generate and provide download link for ZIP report
            zip_data = generate_report_zip(results_all, fig1, fig2, fig3)
            st.download_button(
                label="Download Report ZIP",
                data=zip_data,
                file_name="report.zip",
                mime="application/zip"
            )

if __name__ == '__main__':
    main()