import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import zipfile
import os
from IPython.display import HTML, display


# Define the predictor variables and the outcome variable
predictors = [
    'gdppc_2',             # Gross Domestic Product per capita
    'hci_2',               # Human Capital Index
    'ado_fertility_2',     # Adolescent fertility rate
    'controlling_2',       # Control over resources and decisions
    'efindex_2',           # Economic Freedom Index
    'v2x_freexp_altinf_2', # Freedom of expression and alternative information index
    'v2xcl_rol_2',         # Rule of Law Index
    'v2x_cspart_2',        # Civil society participation index
    'v2cltrnslw_2',        # Transparent laws with predictable enforcement
    'v2clacjstw_2',        # Access to justice for women
    'v2cldiscw_2',         # Freedom from discrimination for women
    'v2cldmovew_2',        # Freedom of movement for women
    'v2cseeorgs_2',        # Freedom of assembly and association
    'v2csreprss_2',        # Repression of civil society
    #'v2csgender_2',        # Gender equality in civil society participation
    #'deaths_battle_2'      # Battle-related deaths (conflict intensity)
]
outcome = 'physical_v'  # Emotional violence rate (Gender-Based Violence Rate)

def prepare_data(donor_file, afg_file):
    """ Load and preprocess donor and treatment country data """
    donor_df = pd.read_csv('/content/donor_data_2_paper.csv')
    afg_df = pd.read_csv('/content/afghanistan_data_2_paper.csv')

    donor_predictors = donor_df.groupby('cname')[predictors].mean()
    donor_outcomes = donor_df.groupby('cname')[outcome].mean()

    afg_predictors = afg_df[predictors].mean().to_frame().T
    afg_outcome = afg_df[outcome].mean()

    donor_predictors.fillna(donor_predictors.mean(), inplace=True)
    donor_outcomes.fillna(donor_outcomes.mean(), inplace=True)
    afg_predictors.fillna(donor_predictors.mean(), inplace=True)

    return donor_predictors, donor_outcomes, afg_predictors, afg_outcome

def check_balance(donor_predictors, afg_predictors, weights):
    """Assess balance between Afghanistan and synthetic control."""
    synthetic_predictors = np.dot(weights, donor_predictors)
    balance_df = pd.DataFrame({
        'Predictor': predictors,
        'Actual Afghanistan': afg_predictors.values[0],
        'Synthetic Afghanistan': synthetic_predictors,
        'Difference': afg_predictors.values[0] - synthetic_predictors
    }).sort_values('Difference', ascending=False)

    print("\nPredictor Balance Check:")
    print(balance_df.round(3))

    return balance_df

def optimize_synthetic_control(donor_data, afg_data):
    """Find optimal weights for synthetic control."""
    scaler = StandardScaler()
    X_donor = scaler.fit_transform(donor_data)
    X_afg = scaler.transform(afg_data)

    n_donors = len(donor_data)

    def objective(weights):
        synthetic = np.dot(weights, X_donor)
        return np.sum((X_afg[0] - synthetic) ** 2)

    w0 = np.ones(n_donors) / n_donors
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, 1) for _ in range(n_donors)]

    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        print(f"Warning: Optimization failed - {result.message}")

    weights = result.x
    weights[weights < 0.01] = 0
    weights /= np.sum(weights)

    return weights

def plot_additional_results(balance_df, afg_outcome, synthetic_gbv):
    """Plot predictor balance and synthetic vs actual outcome."""
    plt.figure(figsize=(12, 5))

    # Predictor balance plot
    plt.subplot(1, 2, 1)
    plt.barh(balance_df['Predictor'], balance_df['Difference'], color='red')
    plt.xlabel('Standardized Difference')
    plt.title('Predictor Balance: Afghanistan vs. Synthetic Control')

    # Outcome comparison plot
    plt.subplot(1, 2, 2)
    plt.bar(['Actual Afghanistan', 'Synthetic Afghanistan'], [afg_outcome, synthetic_gbv], color=['red', 'blue'])
    plt.ylabel('GBV Rate - Physical Violence')
    plt.title('GBV Rate: Actual vs. Synthetic Control')

    plt.tight_layout()
    plt.savefig('/content/synthetic_control_balance.png')
    plt.show(block=False)
    plt.close()

def main():
    donor_file = 'donor_data_2_paper.csv'
    afg_file = 'afghanistan_data_2_paper.csv'

    donor_predictors, donor_outcomes, afg_predictors, afg_outcome = prepare_data(donor_file, afg_file)

    weights = optimize_synthetic_control(donor_predictors, afg_predictors)
    synthetic_gbv = np.dot(weights, donor_outcomes)

    balance_df = check_balance(donor_predictors, afg_predictors, weights)

    print(f"Estimated Sexual Violence for Afghanistan: {synthetic_gbv:.1f}%")
    print(f"Actual Sexual Violence rate for Afghanistan: {afg_outcome:.1f}%")

    # Create a DataFrame for donor weights and contributions
    results = pd.DataFrame({
        'Country': donor_predictors.index,
        'Weight (%)': weights * 100,
        'Sexual Violence': donor_outcomes,
        'Contribution to Synthetic GBV': weights * donor_outcomes
    }).sort_values('Weight (%)', ascending=False)

    # Create a summary DataFrame for synthetic and actual results
    summary_df = pd.DataFrame({
        'Country': ['Synthetic Afghanistan', 'Actual Afghanistan'],
        'Weight (%)': [None, None],
        'Sexual Violence': [synthetic_gbv, afg_outcome],
        'Contribution to Synthetic GBV': [None, None]
    })

    # Combine donor results with the summary rows
    results_all = pd.concat([results, summary_df], ignore_index=True)

    # Save the combined results to CSV
    csv_path = '/content/synthetic_control_results.csv'
    results_all.to_csv(csv_path, index=False)

    # Generate and save the plot
    plot_additional_results(balance_df, afg_outcome, synthetic_gbv)

    # Create a zip archive containing both the CSV and plot
    zip_filename = '/content/results.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(csv_path, arcname='synthetic_control_results.csv')
        zipf.write('/content/synthetic_control_balance.png', arcname='synthetic_control_balance.png')

    # Try to download automatically
    try:
        files.download(zip_filename)
        print("Download initiated automatically.")
    except Exception as e:
        print("Automatic download failed, please use the link below to download manually.")

    # Provide a manual download link if needed
    if os.path.exists(zip_filename):
        display(HTML(f'<a href="{zip_filename}" download>Click here to download results.zip</a>'))
    else:
        print("Error: results.zip not found.")

if __name__ == "__main__":
    main()























