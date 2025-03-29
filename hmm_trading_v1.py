import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta
import copy

from hmmlearn.hmm import GaussianHMM  # For Gaussian HMM modeling

# For evaluation metrics
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class HMMStockTrader:
    def __init__(self, 
                 ticker,
                 stock_name,
                 start_date,
                 end_date,
                 test_start_date,
                 train_window_size=100,
                 test_window_size=100,
                 sim_threshold=0.99,
                 further_calibrate=True,
                 em_iterations=1,
                 look_back_period=-1,
                 look_back_start_date=None,
                 training_mode=0,
                 further_calibration_recal=True,
                 n_shares=100):
        # Data & modeling parameters
        self.ticker = ticker
        self.stock_name = stock_name
        self.start_date = start_date
        self.end_date = end_date
        self.test_start_date = test_start_date
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.sim_threshold = sim_threshold
        self.further_calibrate = further_calibrate
        self.em_iterations = em_iterations
        self.look_back_period = look_back_period
        self.look_back_start_date = look_back_start_date
        self.training_mode = training_mode
        self.further_calibration_recal = further_calibration_recal
        self.n_shares = n_shares

    # ------------------------------
    # Data and Utility Methods
    # ------------------------------
    def get_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retrieve historical stock data (Open, High, Low, Close) from yfinance.
        """
        data = yf.download(ticker, start=start_date, end=end_date)
        return data[['Open', 'High', 'Low', 'Close']]

    def compute_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute log returns for all price columns:
            log_return = ln(current_price / previous_price)
        """
        log_returns = np.log(df / df.shift(1))
        return log_returns.dropna()

    def plot_predictions(self, results_df: pd.DataFrame):
        """
        Plot actual vs. predicted closing prices.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(results_df.index, results_df['Actual_Close'], label='Actual Close', linewidth=2)
        plt.plot(results_df.index, results_df['Naive_Pred_Close'], label='Naive Forecast', linestyle='--')
        plt.plot(results_df.index, results_df['HMM_Best_Pred_Close'], label='HMM Best Candidate', linestyle='-.')
        plt.plot(results_df.index, results_df['HMM_Voting_Pred_Close'], label='HMM Voting', linestyle=':')
        plt.title(f'{self.stock_name} Closing Price Forecasts')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def regression_evaluation(self, actual, naive, best, voting):
        """
        Compute regression evaluation metrics.
        """
        price_std = np.std(actual)
        actual_lr = np.log(actual / actual.shift(1)).dropna()
        log_return_std = np.std(actual_lr)

        r2_naive = r2_score(actual.iloc[1:], naive.iloc[1:])
        r2_best = r2_score(actual.iloc[1:], best.iloc[1:])
        r2_voting = r2_score(actual.iloc[1:], voting.iloc[1:])

        mape_naive = mean_absolute_percentage_error(actual.iloc[1:], naive.iloc[1:])
        mape_best = mean_absolute_percentage_error(actual.iloc[1:], best.iloc[1:])
        mape_voting = mean_absolute_percentage_error(actual.iloc[1:], voting.iloc[1:])

        return {
            'Price Std': price_std,
            'Log Return Std': log_return_std,
            'R2_Naive': r2_naive,
            'R2_Best': r2_best,
            'R2_Voting': r2_voting,
            'MAPE_Naive': mape_naive,
            'MAPE_Best': mape_best,
            'MAPE_Voting': mape_voting
        }

    def classification_evaluation(self, actual, naive, best, voting):
        """
        Compute classification evaluation metrics for directional predictions.
        """
        actual_lr = np.log(actual / actual.shift(1)).dropna()
        naive_lr = np.log(naive / naive.shift(1)).dropna()
        best_lr = np.log(best / best.shift(1)).dropna()
        voting_lr = np.log(voting / voting.shift(1)).dropna()

        actual_dir = (actual_lr > 0).astype(int)
        naive_dir = (naive_lr > 0).astype(int)
        best_dir = (best_lr > 0).astype(int)
        voting_dir = (voting_lr > 0).astype(int)

        common_idx = actual_dir.index.intersection(naive_dir.index).intersection(best_dir.index).intersection(voting_dir.index)
        actual_dir = actual_dir.loc[common_idx]
        naive_dir = naive_dir.loc[common_idx]
        best_dir = best_dir.loc[common_idx]
        voting_dir = voting_dir.loc[common_idx]

        try:
            auc_naive = roc_auc_score(actual_dir, naive_lr.loc[common_idx])
        except:
            auc_naive = np.nan
        try:
            auc_best = roc_auc_score(actual_dir, best_lr.loc[common_idx])
        except:
            auc_best = np.nan
        try:
            auc_voting = roc_auc_score(actual_dir, voting_lr.loc[common_idx])
        except:
            auc_voting = np.nan

        metrics = {}
        for label, pred in zip(['Naive', 'Best', 'Voting'], [naive_dir, best_dir, voting_dir]):
            metrics[label] = {
                'Accuracy': accuracy_score(actual_dir, pred),
                'Precision': precision_score(actual_dir, pred, zero_division=0),
                'Recall': recall_score(actual_dir, pred, zero_division=0),
                'F1': f1_score(actual_dir, pred, zero_division=0)
            }
        metrics['Naive']['AUC'] = auc_naive
        metrics['Best']['AUC'] = auc_best
        metrics['Voting']['AUC'] = auc_voting

        return metrics

    def plot_aic_bic(self, result_dict: dict):
        """
        Plot AIC and BIC values over the training windows.
        """
        x = range(len(result_dict['AIC']))

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(x, result_dict['AIC'], label='AIC', marker='o')
        plt.title(f'AIC over Training Windows for {self.stock_name}')
        plt.xlabel('Window Index')
        plt.ylabel('AIC')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x, result_dict['BIC'], label='BIC', marker='o', color='orange')
        plt.title(f'BIC over Training Windows for {self.stock_name}')
        plt.xlabel('Window Index')
        plt.ylabel('BIC')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # ------------------------------
    # Dynamic Calibration, Prediction & Evaluation
    # ------------------------------
    def train_and_predict_hmm_dynamic(self):
        """
        Train a 2-state HMM on rolling windows of the training set and choose the best candidate
        by BIC. Then, for each test window, further calibrate the static best model and predict.
        Returns prediction DataFrame, evaluation metrics, detailed summary DataFrame, and AIC/BIC dictionary.
        """
        # --- Load Data and Split ---
        price_df = self.get_stock_data(self.ticker, self.start_date, self.end_date)
        log_returns_df = self.compute_log_returns(price_df)
        price_df.index = pd.to_datetime(price_df.index)
        log_returns_df.index = pd.to_datetime(log_returns_df.index)

        test_start_dt = pd.to_datetime(self.test_start_date)
        train_log_returns = log_returns_df.loc[log_returns_df.index < test_start_dt]
        train_dates = train_log_returns.index
        train_values = train_log_returns.values
        n_train = train_values.shape[0]

        # Set look_back_start date (if not provided, use training start date)
        if self.look_back_start_date is None:
            look_back_start = train_dates[0]
        else:
            look_back_start = pd.to_datetime(self.look_back_start_date)

        eps = 1e-8
        k = 31  # for 2 states and 4 dims

        candidate_library = []
        aic_list = []
        bic_list = []
        prev_params = None

        print("=== TRAINING PHASE: Dynamic Calibration over Training Windows ===")
        if self.training_mode == 2:
            # Train a static model on entire training data.
            print("[TRAINING] Training static model on entire training period...")
            model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000,
                                min_covar=1e-3)
            model.fit(train_values)
            L = np.float64(model.score(train_values))
            static_model = model
            static_params = {
                'startprob': model.startprob_.copy(),
                'transmat': model.transmat_.copy(),
                'means': model.means_.copy(),
                'covars': model.covars_.copy()
            }
            candidate_library.append({
                'model': model,
                'L': L,
                'BIC': -2 * L + k * np.log(n_train + eps),
                'AIC': -2 * L + 2 * k,
                'next_lr': train_values[-1, 3],
                'window': train_values.copy(),
                'start_date': train_dates[0],
                'end_date': train_dates[-1],
                'params': copy.deepcopy(static_params)
            })
            aic_list.append(-2 * L + 2 * k)
            bic_list.append(-2 * L + k * np.log(n_train + eps))
        else:
            for i in range(n_train - self.train_window_size):
                window_candidate = train_values[i : i + self.train_window_size]
                window_start_date = train_dates[i]
                window_end_date = train_dates[i + self.train_window_size - 1]
                model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000,
                                    min_covar=1e-3)
                if self.training_mode == 0 and prev_params is not None:
                    model.startprob_ = prev_params['startprob'].copy()
                    model.transmat_ = prev_params['transmat'].copy()
                    model.means_ = prev_params['means'].copy()
                    covars = prev_params['covars'].copy()
                    for j in range(covars.shape[0]):
                        covars[j] = 0.5 * (covars[j] + covars[j].T)
                        min_eig = np.min(np.linalg.eigvalsh(covars[j]))
                        if min_eig < 1e-8:
                            covars[j] += np.eye(covars[j].shape[0]) * (1e-8 - min_eig)
                    model.covars_ = covars
                try:
                    model.fit(window_candidate)
                    L = np.float64(model.score(window_candidate))
                except:
                    L = np.nan
                try:
                    new_params = {
                        'startprob': model.startprob_.copy(),
                        'transmat': model.transmat_.copy(),
                        'means': model.means_.copy(),
                        'covars': model.covars_.copy()
                    }
                    for j in range(new_params['covars'].shape[0]):
                        new_params['covars'][j] = 0.5 * (new_params['covars'][j] + new_params['covars'][j].T)
                        min_eig = np.min(np.linalg.eigvalsh(new_params['covars'][j]))
                        if min_eig < 1e-8:
                            new_params['covars'][j] += np.eye(new_params['covars'][j].shape[0]) * (1e-8 - min_eig)
                    if self.training_mode == 0:
                        prev_params = new_params
                except:
                    if self.training_mode == 0:
                        prev_params = None

                if not np.isnan(L) and (i + self.train_window_size) < n_train:
                    bic_val = -2 * L + k * np.log(self.train_window_size + eps)
                    aic_val = -2 * L + 2 * k
                    next_lr = train_values[i + self.train_window_size, 3]
                    candidate_library.append({
                        'model': model,
                        'L': L,
                        'BIC': bic_val,
                        'AIC': aic_val,
                        'next_lr': next_lr,
                        'window': window_candidate.copy(),
                        'start_date': window_start_date,
                        'end_date': window_end_date,
                        'params': copy.deepcopy(new_params) if new_params is not None else None
                    })
                    aic_list.append(aic_val)
                    bic_list.append(bic_val)
                    print(f"[TRAINING] Window {i}: {window_start_date.date()} to {window_end_date.date()}, "
                          f"L: {L:.2f}, AIC: {aic_val:.2f}, BIC: {bic_val:.2f}")

        ic_dict = {'AIC': aic_list, 'BIC': bic_list}

        if candidate_library:
            if self.training_mode in [0, 1]:
                best_index = np.argmin([cand['BIC'] for cand in candidate_library])
                best_candidate = candidate_library[best_index]
                print(f"\n[TRAINING] Best BIC Model selected: Window {best_index} from "
                      f"{best_candidate['start_date'].date()} to {best_candidate['end_date'].date()} "
                      f"with BIC: {best_candidate['BIC']:.2f}")
                static_model = best_candidate['model']
                static_params = copy.deepcopy(best_candidate['params'])
            else:
                static_model = candidate_library[0]['model']
                static_params = copy.deepcopy(candidate_library[0]['params'])
        else:
            raise ValueError("No valid training candidate found.")

        library_ll = [cand['L'] for cand in candidate_library]
        library_next_lr = [cand['next_lr'] for cand in candidate_library]

        print("\n=== PREDICTION PHASE: Further Calibrating Static Best Model on Test Windows ===")
        all_log_returns = log_returns_df.values
        all_dates = log_returns_df.index

        start_idx = np.where(all_dates >= test_start_dt)[0][0]
        if start_idx < self.test_window_size:
            start_idx = self.test_window_size

        # Arrays for predictions and detailed summary
        dates_pred = []
        actual_close_prices = []
        naive_pred_prices = []
        best_candidate_pred_prices = []
        voting_pred_prices = []

        summary_data = {
            'Prediction Date': [],
            'Prediction Window Start': [],
            'Prediction Window End': [],
            'Best Candidate Window Start': [],
            'Best Candidate Window End': [],
            'Prediction Window Likelihood': [],
            'Best Candidate Window Likelihood': [],
            'Actual T-1': [],
            'Prediction': [],
            'Actual': [],
            'Prediction_direction': [],
            'Actual_direction': []
        }

        test_window_likelihoods = []
        best_candidate_likelihoods = []
        best_candidate_dates = []

        # Further calibration initialization: use static_params for the first prediction.
        pred_prev_params = copy.deepcopy(static_params)

        for t in range(start_idx, len(all_dates)):
            test_window = all_log_returns[t - self.test_window_size : t]
            pred_window_start = all_dates[t - self.test_window_size]
            pred_window_end = all_dates[t - 1]
            current_pred_date = all_dates[t]
            print(f"[PREDICTION] Test window for {current_pred_date.date()}: {pred_window_start.date()} to {pred_window_end.date()}")

            if self.further_calibrate:
                further_calibrated_model = copy.deepcopy(static_model)
                if pred_prev_params is not None:
                    further_calibrated_model.startprob_ = pred_prev_params['startprob'].copy()
                    further_calibrated_model.transmat_ = pred_prev_params['transmat'].copy()
                    further_calibrated_model.means_ = pred_prev_params['means'].copy()
                    covars = pred_prev_params['covars'].copy()
                    for j in range(covars.shape[0]):
                        covars[j] = 0.5 * (covars[j] + covars[j].T)
                        min_eig = np.min(np.linalg.eigvalsh(covars[j]))
                        if min_eig < 1e-8:
                            covars[j] += np.eye(covars[j].shape[0]) * (1e-8 - min_eig)
                    further_calibrated_model.covars_ = covars
                further_calibrated_model.n_iter = self.em_iterations
                try:
                    further_calibrated_model.fit(test_window)
                    L_current = np.float64(further_calibrated_model.score(test_window))
                except:
                    continue
                print(f"[PREDICTION] Further calibrated test window score: {L_current:.2f}")
                current_model = further_calibrated_model
                new_params = {
                    'startprob': further_calibrated_model.startprob_.copy(),
                    'transmat': further_calibrated_model.transmat_.copy(),
                    'means': further_calibrated_model.means_.copy(),
                    'covars': further_calibrated_model.covars_.copy()
                }
                for j in range(new_params['covars'].shape[0]):
                    new_params['covars'][j] = 0.5 * (new_params['covars'][j] + new_params['covars'][j].T)
                    min_eig = np.min(np.linalg.eigvalsh(new_params['covars'][j]))
                    if min_eig < 1e-8:
                        new_params['covars'][j] += np.eye(new_params['covars'][j].shape[0]) * (1e-8 - min_eig)
                pred_prev_params = new_params
            else:
                current_model = static_model
                try:
                    L_current = np.float64(static_model.score(test_window))
                except:
                    continue
                print(f"[PREDICTION] Using static model, test window score: {L_current:.2f}")

            test_window_likelihoods.append(L_current)

            # --- Look-back Filtering: First filter by look_back_start_date, then by look_back_period ---
            eligible_candidates_initial = [cand for cand in candidate_library if cand['end_date'] >= look_back_start]
            if self.look_back_period > 0:
                candidate_lb_date = current_pred_date - pd.Timedelta(days=self.look_back_period)
                eligible_candidates = [cand for cand in eligible_candidates_initial if cand['end_date'] >= candidate_lb_date]
                if len(eligible_candidates) == 0:
                    eligible_candidates = eligible_candidates_initial
            else:
                eligible_candidates = eligible_candidates_initial

            if eligible_candidates:
                sims = []
                for cand in eligible_candidates:
                    if self.further_calibration_recal:
                        try:
                            L_cand_updated = np.float64(current_model.score(cand['window']))
                        except:
                            L_cand_updated = np.nan
                    else:
                        L_cand_updated = cand['L']
                    sim = np.exp(-abs(L_current - L_cand_updated) / (abs(L_current) + 1e-8))
                    sims.append(sim)
                sims = np.array(sims)
                best_idx_local = np.argmax(sims)
                chosen_candidate = eligible_candidates[best_idx_local]
                best_lr = chosen_candidate['next_lr']
                best_cand_start = chosen_candidate['start_date']
                best_cand_end = chosen_candidate['end_date']
                if self.further_calibration_recal:
                    try:
                        best_cand_likelihood = np.float64(current_model.score(chosen_candidate['window']))
                    except:
                        best_cand_likelihood = np.nan
                else:
                    best_cand_likelihood = chosen_candidate['L']
                print(f"[PREDICTION] Best candidate (filtered) selected with similarity {sims[best_idx_local]:.2f}")
            else:
                updated_candidate_ll = []
                for cand in candidate_library:
                    if self.further_calibration_recal:
                        try:
                            L_cand_updated = np.float64(current_model.score(cand['window']))
                        except:
                            L_cand_updated = np.nan
                    else:
                        L_cand_updated = cand['L']
                    updated_candidate_ll.append(L_cand_updated)
                updated_candidate_ll = np.array(updated_candidate_ll)
                if len(updated_candidate_ll) > 0:
                    best_idx = np.argmax(updated_candidate_ll)
                    best_lr = library_next_lr[best_idx]
                    best_cand_start = candidate_library[best_idx]['start_date']
                    best_cand_end = candidate_library[best_idx]['end_date']
                    best_cand_likelihood = updated_candidate_ll[best_idx]
                else:
                    best_lr = np.nan
                    best_cand_start = None
                    best_cand_end = None
                    best_cand_likelihood = np.nan

            best_candidate_likelihoods.append(best_cand_likelihood)

            # HMM Voting using full library
            updated_candidate_ll_all = []
            for cand in candidate_library:
                if self.further_calibration_recal:
                    try:
                        L_cand_updated = np.float64(current_model.score(cand['window']))
                    except:
                        L_cand_updated = np.nan
                else:
                    L_cand_updated = cand['L']
                updated_candidate_ll_all.append(L_cand_updated)
            updated_candidate_ll_all = np.array(updated_candidate_ll_all)
            sims_all = []
            for L_cand in updated_candidate_ll_all:
                try:
                    sim = np.exp(-abs(L_current - L_cand)/(abs(L_current)+1e-8))
                except:
                    sim = 0
                sims_all.append(sim)
            sims_all = np.array(sims_all)
            candidate_indices = np.where(sims_all >= self.sim_threshold)[0]
            if len(candidate_indices) > 0:
                if len(candidate_indices) > 10:
                    sorted_indices = candidate_indices[np.argsort(sims_all[candidate_indices])][-10:]
                else:
                    sorted_indices = candidate_indices
                voting_lr_val = np.mean([library_next_lr[idx] for idx in sorted_indices])
            else:
                voting_lr_val = best_lr

            naive_lr = test_window[-1, 3]

            prev_date = all_dates[t-1]
            try:
                prev_close = price_df.loc[prev_date, 'Close']
                if isinstance(prev_close, pd.Series):
                    prev_close = prev_close.iloc[0]
            except:
                prev_close = np.nan

            naive_price = prev_close * np.exp(naive_lr)
            best_candidate_price = prev_close * np.exp(best_lr)
            voting_price = prev_close * np.exp(voting_lr_val)

            prediction_date = all_dates[t]
            try:
                actual_close = price_df.loc[prediction_date, 'Close']
                if isinstance(actual_close, pd.Series):
                    actual_close = actual_close.iloc[0]
            except:
                actual_close = np.nan

            try:
                prev_actual_close = price_df.loc[prev_date, 'Close']
                if isinstance(prev_actual_close, pd.Series):
                    prev_actual_close = prev_actual_close.iloc[0]
            except:
                prev_actual_close = np.nan

            pred_dir = 1 if (not np.isnan(prev_actual_close) and not np.isnan(best_candidate_price) and best_candidate_price > prev_actual_close) else 0
            act_dir = 1 if (not np.isnan(prev_actual_close) and not np.isnan(actual_close) and actual_close > prev_actual_close) else 0

            dates_pred.append(prediction_date)
            actual_close_prices.append(actual_close)
            naive_pred_prices.append(naive_price)
            best_candidate_pred_prices.append(best_candidate_price)
            voting_pred_prices.append(voting_price)

            summary_data['Prediction Date'].append(prediction_date)
            summary_data['Prediction Window Start'].append(pred_window_start)
            summary_data['Prediction Window End'].append(pred_window_end)
            summary_data['Best Candidate Window Start'].append(best_cand_start)
            summary_data['Best Candidate Window End'].append(best_cand_end)
            summary_data['Prediction Window Likelihood'].append(L_current)
            summary_data['Best Candidate Window Likelihood'].append(best_cand_likelihood)
            summary_data['Actual T-1'].append(prev_close)
            summary_data['Prediction'].append(best_candidate_price)
            summary_data['Actual'].append(actual_close)
            summary_data['Prediction_direction'].append(pred_dir)
            summary_data['Actual_direction'].append(act_dir)

            if t < len(all_dates) - 1:
                next_lr = all_log_returns[t, 3]
                new_candidate = {
                    'model': copy.deepcopy(current_model),
                    'L': L_current,
                    'BIC': -2 * L_current + k * np.log(self.test_window_size + eps),
                    'AIC': -2 * L_current + 2 * k,
                    'next_lr': next_lr,
                    'window': test_window.copy(),
                    'start_date': pred_window_start,
                    'end_date': pred_window_end,
                    'params': copy.deepcopy(pred_prev_params)
                }
                candidate_library.append(new_candidate)
                library_ll.append(L_current)
                library_next_lr.append(next_lr)
                print(f"[PREDICTION] New candidate added from {pred_window_start.date()} to {pred_window_end.date()}")

        pred_df = pd.DataFrame({
            'Date': dates_pred,
            'Actual_Close': actual_close_prices,
            'Naive_Pred_Close': naive_pred_prices,
            'HMM_Best_Pred_Close': best_candidate_pred_prices,
            'HMM_Voting_Pred_Close': voting_pred_prices
        })
        pred_df.set_index('Date', inplace=True)

        final_summary_df = pd.DataFrame(summary_data)

        reg_metrics = self.regression_evaluation(pred_df['Actual_Close'],
                                                 pred_df['Naive_Pred_Close'],
                                                 pred_df['HMM_Best_Pred_Close'],
                                                 pred_df['HMM_Voting_Pred_Close'])
        class_metrics = self.classification_evaluation(pred_df['Actual_Close'],
                                                       pred_df['Naive_Pred_Close'],
                                                       pred_df['HMM_Best_Pred_Close'],
                                                       pred_df['HMM_Voting_Pred_Close'])
        prediction_summary = {
            'Regression': reg_metrics,
            'Classification': class_metrics
        }

        return pred_df, prediction_summary, final_summary_df, ic_dict

    # ------------------------------
    # Fixed Shares Simulation Methods
    # ------------------------------
    def simulate_fixed_shares(self, results_df: pd.DataFrame, n_shares: int, mode: str):
        """
        Fixed Shares Trading Simulation.
        """
        actual = results_df['Actual_Close']
        naive_pred = results_df['Naive_Pred_Close']
        best_pred = results_df['HMM_Best_Pred_Close']
        voting_pred = results_df['HMM_Voting_Pred_Close']

        daily_profit = {'Naive': [], 'HMM_Best': [], 'HMM_Voting': []}
        trade_count = {'Naive': 0, 'HMM_Best': 0, 'HMM_Voting': 0}
        commission_per_roundtrip = 0.005 * 2 * n_shares

        for i in range(1, len(results_df)):
            entry_price = actual.iloc[i-1]
            exit_price = actual.iloc[i]
            # Naive simulation
            if mode == 'long_only':
                if naive_pred.iloc[i] > naive_pred.iloc[i-1]:
                    profit = (exit_price - entry_price) * n_shares - commission_per_roundtrip
                    trade_count['Naive'] += 1
                else:
                    profit = 0
            else:
                if naive_pred.iloc[i] > naive_pred.iloc[i-1]:
                    profit = (exit_price - entry_price) * n_shares - commission_per_roundtrip
                    trade_count['Naive'] += 1
                elif naive_pred.iloc[i] < naive_pred.iloc[i-1]:
                    profit = (entry_price - exit_price) * n_shares - commission_per_roundtrip
                    trade_count['Naive'] += 1
                else:
                    profit = 0
            daily_profit['Naive'].append(profit)

            # HMM Best simulation
            if mode == 'long_only':
                if best_pred.iloc[i] > best_pred.iloc[i-1]:
                    profit = (exit_price - entry_price) * n_shares - commission_per_roundtrip
                    trade_count['HMM_Best'] += 1
                else:
                    profit = 0
            else:
                if best_pred.iloc[i] > best_pred.iloc[i-1]:
                    profit = (exit_price - entry_price) * n_shares - commission_per_roundtrip
                    trade_count['HMM_Best'] += 1
                elif best_pred.iloc[i] < best_pred.iloc[i-1]:
                    profit = (entry_price - exit_price) * n_shares - commission_per_roundtrip
                    trade_count['HMM_Best'] += 1
                else:
                    profit = 0
            daily_profit['HMM_Best'].append(profit)

            # HMM Voting simulation
            if mode == 'long_only':
                if voting_pred.iloc[i] > voting_pred.iloc[i-1]:
                    profit = (exit_price - entry_price) * n_shares - commission_per_roundtrip
                    trade_count['HMM_Voting'] += 1
                else:
                    profit = 0
            else:
                if voting_pred.iloc[i] > voting_pred.iloc[i-1]:
                    profit = (exit_price - entry_price) * n_shares - commission_per_roundtrip
                    trade_count['HMM_Voting'] += 1
                elif voting_pred.iloc[i] < voting_pred.iloc[i-1]:
                    profit = (entry_price - exit_price) * n_shares - commission_per_roundtrip
                    trade_count['HMM_Voting'] += 1
                else:
                    profit = 0
            daily_profit['HMM_Voting'].append(profit)

        cum_profit = {}
        for method in ['Naive', 'HMM_Best', 'HMM_Voting']:
            profits = [0] + daily_profit[method]
            cum_profit[method] = np.cumsum(profits)

        holding_cum_profit = (actual - actual.iloc[0]) * n_shares

        summary = {}
        for method in ['Naive', 'HMM_Best', 'HMM_Voting']:
            initial_buy = actual.iloc[0] * n_shares
            total_trading_cost = trade_count[method] * commission_per_roundtrip
            total_investment = initial_buy + total_trading_cost
            total_profit = cum_profit[method][-1]
            pct_return = total_profit / total_investment if total_investment != 0 else np.nan
            summary[method] = {
                'Initial Buy Price': initial_buy,
                'Trading Cost': total_trading_cost,
                'Total Investment': total_investment,
                'Total Profit': total_profit,
                '% Return': pct_return,
                'Trade Count': trade_count[method]
            }
        summary['Holding'] = {
            'Initial Buy Price': actual.iloc[0] * n_shares,
            'Trading Cost': 0,
            'Total Investment': actual.iloc[0] * n_shares,
            'Total Profit': holding_cum_profit.iloc[-1],
            '% Return': (holding_cum_profit.iloc[-1]) / (actual.iloc[0] * n_shares) if actual.iloc[0] != 0 else np.nan,
            'Trade Count': 1
        }

        cum_profit_df = pd.DataFrame({
            'Naive': cum_profit['Naive'],
            'HMM_Best': cum_profit['HMM_Best'],
            'HMM_Voting': cum_profit['HMM_Voting'],
            'Holding': holding_cum_profit
        }, index=results_df.index)

        return cum_profit_df, summary

    def plot_cumulative_profit(self, sim_df: pd.DataFrame, title: str):
        """
        Plot cumulative profit/capital over time.
        """
        plt.figure(figsize=(14, 7))
        for col in sim_df.columns:
            plt.plot(sim_df.index, sim_df[col], label=col)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Profit / Capital')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------------------------
    # Main Runner Method
    # ------------------------------
    def run(self):
        """
        Execute the full workflow:
          - Train and predict using the HMM dynamic calibration
          - Plot AIC/BIC and predictions
          - Print evaluation metrics and summary tables
          - Run fixed shares simulation (both long only and long-short)
        """
        # Train and predict
        pred_df, eval_metrics, final_summary_df, ic_dict = self.train_and_predict_hmm_dynamic()

        # Plot AIC/BIC over calibrations
        self.plot_aic_bic(ic_dict)

        # Plot predictions
        self.plot_predictions(pred_df)

        # Print evaluation metrics
        print("\nPrediction Summary (Regression & Classification):")
        print("\nRegression Metrics:")
        for key, value in eval_metrics['Regression'].items():
            print(f"  {key}: {value}")
        print("\nClassification Metrics:")
        for method, metrics in eval_metrics['Classification'].items():
            print(f"\n{method} Forecast:")
            for metric, val in metrics.items():
                print(f"  {metric}: {val}")

        # Print final summary table
        print("\nFinal Summary Table:")
        print(final_summary_df)

        # Fixed Shares Simulation.
        sim_long_df, sim_long_summary = self.simulate_fixed_shares(pred_df, n_shares=self.n_shares, mode='long_only')
        sim_ls_df, sim_ls_summary = self.simulate_fixed_shares(pred_df, n_shares=self.n_shares, mode='long_short')

        print("\nFixed Shares Simulation - Long Only Summary:")
        for method, summ in sim_long_summary.items():
            print(f"\n{method}:")
            for k, v in summ.items():
                print(f"  {k}: {v}")

        print("\nFixed Shares Simulation - Long Short Summary:")
        for method, summ in sim_ls_summary.items():
            print(f"\n{method}:")
            for k, v in summ.items():
                print(f"  {k}: {v}")

        self.plot_cumulative_profit(sim_long_df, "Fixed Shares Simulation (Long Only) - Cumulative Profit")
        self.plot_cumulative_profit(sim_ls_df, "Fixed Shares Simulation (Long Short) - Cumulative Profit")


# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    trader = HMMStockTrader(
        ticker='SPY',
        stock_name='SPDR S&P 500 ETF Trust',
        start_date='2015-01-01',
        end_date='2017-08-11',
        test_start_date='2016-08-17',
        train_window_size=100,
        test_window_size=100,
        sim_threshold=0.99,
        further_calibrate=True,
        em_iterations=1,
        look_back_period=-1,
        look_back_start_date=None,
        training_mode=0,
        further_calibration_recal=True,
        n_shares=100
    )
    trader.run()
