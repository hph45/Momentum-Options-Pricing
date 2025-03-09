import json
import math
import sys
from datetime import datetime
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm # type: ignore
from selenium import webdriver
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.preprocessing import PolynomialFeatures # type: ignore


class StockAnalyzer:
    """Class to analyze stock data and compute option prices."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.start_date = '2021-03-04'
        # self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.end_date = "2023-10-30"
        self.asset_class = 'stocks'

    def fetch_data(self) -> Dict:
        """Fetch historical stock data from Nasdaq API."""
        print(f"\rGathering data for {self.ticker} from Nasdaq's API...", end='')
        url = (
            f"https://api.nasdaq.com/api/quote/{self.ticker}/historical?"
            f"assetclass={self.asset_class}&fromdate={self.start_date}&"
            f"limit=9999&todate={self.end_date}&random=3"
        )
        
        driver = webdriver.Chrome()
        driver.get(url)
        raw_data = driver.page_source
        driver.quit()
        
        print("\rGathered data.                       ")
        return self._clean_data(raw_data)

    def _clean_data(self, raw_data: str) -> Dict:
        """Clean and parse raw HTML data into structured format."""
        print("Cleaning data...", end='')
        try:
            data_slice = raw_data[279 + len(self.ticker):-149]
            parsed_data = json.loads(data_slice)
        except json.JSONDecodeError:
            data_slice = raw_data[278 + len(self.ticker):-149]
            parsed_data = json.loads(data_slice)
            
        print("\rData cleaned              ")
        return {item['date']: item for item in parsed_data}

    def process_data(self, clean_data: Dict) -> Tuple[List, List, List, List]:
        """Process cleaned data into usable lists."""
        dates, opens, closes, volumes = [], [], [], []
        
        for date, data in clean_data.items():
            opens.insert(0, float(data['open'][1:].replace(',', '')))
            closes.insert(0, float(data['close'][1:].replace(',', '')))
            volumes.insert(0, float(data['volume'].replace(',', '')))
            dates.insert(0, date)
            
        return dates, opens, closes, volumes

    def fit_polynomial_model(self, dates: List[str], values: List[float]) -> Tuple[np.ndarray, float]:
        """Fit a 5th-degree polynomial model to the data."""
        print("\rFitting the model...", end='')
        
        date_objects = [datetime.strptime(d, '%m/%d/%Y') for d in dates]
        time_indices = np.array([(d - date_objects[0]).days for d in date_objects]).reshape(-1, 1)
        
        poly = PolynomialFeatures(degree=5, include_bias=False)
        time_poly = poly.fit_transform(time_indices)
        
        model = LinearRegression()
        model.fit(time_poly, values)
        fit_values = model.predict(time_poly)
        r2_score = model.score(time_poly, values)
        
        print("\rFit the model.       ")
        return fit_values, r2_score

    def calculate_momentum(self, prices: List[float]) -> List[float]:
        """Calculate momentum values based on price changes."""
        velocity = [10.0]  # Initial velocity in m/s
        
        for i in range(1, len(prices)):
            price_change = prices[i] - prices[i - 1]
            prev_velocity = velocity[-1]
            
            if price_change > 0:
                new_v = math.sqrt(max(0, 14 * (price_change - 0.2) + prev_velocity ** 2))
            elif price_change < 0:
                new_v = math.sqrt(max(0, prev_velocity ** 2 - 14 * (abs(price_change) + 0.2)))
            else:
                new_v = math.sqrt(max(0, prev_velocity ** 2 - 14 * 0.2))
                
            velocity.append(new_v)
        return velocity

    def black_scholes_call(self, prices: List[float], momentum: List[float], 
                         use_momentum: bool) -> Tuple[str, float, float, float, float, float]:
        """Calculate Black-Scholes-Merton call option price."""
        df_prices = pd.DataFrame(prices)
        current_price = float(df_prices.iloc[-1])
        
        # Adjust strike price
        strike_price = round(current_price, -1)
        if abs(strike_price - current_price) > 2.5:
            strike_price += 5 if current_price > strike_price else -5
            
        time_to_expiry = 0.25  # 3 months
        risk_free_rate = 0.0345  # 3.45% risk free interest rate
        
        # Calculate volatility
        daily_returns = df_prices.pct_change().dropna()
        annualized_volatility = float(daily_returns.std() * np.sqrt(252))
        
        if use_momentum:
            price_ratio = current_price / max(prices)
            momentum_ratio = momentum[-1] / max(momentum)
            annualized_volatility *= (momentum_ratio / price_ratio) ** 0.5
        
        # Black-Scholes calculation
        d1 = (np.log(current_price / strike_price) + 
              (risk_free_rate + 0.5 * annualized_volatility ** 2) * time_to_expiry) / \
             (annualized_volatility * np.sqrt(time_to_expiry))
        d2 = d1 - annualized_volatility * np.sqrt(time_to_expiry)
        
        call_price = float(
            current_price * norm.cdf(d1) - 
            strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        )
        
        return (self.ticker, current_price, strike_price, time_to_expiry, 
                risk_free_rate, call_price)

    def plot_results(self, dates: List[str], opens: List[float], volumes: List[float], 
                    fit_values: np.ndarray, momentum: List[float], r2: float, 
                    bsm: Tuple, bsmh: Tuple) -> None:
        """Create and display the analysis plot."""
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        # Plot price data
        date_objects = [datetime.strptime(d, '%m/%d/%Y') for d in dates]
        ax1.plot(date_objects, opens, 'b-', lw=0.8, label='Open Prices')
        ax1.plot(date_objects, fit_values, 'r-', lw=1, 
                label=f'Polynomial Fit (Degree 5), R² = {r2:.5f}')
        
        # Set up axes
        company_name = yf.Ticker(self.ticker).info['longName']
        ax1.set_title(f"{self.ticker} - {company_name}")
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price USD')
        ax1.set_xlim(date_objects[0], date_objects[-1])
        
        # Volume subplot
        ax2 = ax1.twinx()
        ax2.bar(date_objects, volumes, alpha=0.5, color='grey', label='Volume')
        ax2.set_yticks([])
        
        # Momentum subplot
        ax3 = ax1.twinx()
        ax3.plot(date_objects, momentum, 'g-', lw=1, 
                label=f'Momentum η\' Values, η\'final = {momentum[-1]:.5f}')
        ax3.set_ylabel('Momentum η\' kgm/s')
        
        # Add BSM/BSMH text
        text_str = (
            f"With {company_name}'s stock price S=${bsm[1]:.2f} as of {self.end_date},\n"
            f"Strike price K=${bsm[2]}, time period t={bsm[3]} years, "
            f"and interest rate r={bsm[4]*100:.1f}%,\n"
            f"Black-Scholes-Merton computes the European Call Option Price for "
            f"{bsm[0]} to ${bsm[5]:.2f}\n"
            f"Adjusting for momentum η, the price moves to ${bsmh[5]:.2f}"
        )
        plt.text(0.5, -0.2, text_str, ha='center', va='top', transform=ax1.transAxes)
        
        # Legend and layout
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        plt.tight_layout()
        print(f"{self.ticker} done.\n")
        plt.show()

def analyze_stock(ticker: str) -> None:
    """Main function to analyze a stock."""
    analyzer = StockAnalyzer(ticker)
    clean_data = analyzer.fetch_data()
    dates, opens, closes, volumes = analyzer.process_data(clean_data)
    fit_values, r2 = analyzer.fit_polynomial_model(dates, opens)
    momentum = analyzer.calculate_momentum(opens)
    bsm = analyzer.black_scholes_call(closes, momentum, False)
    bsmh = analyzer.black_scholes_call(closes, momentum, True)
    analyzer.plot_results(dates, opens, volumes, fit_values, momentum, r2, bsm, bsmh)

if __name__ == "__main__":
    # Example usage
    analyze_stock('meta')
    
    # S&P companies list (commented out)
    # sp_companies = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 
    #                 'PEP', 'AVGO', 'COST', 'CSCO', 'TMUS', 'ADBE', 'TXN', 
    #                 'CMCSA', 'AMD', 'NFLX', 'QCOM', 'INTC', 'SBUX']
    # for ticker in sp_companies:
    #     analyze_stock(ticker)
    