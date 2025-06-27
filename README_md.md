# Portfolio Optimization Tool 📈

A comprehensive full-stack Python application for portfolio optimization using Modern Portfolio Theory. Built with Streamlit for an interactive web interface and powered by financial data analysis libraries.

## Features

### 🎯 Core Functionality
- **Stock Data Analysis**: Fetch real-time stock data using Yahoo Finance API
- **Portfolio Filtering**: Filter stocks by market capitalization and trading volume
- **Efficient Frontier**: Generate and visualize the efficient frontier curve
- **Portfolio Optimization**: Find optimal portfolios for maximum Sharpe ratio and minimum volatility
- **Interactive Visualizations**: Dynamic charts with Plotly for deep analysis
- **Report Generation**: Export analysis results as PDF reports and CSV data

### 📊 Analysis Features
- Market capitalization and volume-based filtering
- Historical return and risk analysis
- Covariance matrix calculations
- Sharpe ratio optimization
- Risk-return trade-off visualization
- Individual stock vs portfolio comparison

### 🖥️ User Interface
- Responsive web interface built with Streamlit
- File upload support for ticker lists
- Real-time progress tracking
- Interactive charts and tables
- Export functionality for all results

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the application files**
   ```bash
   mkdir portfolio_optimizer
   cd portfolio_optimizer
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

5. **Open your browser**
   The application will automatically open at `http://localhost:8501`

## Usage Guide

### 1. Input Stock Tickers
Choose one of two methods:
- **Manual Input**: Type stock tickers directly in the text area (one per line)
- **File Upload**: Upload a `.txt` file containing tickers

**Example tickers**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX

### 2. Set Filter Criteria
Configure the filtering parameters:
- **Market Capitalization**: Set minimum and maximum values in billions
- **Average Daily Volume**: Set minimum and maximum values in millions

### 3. Run Analysis
Click the "🚀 Run Analysis" button to:
- Fetch historical stock data
- Apply your filtering criteria
- Generate the efficient frontier
- Calculate optimal portfolios

### 4. Explore Results
Navigate through four main tabs:

#### 📈 Efficient Frontier
- Interactive plot showing risk-return trade-offs
- Highlighted optimal portfolios (Max Sharpe, Min Volatility)
- Individual stock positions for comparison

#### 📊 Portfolio Analysis
- Detailed metrics for optimal portfolios
- Portfolio composition pie charts
- Expected returns, volatility, and Sharpe ratios

#### 📋 Stock Details
- Comprehensive table of filtered stocks
- Market cap, volume, sector, and industry data
- CSV export functionality

#### 📄 Reports
- Generate comprehensive PDF reports
- Download interactive HTML charts
- Export all analysis results

## Understanding the Results

### Efficient Frontier
The efficient frontier represents the set of optimal portfolios offering the highest expected return for each level of risk. Points on the curve are mathematically optimal combinations.

### Key Metrics
- **Expected Return**: Annualized expected portfolio return
- **Volatility**: Annualized portfolio risk (standard deviation)
- **Sharpe Ratio**: Risk-adjusted return metric (return per unit of risk)

### Optimal Portfolios
- **Maximum Sharpe Ratio**: Best risk-adjusted returns
- **Minimum Volatility**: Lowest risk portfolio

### How Filtering Works
- **Market Cap Filtering**: Focuses on companies within specified size ranges
- **Volume Filtering**: Ensures adequate liquidity for trading
- **Combined Effect**: Creates a refined universe for optimization

## Technical Details

### Libraries Used
- **Streamlit**: Web application framework
- **yfinance**: Stock data retrieval
- **pandas/numpy**: Data manipulation and numerical computing
- **scipy**: Optimization algorithms
- **plotly**: Interactive visualizations
- **reportlab**: PDF report generation

### Optimization Algorithm
The application uses Modern Portfolio Theory with:
- Mean-variance optimization
- Quadratic programming for portfolio weights
- Constraint handling for portfolio sum and bounds
- Sharpe ratio maximization and risk minimization

### Data Sources
- Historical price data from Yahoo Finance
- Company fundamentals (market cap, volume, sector)
- 1-year historical period for return calculations

## Troubleshooting

### Common Issues

**"No valid tickers found"**
- Check ticker symbols are correct (use official exchange symbols)
- Ensure internet connection for data fetching
- Some tickers may be delisted or have limited data

**"No tickers meet criteria"**
- Adjust market cap and volume filters
- Check if filters are too restrictive
- Verify the tickers have sufficient market data

**Optimization fails**
- Ensure at least 2-3 valid tickers after filtering
- Check for sufficient historical data
- Try adjusting the time period or filters

### Performance Tips
- Use 5-20 tickers for optimal performance
- Larger ticker lists will take longer to process
- The application caches data during the session

## Deployment Options

### Local Development
```bash
streamlit run main.py
```

### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy directly from repository

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "main.py"]
```

## Educational Resources

### Modern Portfolio Theory
- Developed by Harry Markowitz in 1952
- Nobel Prize-winning approach to portfolio construction
- Balances expected returns against portfolio risk

### Key Concepts
- **Diversification**: Reduces risk through asset correlation
- **Risk-Return Trade-off**: Higher returns generally require higher risk
- **Efficient Markets**: Price information is reflected in stock prices

### Further Reading
- "Portfolio Selection" by Harry Markowitz
- "A Random Walk Down Wall Street" by Burton Malkiel
- Modern portfolio theory academic papers and resources

## License

This project is