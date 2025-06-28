import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PortfolioOptimizer:
    def __init__(self):
        self.data = None
        self.returns = None
        self.filtered_tickers = []
        
    def fetch_stock_data(self, tickers, period="1y"):
        """Fetch historical stock data and company info"""
        valid_tickers = []
        stock_data = {}
        company_info = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(tickers):
            try:
                status_text.text(f'Fetching data for {ticker}...')
                stock = yf.Ticker(ticker)
                
                # Get historical data
                hist = stock.history(period=period)
                if len(hist) > 0:
                    stock_data[ticker] = hist['Close']
                    
                    # Get company info
                    info = stock.info
                    company_info[ticker] = {
                        'market_cap': info.get('marketCap', 0) / 1e9,  # Convert to billions
                        'avg_volume': info.get('averageVolume', 0) / 1e6,  # Convert to millions
                        'sector': info.get('sector', 'N/A'),
                        'industry': info.get('industry', 'N/A')
                    }
                    valid_tickers.append(ticker)
                    
            except Exception as e:
                st.warning(f"Could not fetch data for {ticker}: {str(e)}")
                
            progress_bar.progress((i + 1) / len(tickers))
        
        status_text.empty()
        progress_bar.empty()
        
        if valid_tickers:
            self.data = pd.DataFrame(stock_data)
            self.returns = self.data.pct_change().dropna()
            
        return valid_tickers, company_info
    
    def filter_by_criteria(self, tickers, company_info, min_mcap, max_mcap, min_volume, max_volume):
        """Filter tickers based on market cap and volume criteria"""
        filtered = []
        filtered_info = {}
        
        for ticker in tickers:
            info = company_info[ticker]
            mcap = info['market_cap']
            volume = info['avg_volume']
            
            if (min_mcap <= mcap <= max_mcap and min_volume <= volume <= max_volume):
                filtered.append(ticker)
                filtered_info[ticker] = info
                
        self.filtered_tickers = filtered
        return filtered, filtered_info
    
    def calculate_portfolio_stats(self, weights, returns):
        """Calculate portfolio statistics"""
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def optimize_portfolio(self, target_return=None, minimize_risk=False):
        """Optimize portfolio for maximum Sharpe ratio or minimum risk"""
        if self.returns is None or len(self.filtered_tickers) == 0:
            return None
            
        returns = self.returns[self.filtered_tickers]
        n_assets = len(self.filtered_tickers)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        if target_return:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: np.sum(returns.mean() * x) * 252 - target_return
            })
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        if minimize_risk:
            # Minimize volatility
            def objective(weights):
                return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        else:
            # Maximize Sharpe ratio (minimize negative Sharpe)
            def objective(weights):
                ret, vol, sharpe = self.calculate_portfolio_stats(weights, returns)
                return -sharpe if vol > 0 else 1e6
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        return None
    
    def generate_efficient_frontier(self, n_portfolios=100):
        """Generate efficient frontier"""
        if self.returns is None or len(self.filtered_tickers) == 0:
            return None, None, None
            
        returns = self.returns[self.filtered_tickers]
        
        # Calculate range of target returns
        min_ret = returns.mean().min() * 252
        max_ret = returns.mean().max() * 252
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        efficient_portfolios = []
        for target_ret in target_returns:
            weights = self.optimize_portfolio(target_return=target_ret)
            if weights is not None:
                ret, vol, sharpe = self.calculate_portfolio_stats(weights, returns)
                efficient_portfolios.append({
                    'return': ret,
                    'volatility': vol,
                    'sharpe': sharpe,
                    'weights': weights
                })
        
        if not efficient_portfolios:
            return None, None, None
            
        # Find optimal portfolios
        max_sharpe_idx = max(range(len(efficient_portfolios)), 
                           key=lambda i: efficient_portfolios[i]['sharpe'])
        min_vol_idx = min(range(len(efficient_portfolios)), 
                         key=lambda i: efficient_portfolios[i]['volatility'])
        
        return efficient_portfolios, max_sharpe_idx, min_vol_idx

def create_efficient_frontier_plot(efficient_portfolios, max_sharpe_idx, min_vol_idx, filtered_tickers, returns):
    """Create interactive efficient frontier plot"""
    if not efficient_portfolios:
        return None
        
    # Extract data for plotting
    vols = [p['volatility'] for p in efficient_portfolios]
    rets = [p['return'] for p in efficient_portfolios]
    sharpes = [p['sharpe'] for p in efficient_portfolios]
    
    # Create figure
    fig = go.Figure()
    
    # Add efficient frontier
    fig.add_trace(go.Scatter(
        x=vols, y=rets,
        mode='lines+markers',
        name='Efficient Frontier',
        line=dict(color='blue', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Risk:</b> %{x:.2%}<br><b>Return:</b> %{y:.2%}<br><b>Sharpe:</b> %{customdata:.3f}<extra></extra>',
        customdata=sharpes
    ))
    
    # Add optimal portfolios
    fig.add_trace(go.Scatter(
        x=[vols[max_sharpe_idx]], y=[rets[max_sharpe_idx]],
        mode='markers',
        name='Max Sharpe Ratio',
        marker=dict(color='red', size=12, symbol='star'),
        hovertemplate='<b>Max Sharpe Portfolio</b><br><b>Risk:</b> %{x:.2%}<br><b>Return:</b> %{y:.2%}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[vols[min_vol_idx]], y=[rets[min_vol_idx]],
        mode='markers',
        name='Min Volatility',
        marker=dict(color='green', size=12, symbol='diamond'),
        hovertemplate='<b>Min Volatility Portfolio</b><br><b>Risk:</b> %{x:.2%}<br><b>Return:</b> %{y:.2%}<extra></extra>'
    ))
    
    # Add individual stocks
    if returns is not None and len(filtered_tickers) > 0:
        stock_returns = returns[filtered_tickers].mean() * 252
        stock_vols = returns[filtered_tickers].std() * np.sqrt(252)
        
        fig.add_trace(go.Scatter(
            x=stock_vols, y=stock_returns,
            mode='markers',
            name='Individual Stocks',
            marker=dict(color='lightblue', size=8),
            text=filtered_tickers,
            hovertemplate='<b>%{text}</b><br><b>Risk:</b> %{x:.2%}<br><b>Return:</b> %{y:.2%}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Efficient Frontier Analysis',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Expected Return',
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=600
    )
    
    fig.update_xaxes(tickformat='.1%')
    fig.update_yaxes(tickformat='.1%')
    
    return fig

def create_portfolio_composition_chart(weights, tickers, title):
    """Create portfolio composition pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=tickers,
        values=weights,
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.1%}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        template='plotly_white',
        width=400,
        height=400
    )
    
    return fig

def main():
    st.title("📈 Portfolio Optimization Tool")
    st.markdown("Optimize your investment portfolio using Modern Portfolio Theory")
    
    # Initialize optimizer
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = PortfolioOptimizer()
    
    # Sidebar for inputs
    st.sidebar.header("📊 Portfolio Configuration")
    
    # Ticker input options
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Manual Input", "Upload File"]
    )
    
    tickers = []
    if input_method == "Manual Input":
        ticker_input = st.sidebar.text_area(
            "Enter stock tickers (one per line):",
            value="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA\nNVDA\nMETA\nNFLX",
            height=150
        )
        tickers = [ticker.strip().upper() for ticker in ticker_input.split('\n') if ticker.strip()]
    else:
        uploaded_file = st.sidebar.file_uploader("Upload ticker file (.txt)", type=['txt'])
        if uploaded_file:
            tickers = [line.decode('utf-8').strip().upper() for line in uploaded_file.readlines() if line.strip()]
    
    # Filter criteria
    st.sidebar.subheader("🔍 Filter Criteria")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_mcap = st.number_input("Min Market Cap (B)", min_value=0.0, value=1.0, step=0.1)
        min_volume = st.number_input("Min Avg Volume (M)", min_value=0.0, value=1.0, step=0.1)
    
    with col2:
        max_mcap = st.number_input("Max Market Cap (B)", min_value=0.0, value=1000.0, step=1.0)
        max_volume = st.number_input("Max Avg Volume (M)", min_value=0.0, value=1000.0, step=1.0)
    
    # Analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        if not tickers:
            st.error("Please provide at least one ticker symbol")
            return
        
        with st.spinner("Fetching stock data and optimizing portfolio..."):
            # Fetch data
            valid_tickers, company_info = st.session_state.optimizer.fetch_stock_data(tickers)
            
            if not valid_tickers:
                st.error("No valid tickers found. Please check your input.")
                return
            
            # Filter tickers
            filtered_tickers, filtered_info = st.session_state.optimizer.filter_by_criteria(
                valid_tickers, company_info, min_mcap, max_mcap, min_volume, max_volume
            )
            
            if not filtered_tickers:
                st.error("No tickers meet the specified criteria. Please adjust your filters.")
                return
            
            st.success(f"Found {len(filtered_tickers)} stocks meeting your criteria!")
            
            # Store results in session state
            st.session_state.filtered_tickers = filtered_tickers
            st.session_state.filtered_info = filtered_info
            st.session_state.analysis_complete = True
    
    # Display results
    if getattr(st.session_state, 'analysis_complete', False):
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Efficient Frontier", "📊 Portfolio Analysis", "📋 Stock Details"])
        
        with tab1:
            st.header("Efficient Frontier Analysis")
            
            # Generate efficient frontier
            efficient_portfolios, max_sharpe_idx, min_vol_idx = st.session_state.optimizer.generate_efficient_frontier()
            
            if efficient_portfolios:
                # Create and display plot
                fig = create_efficient_frontier_plot(
                    efficient_portfolios, max_sharpe_idx, min_vol_idx, 
                    st.session_state.filtered_tickers, st.session_state.optimizer.returns
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Store for other tabs
                st.session_state.efficient_portfolios = efficient_portfolios
                st.session_state.max_sharpe_idx = max_sharpe_idx
                st.session_state.min_vol_idx = min_vol_idx
            else:
                st.error("Could not generate efficient frontier. Please try with different parameters.")
        
        with tab2:
            st.header("Optimal Portfolio Compositions")
            
            if hasattr(st.session_state, 'efficient_portfolios'):
                max_sharpe = st.session_state.efficient_portfolios[st.session_state.max_sharpe_idx]
                min_vol = st.session_state.efficient_portfolios[st.session_state.min_vol_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Maximum Sharpe Ratio Portfolio")
                    st.metric("Expected Return", f"{max_sharpe['return']:.2%}")
                    st.metric("Volatility", f"{max_sharpe['volatility']:.2%}")
                    st.metric("Sharpe Ratio", f"{max_sharpe['sharpe']:.3f}")
                    
                    # Portfolio composition chart
                    fig_sharpe = create_portfolio_composition_chart(
                        max_sharpe['weights'], st.session_state.filtered_tickers,
                        "Max Sharpe Portfolio Composition"
                    )
                    st.plotly_chart(fig_sharpe, use_container_width=True)
                
                with col2:
                    st.subheader("Minimum Volatility Portfolio")
                    st.metric("Expected Return", f"{min_vol['return']:.2%}")
                    st.metric("Volatility", f"{min_vol['volatility']:.2%}")
                    st.metric("Sharpe Ratio", f"{min_vol['sharpe']:.3f}")
                    
                    # Portfolio composition chart
                    fig_vol = create_portfolio_composition_chart(
                        min_vol['weights'], st.session_state.filtered_tickers,
                        "Min Volatility Portfolio Composition"
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)
        
        with tab3:
            st.header("Stock Analysis Details")
            
            # Create DataFrame for display
            df_display = pd.DataFrame.from_dict(st.session_state.filtered_info, orient='index')
            df_display.index.name = 'Ticker'
            df_display = df_display.reset_index()
            
            # Format columns
            df_display['market_cap'] = df_display['market_cap'].apply(lambda x: f"${x:.1f}B")
            df_display['avg_volume'] = df_display['avg_volume'].apply(lambda x: f"{x:.1f}M")
            
            df_display.columns = ['Ticker', 'Market Cap', 'Avg Volume', 'Sector', 'Industry']
            
            st.dataframe(df_display, use_container_width=True)

if __name__ == "__main__":
    main()