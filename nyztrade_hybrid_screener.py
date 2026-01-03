import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Page configuration
st.set_page_config(
    page_title="NYZTrade Hybrid Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with gradient theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d1b4e 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d1b4e 100%);
    }
    h1, h2, h3 {
        color: #a78bfa !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin: 10px 0;
    }
    .recommendation-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
    .buy { background-color: #10b981; color: white; }
    .hold { background-color: #f59e0b; color: white; }
    .sell { background-color: #ef4444; color: white; }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #a78bfa;
    }
    .stProgress > div > div {
        background-color: #7c3aed;
    }
</style>
""", unsafe_allow_html=True)

# Authentication
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets.get("passwords", {}) and \
           st.session_state["password"] == st.secrets["passwords"][st.session_state["username"]]:
            st.session_state["password_correct"] = True
            st.session_state["user"] = st.session_state["username"]
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<h1 style='text-align: center;'>🔐 NYZTrade Login</h1>", unsafe_allow_html=True)
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password", on_change=password_entered)
            if st.button("Login", use_container_width=True):
                password_entered()
        return False
    elif not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<h1 style='text-align: center;'>🔐 NYZTrade Login</h1>", unsafe_allow_html=True)
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password", on_change=password_entered)
            st.error("😕 User not known or password incorrect")
            if st.button("Login", use_container_width=True):
                password_entered()
        return False
    else:
        return True

# Industry benchmarks for PE ratios and EV/EBITDA
INDUSTRY_BENCHMARKS = {
    "Technology": {"pe": 28, "ev_ebitda": 18},
    "Financial Services": {"pe": 15, "ev_ebitda": 10},
    "Consumer Cyclical": {"pe": 20, "ev_ebitda": 12},
    "Healthcare": {"pe": 25, "ev_ebitda": 15},
    "Industrials": {"pe": 18, "ev_ebitda": 11},
    "Energy": {"pe": 12, "ev_ebitda": 8},
    "Basic Materials": {"pe": 14, "ev_ebitda": 9},
    "Consumer Defensive": {"pe": 22, "ev_ebitda": 13},
    "Communication Services": {"pe": 16, "ev_ebitda": 10},
    "Real Estate": {"pe": 20, "ev_ebitda": 14},
    "Utilities": {"pe": 16, "ev_ebitda": 9},
    "Default": {"pe": 20, "ev_ebitda": 12}
}

MIDCAP_BENCHMARKS = {
    "Technology": {"pe": 25, "ev_ebitda": 16},
    "Financial Services": {"pe": 14, "ev_ebitda": 9},
    "Consumer Cyclical": {"pe": 18, "ev_ebitda": 11},
    "Healthcare": {"pe": 22, "ev_ebitda": 14},
    "Industrials": {"pe": 16, "ev_ebitda": 10},
    "Energy": {"pe": 11, "ev_ebitda": 7},
    "Basic Materials": {"pe": 13, "ev_ebitda": 8},
    "Consumer Defensive": {"pe": 20, "ev_ebitda": 12},
    "Communication Services": {"pe": 15, "ev_ebitda": 9},
    "Real Estate": {"pe": 18, "ev_ebitda": 13},
    "Utilities": {"pe": 15, "ev_ebitda": 8},
    "Default": {"pe": 18, "ev_ebitda": 11}
}

SMALLCAP_BENCHMARKS = {
    "Technology": {"pe": 22, "ev_ebitda": 14},
    "Financial Services": {"pe": 12, "ev_ebitda": 8},
    "Consumer Cyclical": {"pe": 16, "ev_ebitda": 10},
    "Healthcare": {"pe": 20, "ev_ebitda": 12},
    "Industrials": {"pe": 15, "ev_ebitda": 9},
    "Energy": {"pe": 10, "ev_ebitda": 6},
    "Basic Materials": {"pe": 12, "ev_ebitda": 7},
    "Consumer Defensive": {"pe": 18, "ev_ebitda": 11},
    "Communication Services": {"pe": 14, "ev_ebitda": 8},
    "Real Estate": {"pe": 16, "ev_ebitda": 12},
    "Utilities": {"pe": 14, "ev_ebitda": 7},
    "Default": {"pe": 16, "ev_ebitda": 10}
}

# Database path
DB_PATH = "stocks_database.db"

def retry_on_failure(func, max_retries=3, delay=1):
    """Retry a function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay * (2 ** attempt))
    return None

def fetch_stock_data(ticker, max_retries=3):
    """Fetch stock data with retry logic and rate limiting"""
    def _fetch():
        stock = yf.Ticker(ticker)
        info = stock.info
        time.sleep(0.3)  # Rate limiting
        return info
    
    try:
        return retry_on_failure(_fetch, max_retries)
    except Exception as e:
        return None

def safe_float(value, default=None):
    """Safely convert value to float"""
    if value is None or value == '' or pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=None):
    """Safely convert value to int"""
    if value is None or value == '' or pd.isna(value):
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def calculate_valuations(stock_data, market_cap_category):
    """Calculate fair value using multiple methods"""
    try:
        # Get industry
        industry = stock_data.get('industry', 'Default')
        if industry not in INDUSTRY_BENCHMARKS:
            industry = 'Default'
        
        # Select benchmarks based on market cap
        if market_cap_category == "Large Cap":
            benchmarks = INDUSTRY_BENCHMARKS[industry]
        elif market_cap_category == "Mid Cap":
            benchmarks = MIDCAP_BENCHMARKS[industry]
        else:
            benchmarks = SMALLCAP_BENCHMARKS[industry]
        
        current_price = safe_float(stock_data.get('currentPrice'))
        if not current_price:
            return None, None, None, None
        
        # PE Multiple Method
        trailing_pe = safe_float(stock_data.get('trailingPE'))
        forward_pe = safe_float(stock_data.get('forwardPE'))
        eps = safe_float(stock_data.get('trailingEps'))
        
        pe_fair_value = None
        if eps and eps > 0:
            if trailing_pe:
                historical_fair_pe = trailing_pe * 0.9
                target_pe = (historical_fair_pe * 0.7) + (benchmarks['pe'] * 0.3)
                pe_fair_value = eps * target_pe
            elif forward_pe:
                target_pe = (forward_pe * 0.7) + (benchmarks['pe'] * 0.3)
                pe_fair_value = eps * target_pe
        
        # EV/EBITDA Method
        enterprise_value = safe_float(stock_data.get('enterpriseValue'))
        ebitda = safe_float(stock_data.get('ebitda'))
        market_cap = safe_float(stock_data.get('marketCap'))
        
        ev_ebitda_fair_value = None
        if ebitda and ebitda > 0 and enterprise_value and market_cap:
            current_ev_ebitda = enterprise_value / ebitda
            target_ev_ebitda = (current_ev_ebitda * 0.5) + (benchmarks['ev_ebitda'] * 0.5)
            fair_enterprise_value = ebitda * target_ev_ebitda
            ev_to_mcap_ratio = market_cap / enterprise_value
            ev_ebitda_fair_value = fair_enterprise_value * ev_to_mcap_ratio
        
        # Calculate average fair value
        fair_values = [v for v in [pe_fair_value, ev_ebitda_fair_value] if v is not None]
        if not fair_values:
            return None, None, None, None
        
        avg_fair_value = sum(fair_values) / len(fair_values)
        upside_potential = ((avg_fair_value - current_price) / current_price) * 100
        
        # Calculate individual upsides
        pe_upside = ((pe_fair_value - current_price) / current_price * 100) if pe_fair_value else None
        ev_upside = ((ev_ebitda_fair_value - current_price) / current_price * 100) if ev_ebitda_fair_value else None
        
        return avg_fair_value, upside_potential, pe_upside, ev_upside
    
    except Exception as e:
        return None, None, None, None

def get_market_cap_category(market_cap):
    """Categorize stock by market cap"""
    if market_cap >= 100000_00_00_000:  # >= 1 Lakh Crore
        return "Large Cap"
    elif market_cap >= 25000_00_00_000:  # >= 25k Crore
        return "Mid Cap"
    elif market_cap >= 5000_00_00_000:  # >= 5k Crore
        return "Small Cap"
    else:
        return "Micro Cap"

def create_database():
    """Create SQLite database for storing stock data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            ticker TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL,
            market_cap REAL,
            pe_ratio REAL,
            forward_pe REAL,
            eps REAL,
            pb_ratio REAL,
            dividend_yield REAL,
            beta REAL,
            roe REAL,
            profit_margin REAL,
            revenue REAL,
            ebitda REAL,
            enterprise_value REAL,
            total_debt REAL,
            total_cash REAL,
            shares_outstanding REAL,
            week_52_high REAL,
            week_52_low REAL,
            sector TEXT,
            industry TEXT,
            last_updated TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def update_database(stock_dict, progress_callback=None):
    """Update database with current stock data"""
    create_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    total_stocks = sum(len(stocks) for stocks in stock_dict.values())
    processed = 0
    failed_tickers = []
    
    for category, stocks in stock_dict.items():
        for ticker, name in stocks.items():
            try:
                stock_data = fetch_stock_data(ticker)
                if stock_data:
                    cursor.execute('''
                        INSERT OR REPLACE INTO stocks VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    ''', (
                        ticker,
                        name,
                        category,
                        safe_float(stock_data.get('currentPrice')),
                        safe_float(stock_data.get('marketCap')),
                        safe_float(stock_data.get('trailingPE')),
                        safe_float(stock_data.get('forwardPE')),
                        safe_float(stock_data.get('trailingEps')),
                        safe_float(stock_data.get('priceToBook')),
                        safe_float(stock_data.get('dividendYield')),
                        safe_float(stock_data.get('beta')),
                        safe_float(stock_data.get('returnOnEquity')),
                        safe_float(stock_data.get('profitMargins')),
                        safe_float(stock_data.get('totalRevenue')),
                        safe_float(stock_data.get('ebitda')),
                        safe_float(stock_data.get('enterpriseValue')),
                        safe_float(stock_data.get('totalDebt')),
                        safe_float(stock_data.get('totalCash')),
                        safe_float(stock_data.get('sharesOutstanding')),
                        safe_float(stock_data.get('fiftyTwoWeekHigh')),
                        safe_float(stock_data.get('fiftyTwoWeekLow')),
                        stock_data.get('sector', 'N/A'),
                        stock_data.get('industry', 'N/A'),
                        datetime.now().isoformat()
                    ))
                    conn.commit()
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                failed_tickers.append(ticker)
            
            processed += 1
            if progress_callback:
                progress_callback(processed, total_stocks, failed_tickers)
    
    conn.close()
    return failed_tickers

@st.cache_data(ttl=3600)
def load_database():
    """Load stock data from database with proper type conversion"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM stocks", conn)
        conn.close()
        
        # Convert numeric columns explicitly
        numeric_columns = [
            'price', 'market_cap', 'pe_ratio', 'forward_pe', 'eps', 'pb_ratio',
            'dividend_yield', 'beta', 'roe', 'profit_margin', 'revenue', 'ebitda',
            'enterprise_value', 'total_debt', 'total_cash', 'shares_outstanding',
            'week_52_high', 'week_52_low'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return pd.DataFrame()

def check_database_exists():
    """Check if database exists and is not empty"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stocks")
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0, count
    except:
        return False, 0

def parallel_fetch_stocks(stock_list, max_workers=15):
    """Fetch stock data in parallel with progress bar"""
    results = []
    failed = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(stock_list)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stock = {
            executor.submit(fetch_stock_data, ticker): (ticker, name, category)
            for category, stocks in stock_list.items()
            for ticker, name in stocks.items()
        }
        
        for future in as_completed(future_to_stock):
            ticker, name, category = future_to_stock[future]
            completed += 1
            
            try:
                stock_data = future.result()
                if stock_data:
                    results.append({
                        'ticker': ticker,
                        'name': name,
                        'category': category,
                        'data': stock_data
                    })
                else:
                    failed.append(ticker)
            except Exception as e:
                failed.append(ticker)
            
            progress_bar.progress(completed / total)
            status_text.text(f"Processed {completed}/{total} stocks ({len(failed)} failed)")
    
    progress_bar.empty()
    status_text.empty()
    
    return results, failed

def screen_from_database(df, criteria):
    """Screen stocks from database based on criteria"""
    filtered_df = df.copy()
    
    # Apply filters
    if criteria.get('categories'):
        filtered_df = filtered_df[filtered_df['category'].isin(criteria['categories'])]
    
    if criteria.get('market_cap_category'):
        if criteria['market_cap_category'] == "Large Cap":
            filtered_df = filtered_df[filtered_df['market_cap'] >= 100000_00_00_000]
        elif criteria['market_cap_category'] == "Mid Cap":
            filtered_df = filtered_df[
                (filtered_df['market_cap'] >= 25000_00_00_000) & 
                (filtered_df['market_cap'] < 100000_00_00_000)
            ]
        elif criteria['market_cap_category'] == "Small Cap":
            filtered_df = filtered_df[
                (filtered_df['market_cap'] >= 5000_00_00_000) & 
                (filtered_df['market_cap'] < 25000_00_00_000)
            ]
        elif criteria['market_cap_category'] == "Micro Cap":
            filtered_df = filtered_df[filtered_df['market_cap'] < 5000_00_00_000]
    
    if criteria.get('min_price'):
        filtered_df = filtered_df[filtered_df['price'] >= criteria['min_price']]
    
    if criteria.get('max_price'):
        filtered_df = filtered_df[filtered_df['price'] <= criteria['max_price']]
    
    if criteria.get('max_pe'):
        filtered_df = filtered_df[
            (filtered_df['pe_ratio'] <= criteria['max_pe']) & 
            (filtered_df['pe_ratio'] > 0)
        ]
    
    return filtered_df

def calculate_valuations_batch(df):
    """Calculate valuations for a batch of stocks from database"""
    results = []
    
    for idx, row in df.iterrows():
        # Reconstruct stock_data dictionary from row
        stock_data = {
            'currentPrice': row['price'],
            'marketCap': row['market_cap'],
            'trailingPE': row['pe_ratio'],
            'forwardPE': row['forward_pe'],
            'trailingEps': row['eps'],
            'enterpriseValue': row['enterprise_value'],
            'ebitda': row['ebitda'],
            'industry': row['industry']
        }
        
        market_cap_cat = get_market_cap_category(row['market_cap']) if pd.notna(row['market_cap']) else "Micro Cap"
        fair_value, upside, pe_upside, ev_upside = calculate_valuations(stock_data, market_cap_cat)
        
        if fair_value is not None:
            results.append({
                'Ticker': row['ticker'],
                'Name': row['name'],
                'Category': row['category'],
                'Current Price': row['price'],
                'Fair Value': fair_value,
                'Upside %': upside,
                'PE Upside %': pe_upside,
                'EV/EBITDA Upside %': ev_upside,
                'Market Cap': row['market_cap'],
                'Market Cap Category': market_cap_cat,
                'PE Ratio': row['pe_ratio'],
                '52W High': row['week_52_high'],
                '52W Low': row['week_52_low']
            })
    
    return pd.DataFrame(results)

# Preset screener functions
def get_preset_screeners():
    """Define preset screeners"""
    return {
        "🚀 Top 50 Undervalued Large Caps": {
            "market_cap_category": "Large Cap",
            "valuation_type": "Undervalued",
            "min_upside": 20,
            "limit": 50
        },
        "📈 Top 50 Undervalued Mid Caps": {
            "market_cap_category": "Mid Cap",
            "valuation_type": "Undervalued",
            "min_upside": 25,
            "limit": 50
        },
        "💎 Top 50 Undervalued Small Caps": {
            "market_cap_category": "Small Cap",
            "valuation_type": "Undervalued",
            "min_upside": 30,
            "limit": 50
        },
        "⚠️ Top 50 Overvalued Large Caps": {
            "market_cap_category": "Large Cap",
            "valuation_type": "Overvalued",
            "max_upside": -10,
            "limit": 50
        },
        "💰 Top 50 Value Stocks (PE < 15)": {
            "max_pe": 15,
            "valuation_type": "Undervalued",
            "min_upside": 15,
            "limit": 50
        },
        "🎯 Undervalued Near 52W High": {
            "near_52w_high": True,
            "valuation_type": "Undervalued",
            "min_upside": 20,
            "limit": 50
        }
    }

def create_gauge_chart(value, title, range_max=100):
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#a78bfa'}},
        number={'suffix': "%", 'font': {'size': 32, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [None, range_max], 'tickwidth': 1, 'tickcolor': "#a78bfa"},
            'bar': {'color': "#7c3aed"},
            'bgcolor': "rgba(30, 30, 46, 0.5)",
            'borderwidth': 2,
            'bordercolor': "#a78bfa",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [25, 50], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [50, range_max], 'color': 'rgba(16, 185, 129, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(30, 30, 46, 0.2)",
        plot_bgcolor="rgba(30, 30, 46, 0.2)",
        font={'color': "#ffffff", 'family': "Arial"},
        height=300
    )
    
    return fig

def create_bar_chart(current, fair, ticker):
    """Create a bar chart comparing current and fair value"""
    fig = go.Figure(data=[
        go.Bar(name='Current Price', x=['Price'], y=[current], marker_color='#ef4444'),
        go.Bar(name='Fair Value', x=['Price'], y=[fair], marker_color='#10b981')
    ])
    
    fig.update_layout(
        title=f"{ticker} - Price Comparison",
        yaxis_title="Price (₹)",
        barmode='group',
        paper_bgcolor="rgba(30, 30, 46, 0.2)",
        plot_bgcolor="rgba(30, 30, 46, 0.2)",
        font={'color': "#ffffff"},
        title_font_color="#a78bfa",
        height=400
    )
    
    return fig

def generate_pdf_report(stock_info):
    """Generate PDF report for a stock"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#7c3aed'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph(f"Stock Analysis Report: {stock_info['ticker']}", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Company Info
    data = [
        ['Company Name', stock_info['name']],
        ['Ticker', stock_info['ticker']],
        ['Category', stock_info['category']],
        ['Sector', stock_info.get('sector', 'N/A')],
        ['Industry', stock_info.get('industry', 'N/A')],
    ]
    
    t = Table(data, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#7c3aed')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(t)
    story.append(Spacer(1, 0.3*inch))
    
    # Valuation Metrics
    story.append(Paragraph("Valuation Analysis", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    valuation_data = [
        ['Metric', 'Value'],
        ['Current Price', f"₹{stock_info.get('current_price', 0):.2f}"],
        ['Fair Value', f"₹{stock_info.get('fair_value', 0):.2f}"],
        ['Upside Potential', f"{stock_info.get('upside', 0):.2f}%"],
        ['Market Cap Category', stock_info.get('market_cap_cat', 'N/A')],
        ['PE Ratio', f"{stock_info.get('pe_ratio', 0):.2f}"],
        ['Recommendation', stock_info.get('recommendation', 'N/A')],
    ]
    
    t2 = Table(valuation_data, colWidths=[2.5*inch, 3.5*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7c3aed')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(t2)
    story.append(Spacer(1, 0.5*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph(
        "Disclaimer: This report is for educational purposes only. Not investment advice. "
        "Please consult a financial advisor before making investment decisions.",
        disclaimer_style
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Main application
def main():
    if not check_password():
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 Account")
        st.info(f"User: {st.session_state['user'].title()}")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Database Status
        st.markdown("### 📊 Database Status")
        db_exists, stock_count = check_database_exists()
        
        if db_exists:
            st.success(f"✅ Database: {stock_count} stocks")
            
            # Show last updated time
            try:
                df = load_database()
                if not df.empty and 'last_updated' in df.columns:
                    last_updated = pd.to_datetime(df['last_updated']).max()
                    st.info(f"🕒 Updated: {last_updated.strftime('%Y-%m-%d %H:%M')}")
            except:
                pass
        else:
            st.warning("⚠️ No database found")
            st.info("👉 Update database to start screening")
        
        st.markdown("---")
        
        # Database Management
        st.markdown("### ⚙️ Database Management")
        
        # Calculate total stocks that will be updated
        total_to_update = sum(len(stocks) for stocks in INDIAN_STOCKS.values())
        st.info(f"📌 Will update {total_to_update} stocks")
        
        if st.button("🔄 Update Database Now", use_container_width=True):
            st.session_state['show_update_confirmation'] = True
        
        if st.session_state.get('show_update_confirmation', False):
            st.warning("⚠️ This may take 30-60 minutes")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm", use_container_width=True):
                    st.session_state['show_update_confirmation'] = False
                    st.session_state['updating_database'] = True
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state['show_update_confirmation'] = False
                    st.rerun()
    
    # Handle database update
    if st.session_state.get('updating_database', False):
        st.markdown("## 🔄 Updating Database...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        failed_text = st.empty()
        
        def progress_callback(processed, total, failed):
            progress_bar.progress(processed / total)
            status_text.text(f"Processed: {processed}/{total}")
            if failed:
                failed_text.text(f"Failed: {len(failed)} stocks")
        
        failed_tickers = update_database(INDIAN_STOCKS, progress_callback)
        
        st.session_state['updating_database'] = False
        st.success(f"✅ Database updated! ({len(failed_tickers)} failed)")
        
        if failed_tickers:
            with st.expander("View Failed Tickers"):
                st.write(failed_tickers)
        
        # Clear cache to reload new data
        st.cache_data.clear()
        
        time.sleep(2)
        st.rerun()
    
    # Main content
    st.markdown("<h1 style='text-align: center;'>📊 NYZTrade Hybrid Stock Screener</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a78bfa;'>⚡ Instant Database Screening + 🔍 Real-time Custom Analysis</p>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["⚡ Instant Presets", "🔍 Custom Real-time", "📈 Individual Analysis"])
    
    # Tab 1: Instant Presets
    with tab1:
        st.markdown("### ⚡ INSTANT RESULTS - Powered by cached database")
        
        if not db_exists:
            st.warning("📊 Please update the database first using the sidebar button")
        else:
            preset_screeners = get_preset_screeners()
            
            st.markdown("### 🎯 Select a Preset Screener")
            selected_preset = st.selectbox(
                "Choose Screener",
                list(preset_screeners.keys()),
                label_visibility="collapsed"
            )
            
            if st.button("🚀 Run Instant Screening", use_container_width=True, type="primary"):
                with st.spinner("Screening database..."):
                    df = load_database()
                    
                    if df.empty:
                        st.error("Database is empty. Please update it first.")
                    else:
                        criteria = preset_screeners[selected_preset]
                        
                        # Filter based on criteria
                        filtered_df = screen_from_database(df, {
                            'market_cap_category': criteria.get('market_cap_category'),
                            'max_pe': criteria.get('max_pe')
                        })
                        
                        # Calculate valuations
                        results_df = calculate_valuations_batch(filtered_df)
                        
                        # Apply valuation filters
                        if criteria.get('valuation_type') == 'Undervalued':
                            results_df = results_df[results_df['Upside %'] > 0]
                        elif criteria.get('valuation_type') == 'Overvalued':
                            results_df = results_df[results_df['Upside %'] < 0]
                        
                        if criteria.get('min_upside'):
                            results_df = results_df[results_df['Upside %'] >= criteria['min_upside']]
                        
                        if criteria.get('max_upside'):
                            results_df = results_df[results_df['Upside %'] <= criteria['max_upside']]
                        
                        if criteria.get('near_52w_high'):
                            results_df['Distance from 52W High %'] = (
                                (results_df['52W High'] - results_df['Current Price']) / 
                                results_df['52W High'] * 100
                            )
                            results_df = results_df[results_df['Distance from 52W High %'] <= 10]
                        
                        # Sort and limit
                        results_df = results_df.nlargest(criteria.get('limit', 50), 'Upside %')
                        
                        st.success(f"✅ Found {len(results_df)} stocks matching criteria")
                        
                        # Display results
                        if not results_df.empty:
                            display_df = results_df[[
                                'Ticker', 'Name', 'Category', 'Current Price', 
                                'Fair Value', 'Upside %', 'Market Cap Category', 'PE Ratio'
                            ]].copy()
                            
                            # Format display
                            display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"₹{x:.2f}")
                            display_df['Fair Value'] = display_df['Fair Value'].apply(lambda x: f"₹{x:.2f}")
                            display_df['Upside %'] = display_df['Upside %'].apply(lambda x: f"{x:.2f}%")
                            display_df['PE Ratio'] = display_df['PE Ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                            
                            st.dataframe(display_df, use_container_width=True, height=400)
                            
                            # Download option
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results (CSV)",
                                data=csv,
                                file_name=f"{selected_preset.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
    
    # Tab 2: Custom Real-time
    with tab2:
        st.markdown("### 🔍 Custom Real-time Screening")
        st.info("💡 Fetches live data from yfinance - takes 10-30 seconds depending on filters")
        
        with st.form("custom_screening_form"):
            st.markdown("#### 🎯 Market & Price Filters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                categories = st.multiselect(
                    "Select Categories (3-5 recommended)",
                    list(INDIAN_STOCKS.keys()),
                    max_selections=10
                )
                
                market_cap_filter = st.selectbox(
                    "Market Cap Category",
                    ["All", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap"]
                )
            
            with col2:
                price_range = st.slider(
                    "Price Range (₹)",
                    0, 10000, (0, 10000), step=100
                )
                
                max_pe_filter = st.slider(
                    "Maximum PE Ratio",
                    0, 100, 50, step=5
                )
            
            st.markdown("#### 📊 Valuation Filters")
            
            col3, col4 = st.columns(2)
            
            with col3:
                valuation_type = st.selectbox(
                    "Valuation Type",
                    ["All", "Undervalued (>0%)", "Overvalued (<0%)"]
                )
                
                upside_range = st.slider(
                    "Upside Range (%)",
                    -50, 100, (-50, 100), step=5
                )
            
            with col4:
                near_high = st.checkbox("Near 52W High (within 10%)")
                
                max_stocks = st.select_slider(
                    "Max Stocks to Fetch",
                    options=[50, 100, 200, 300, 500, 1000],
                    value=300
                )
            
            # Estimate time
            estimated_time = max_stocks * 0.3 / 15  # Parallel with 15 workers
            st.info(f"⏱️ Estimated time: {estimated_time:.0f} seconds")
            
            submit_button = st.form_submit_button("🔍 Search Stocks", use_container_width=True, type="primary")
        
        if submit_button:
            if not categories:
                st.error("Please select at least one category")
            else:
                # Build stock list
                stock_list = {}
                for cat in categories:
                    stock_list[cat] = INDIAN_STOCKS[cat]
                
                # Limit total stocks
                total_stocks = sum(len(stocks) for stocks in stock_list.values())
                if total_stocks > max_stocks:
                    st.warning(f"Limiting to first {max_stocks} stocks from selected categories")
                    # TODO: Implement proper limiting logic
                
                with st.spinner(f"Fetching data for {min(total_stocks, max_stocks)} stocks..."):
                    results, failed = parallel_fetch_stocks(stock_list)
                    
                    if results:
                        # Process results
                        screened_stocks = []
                        
                        for result in results:
                            stock_data = result['data']
                            
                            # Apply price filter
                            current_price = safe_float(stock_data.get('currentPrice'))
                            if not current_price or current_price < price_range[0] or current_price > price_range[1]:
                                continue
                            
                            # Apply PE filter
                            pe_ratio = safe_float(stock_data.get('trailingPE'))
                            if pe_ratio and pe_ratio > max_pe_filter:
                                continue
                            
                            # Apply market cap filter
                            market_cap = safe_float(stock_data.get('marketCap'))
                            if market_cap:
                                market_cap_cat = get_market_cap_category(market_cap)
                                if market_cap_filter != "All" and market_cap_cat != market_cap_filter:
                                    continue
                            else:
                                market_cap_cat = "N/A"
                            
                            # Calculate valuation
                            fair_value, upside, pe_upside, ev_upside = calculate_valuations(stock_data, market_cap_cat)
                            
                            if fair_value is None:
                                continue
                            
                            # Apply valuation filters
                            if valuation_type == "Undervalued (>0%)" and upside <= 0:
                                continue
                            elif valuation_type == "Overvalued (<0%)" and upside >= 0:
                                continue
                            
                            if upside < upside_range[0] or upside > upside_range[1]:
                                continue
                            
                            # Apply 52W high filter
                            if near_high:
                                week_52_high = safe_float(stock_data.get('fiftyTwoWeekHigh'))
                                if week_52_high:
                                    distance = ((week_52_high - current_price) / week_52_high) * 100
                                    if distance > 10:
                                        continue
                            
                            screened_stocks.append({
                                'Ticker': result['ticker'],
                                'Name': result['name'],
                                'Category': result['category'],
                                'Current Price': current_price,
                                'Fair Value': fair_value,
                                'Upside %': upside,
                                'PE Upside %': pe_upside,
                                'EV/EBITDA Upside %': ev_upside,
                                'Market Cap Category': market_cap_cat,
                                'PE Ratio': pe_ratio
                            })
                        
                        if screened_stocks:
                            results_df = pd.DataFrame(screened_stocks)
                            results_df = results_df.sort_values('Upside %', ascending=False)
                            
                            st.success(f"✅ Found {len(results_df)} stocks matching criteria ({len(failed)} failed to fetch)")
                            
                            # Display results
                            display_df = results_df[[
                                'Ticker', 'Name', 'Category', 'Current Price',
                                'Fair Value', 'Upside %', 'Market Cap Category', 'PE Ratio'
                            ]].copy()
                            
                            # Format display
                            display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"₹{x:.2f}")
                            display_df['Fair Value'] = display_df['Fair Value'].apply(lambda x: f"₹{x:.2f}")
                            display_df['Upside %'] = display_df['Upside %'].apply(lambda x: f"{x:.2f}%")
                            display_df['PE Ratio'] = display_df['PE Ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                            
                            st.dataframe(display_df, use_container_width=True, height=400)
                            
                            # Download option
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results (CSV)",
                                data=csv,
                                file_name=f"custom_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.warning("No stocks matched your criteria")
                    else:
                        st.error("Failed to fetch stock data")
    
    # Tab 3: Individual Analysis
    with tab3:
        st.markdown("### 📈 Individual Stock Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ticker_input = st.text_input(
                "Enter Stock Ticker (e.g., RELIANCE.NS)",
                placeholder="TICKER.NS"
            )
        
        with col2:
            analyze_button = st.button("📊 Analyze Stock", use_container_width=True, type="primary")
        
        if analyze_button and ticker_input:
            with st.spinner(f"Analyzing {ticker_input}..."):
                stock_data = fetch_stock_data(ticker_input)
                
                if stock_data:
                    current_price = safe_float(stock_data.get('currentPrice'))
                    market_cap = safe_float(stock_data.get('marketCap'))
                    
                    if current_price and market_cap:
                        market_cap_cat = get_market_cap_category(market_cap)
                        fair_value, upside, pe_upside, ev_upside = calculate_valuations(stock_data, market_cap_cat)
                        
                        if fair_value:
                            # Display metrics
                            st.markdown(f"### {stock_data.get('longName', ticker_input)}")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Current Price", f"₹{current_price:.2f}")
                            with col2:
                                st.metric("Fair Value", f"₹{fair_value:.2f}")
                            with col3:
                                st.metric("Upside Potential", f"{upside:.2f}%")
                            with col4:
                                if upside > 25:
                                    recommendation = "🚀 Highly Undervalued"
                                    badge_class = "buy"
                                elif upside > 15:
                                    recommendation = "✅ Undervalued"
                                    badge_class = "buy"
                                elif upside > 0:
                                    recommendation = "📥 Fairly Valued"
                                    badge_class = "hold"
                                elif upside > -10:
                                    recommendation = "⏸️ HOLD"
                                    badge_class = "hold"
                                else:
                                    recommendation = "⚠️ Overvalued"
                                    badge_class = "sell"
                                
                                st.markdown(f"<div class='recommendation-badge {badge_class}'>{recommendation}</div>", 
                                          unsafe_allow_html=True)
                            
                            # Charts
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if pe_upside is not None:
                                    st.plotly_chart(create_gauge_chart(pe_upside, "PE Multiple Upside"), use_container_width=True)
                            
                            with col2:
                                if ev_upside is not None:
                                    st.plotly_chart(create_gauge_chart(ev_upside, "EV/EBITDA Upside"), use_container_width=True)
                            
                            # Price comparison chart
                            st.plotly_chart(create_bar_chart(current_price, fair_value, ticker_input), use_container_width=True)
                            
                            # Additional info
                            with st.expander("📋 Detailed Information"):
                                info_df = pd.DataFrame({
                                    'Metric': [
                                        'Sector', 'Industry', 'Market Cap', 'Market Cap Category',
                                        'PE Ratio', 'Forward PE', 'PB Ratio', 'Dividend Yield',
                                        'Beta', 'ROE', '52W High', '52W Low'
                                    ],
                                    'Value': [
                                        stock_data.get('sector', 'N/A'),
                                        stock_data.get('industry', 'N/A'),
                                        f"₹{market_cap/10000000:.2f} Cr",
                                        market_cap_cat,
                                        f"{stock_data.get('trailingPE', 0):.2f}",
                                        f"{stock_data.get('forwardPE', 0):.2f}",
                                        f"{stock_data.get('priceToBook', 0):.2f}",
                                        f"{(stock_data.get('dividendYield', 0) * 100):.2f}%",
                                        f"{stock_data.get('beta', 0):.2f}",
                                        f"{(stock_data.get('returnOnEquity', 0) * 100):.2f}%",
                                        f"₹{stock_data.get('fiftyTwoWeekHigh', 0):.2f}",
                                        f"₹{stock_data.get('fiftyTwoWeekLow', 0):.2f}"
                                    ]
                                })
                                st.dataframe(info_df, use_container_width=True, hide_index=True)
                            
                            # PDF Report
                            if st.button("📄 Generate PDF Report", use_container_width=True):
                                pdf_data = {
                                    'ticker': ticker_input,
                                    'name': stock_data.get('longName', ticker_input),
                                    'category': 'N/A',
                                    'sector': stock_data.get('sector', 'N/A'),
                                    'industry': stock_data.get('industry', 'N/A'),
                                    'current_price': current_price,
                                    'fair_value': fair_value,
                                    'upside': upside,
                                    'market_cap_cat': market_cap_cat,
                                    'pe_ratio': stock_data.get('trailingPE', 0),
                                    'recommendation': recommendation
                                }
                                
                                pdf_buffer = generate_pdf_report(pdf_data)
                                st.download_button(
                                    label="📥 Download PDF Report",
                                    data=pdf_buffer,
                                    file_name=f"{ticker_input}_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                        else:
                            st.error("Could not calculate valuation for this stock")
                    else:
                        st.error("Could not fetch price or market cap data")
                else:
                    st.error("Stock not found or data unavailable")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #6b7280;'>"
        "Disclaimer: This tool is for educational purposes only. Not financial advice. "
        "Please consult a financial advisor before making investment decisions."
        "</p>",
        unsafe_allow_html=True
    )

# INDIAN_STOCKS dictionary - USER MUST POPULATE THIS
INDIAN_STOCKS = {
    # Example structure - user needs to populate with actual data
    "Example Category": {
        "RELIANCE.NS": "Reliance Industries Limited",
        "TCS.NS": "Tata Consultancy Services Limited",
        # Add more stocks here...
    },
    # Add more categories and stocks...
}

if __name__ == "__main__":
    main()
