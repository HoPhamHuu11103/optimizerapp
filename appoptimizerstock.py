
import streamlit as st
import pandas as pd
import numpy as np
from vnstock import Vnstock
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
import random
import streamlit as st
import pandas as pd
import datetime
import threading
# Hàm tạo màu ngẫu nhiên
def random_color():
    return "#%06x" % random.randint(0, 0xFFFFFF)


# Cấu hình trang với sidebar mở rộng
st.set_page_config(
    page_title="Portfolio Optimization Dashboard 📈", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Thêm CSS tùy chỉnh với Google Font (Poppins), background gradient, hiệu ứng hiện đại cho các thành phần
st.markdown(
    """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    /* Tổng thể background với gradient mềm mại */
    body {
        background: linear-gradient(135deg, #f6f9fc, #e9eff5);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header chính */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #333333;
        text-align: center;
        padding: 1.5rem 0;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        position: relative;
    }
    
    /* Sub-header */
    .sub-header {
        font-size: 1.25rem;
        color: #555555;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Style cho các tab */
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333333;
        border-radius: 8px;
        transition: transform 0.3s ease, background-color 0.3s ease, color 0.3s ease;
        padding: 0.5rem 1rem;
    }
    /* Hover cho tab: scale nhẹ, nền chuyển sang màu pastel đỏ nhạt và chữ chuyển thành màu đỏ đậm */
    .stTabs [data-baseweb="tab"]:hover {
        transform: scale(1.05);
        background-color: #ffebee;
        color: #d32f2f;
    }
    /* Tab đang active */
    .stTabs [data-baseweb="tab"] > div[role="button"][aria-selected="true"] {
        background-color: #d32f2f !important;
        color: #ffffff !important;
        border-radius: 8px;
    }
    
    /* Cải tiến cho sidebar (nếu có) */
    .css-1d391kg {  
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Thêm padding cho container chính */
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header chính (đã bỏ dòng chèn sticker)
st.markdown(
    """
    <div class="main-header">
        Portfolio Optimization Dashboard
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Ứng dụng tích hợp quy trình: tải dữ liệu cổ phiếu, xử lý, tối ưu hóa danh mục đầu tư (SLSQP, SGD, SGD - Sharpe), so sánh với VN-Index và trực quan hóa dữ liệu.</div>',
    unsafe_allow_html=True
)

# Tạo các tab ngang cho các trang với tên được tùy chỉnh
tab_names = [
    "Tải dữ liệu cổ phiếu",
    "Tối ưu danh mục (SLSQP)",
    "Tối ưu danh mục (SGD-Volatility)",
    "Tối ưu danh mục (SGD - Sharpe)",
    "Trực quan hóa dữ liệu",
    "Thông tin công ty",
    "Báo cáo tài chính",
    "Phân tích kỹ thuật",
    "Bảng giá giao dịch"
]
tabs = st.tabs(tab_names)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = tabs
###########################################
# Tab 1: Tải dữ liệu cổ phiếu
###########################################
with tab1:
    st.header("Nhập mã cổ phiếu và tải dữ liệu")
    st.write("Nhập các mã cổ phiếu (phân cách bởi dấu phẩy, ví dụ: ACB, VCB):")
    symbols_input = st.text_input("Mã cổ phiếu")
    
    # Thêm trường nhập số tiền đầu tư
    investment_amount = st.number_input("Nhập số tiền đầu tư (VND):", min_value=0.0, step=1000000.0, format="%.0f")
    
    if st.button("Tải dữ liệu"):
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        if not symbols:
            st.error("Danh sách mã cổ phiếu không được để trống!")
        else:
            # Lưu symbols và investment_amount vào session state
            st.session_state['symbols'] = symbols
            st.session_state['investment_amount'] = investment_amount
            all_data = []
            for symbol in symbols:
                try:
                    stock = Vnstock().stock(symbol=symbol, source='VCI')
                    historical_data = stock.quote.history(start='2020-01-01', end='2024-12-31')
                    if historical_data.empty:
                        st.warning(f"Không tìm thấy dữ liệu cho mã: {symbol}")
                        continue
                    historical_data['symbol'] = symbol
                    all_data.append(historical_data)
                    st.success(f"Đã tải dữ liệu cho: {symbol}")
                except Exception as e:
                    st.error(f"Lỗi khi tải dữ liệu cho {symbol}: {e}")
            if all_data:
                final_data = pd.concat(all_data, ignore_index=True)
                st.write("Đã kết hợp toàn bộ dữ liệu thành công!")
                
                def calculate_features(data):
                    data['daily_return'] = data['close'].pct_change()
                    data['volatility'] = data['daily_return'].rolling(window=30).std()
                    data.dropna(inplace=True)
                    return data
                
                processed_data = final_data.groupby('symbol').apply(calculate_features)
                processed_data = processed_data.reset_index(drop=True)
                processed_data.to_csv("processed_stock_data.csv", index=False)
                st.success("Dữ liệu xử lý đã được lưu vào file 'processed_stock_data.csv'.")
                st.dataframe(processed_data)
            else:
                st.error("Không có dữ liệu hợp lệ để xử lý!")

###########################################
# Tab 2: Tối ưu danh mục (SLSQP)
###########################################
with tab2:
    st.header("Tối ưu danh mục (SLSQP)")
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
        st.success("Đã tải dữ liệu xử lý thành công.")
    except FileNotFoundError:
        st.error("File 'processed_stock_data.csv' không tồn tại. Vui lòng tải dữ liệu ở tab 'Tải dữ liệu cổ phiếu' trước.")
        st.stop()
    
    # Tính toán kỳ vọng lợi nhuận và ma trận hiệp phương sai
    expected_returns = processed_data.groupby('symbol')['daily_return'].mean()
    pivot_returns = processed_data.pivot(index='time', columns='symbol', values='daily_return')
    cov_matrix = pivot_returns.cov()
    
    def objective(weights, expected_returns, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(len(expected_returns)))
    total_expected_return = expected_returns.sum()
    init_weights = expected_returns / total_expected_return
    result = minimize(objective, init_weights, args=(expected_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights_slsqp = result.x

    # Lấy số tiền đầu tư từ session state
    investment_amount = st.session_state.get('investment_amount', 0)
    
    st.subheader("Trọng số và số tiền đầu tư tối ưu (SLSQP)")
    # Tạo DataFrame chứa kết quả
    results_slsqp = pd.DataFrame({
        'Cổ phiếu': expected_returns.index,
        'Trọng số tối ưu': optimal_weights_slsqp,
        'Số tiền đầu tư (VND)': optimal_weights_slsqp * investment_amount
    })
    # Định dạng cột 'Số tiền đầu tư (VND)' với dấu phẩy và đơn vị VND
    results_slsqp['Số tiền đầu tư (VND)'] = results_slsqp['Số tiền đầu tư (VND)'].apply(lambda x: f"{x:,.0f} VND")
    # Hiển thị bảng
    st.dataframe(results_slsqp)
    
    # Biểu đồ trực quan: Pie & Bar
    portfolio_data_slsqp = pd.DataFrame({
        'Cổ phiếu': expected_returns.index,
        'Trọng số tối ưu': optimal_weights_slsqp,
        'Số tiền đầu tư': optimal_weights_slsqp * investment_amount
    })
    portfolio_data_filtered = portfolio_data_slsqp[portfolio_data_slsqp['Trọng số tối ưu'] > 0]

    fig_slsqp = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Trọng số tối ưu (Pie)', 'Số tiền đầu tư (Bar)'],
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )

    # Vẽ biểu đồ tròn với dữ liệu đã lọc
    fig_slsqp.add_trace(
        go.Pie(
            labels=portfolio_data_filtered['Cổ phiếu'],
            values=portfolio_data_filtered['Trọng số tối ưu'],
            hole=0.3,
            textinfo='percent+label',
            textfont_size=14,
            marker=dict(
                colors=[random_color() for _ in range(len(portfolio_data_filtered))],
                line=dict(color='#000000', width=2)
            ),
            hoverinfo='label+percent'
        ),
        row=1, col=1
    )

    # Vẽ biểu đồ cột với dữ liệu số tiền đầu tư
    fig_slsqp.add_trace(
        go.Bar(
            x=portfolio_data_filtered['Cổ phiếu'],
            y=portfolio_data_filtered['Số tiền đầu tư'],
            marker=dict(
                color=[random_color() for _ in range(len(portfolio_data_filtered))],
                line=dict(color='#000000', width=2)
            ),
            showlegend=False  # Ẩn legend của biểu đồ cột
        ),
        row=1, col=2
    )

    fig_slsqp.update_layout(
        title="So sánh trọng số và số tiền đầu tư tối ưu (SLSQP)",
        title_x=0.5,
        height=500,
        width=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )

    st.plotly_chart(fig_slsqp, use_container_width=True)
    
    # Tính lợi nhuận tích lũy của danh mục (SLSQP)
    processed_data['weighted_return_slsqp'] = processed_data['daily_return'] * processed_data['symbol'].map(
        dict(zip(expected_returns.index, optimal_weights_slsqp))
    )
    portfolio_daily_return_slsqp = processed_data.groupby('time')['weighted_return_slsqp'].sum().reset_index()
    portfolio_daily_return_slsqp.rename(columns={'weighted_return_slsqp': 'daily_return'}, inplace=True)
    # Tính lợi nhuận tích lũy và chuyển sang % (ví dụ: 125% thay vì 1.25)
    portfolio_daily_return_slsqp['cumulative_portfolio_return'] = (1 + portfolio_daily_return_slsqp['daily_return']).cumprod() * 100

    # Sử dụng Plotly để hiển thị biểu đồ với ký hiệu %
    fig_portfolio = go.Figure()
    fig_portfolio.add_trace(go.Scatter(
        x=portfolio_daily_return_slsqp['time'],
        y=portfolio_daily_return_slsqp['cumulative_portfolio_return'],
        mode='lines',
        name='Lợi nhuận tích lũy',
        hovertemplate='Ngày: %{x}<br>Lợi nhuận tích lũy: %{y:.2f}%<extra></extra>'
    ))
    fig_portfolio.update_layout(
        title="Lợi nhuận tích lũy của danh mục (SLSQP)",
        xaxis_title="Thời gian",
        yaxis_title="Lợi nhuận tích lũy (%)",
        template="plotly_white"
    )
    # Thêm ký hiệu % vào nhãn của trục Y
    fig_portfolio.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_portfolio, use_container_width=True)

    # So sánh với VN-Index
    with st.expander("So sánh với VN-Index"):
        try:
            vnindex_data = pd.read_csv("vnindex_data.csv")
            vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
            st.success("Đã tải dữ liệu VN-Index từ file 'vnindex_data.csv'.")
        except:
            st.warning("Không tìm thấy file 'vnindex_data.csv'. Đang tải dữ liệu VN-Index...")
            try:
                stock = Vnstock().stock(symbol='VNINDEX', source='VCI')
                vnindex_data = stock.quote.history(start='2020-01-01', end='2024-12-31')
                vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
                vnindex_data.to_csv("vnindex_data.csv", index=False)
                st.success("Đã lưu dữ liệu VN-Index vào file 'vnindex_data.csv'.")
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu VN-Index: {e}")
                st.stop()
        
        vnindex_data['market_return'] = vnindex_data['close'].pct_change()
        # Tính lợi nhuận tích lũy của VN-Index và chuyển sang %
        vnindex_data['cumulative_daily_return'] = (1 + vnindex_data['market_return']).cumprod() * 100
        
        comparison_slsqp = pd.merge(
            portfolio_daily_return_slsqp,
            vnindex_data[['time', 'cumulative_daily_return']],
            on='time',
            how='inner'
        )
        # Đổi tên cột để thể hiện đơn vị %
        comparison_slsqp.rename(columns={
            'cumulative_portfolio_return': 'Lợi nhuận danh mục (SLSQP) (%)',
            'cumulative_daily_return': 'Lợi nhuận VN-Index (%)'
        }, inplace=True)
        
        st.subheader("Bảng so sánh lợi nhuận (10 dòng cuối)")
        st.dataframe(comparison_slsqp[['time', 'Lợi nhuận danh mục (SLSQP) (%)', 'Lợi nhuận VN-Index (%)']].tail(10))
        
        fig_comp_slsqp = go.Figure()
        fig_comp_slsqp.add_trace(go.Scatter(
            x=comparison_slsqp['time'],
            y=comparison_slsqp['Lợi nhuận danh mục (SLSQP) (%)'],
            mode='lines',
            name='Lợi nhuận danh mục (SLSQP)',
            line=dict(color='blue', width=2),
            hovertemplate='Ngày: %{x}<br>Lợi nhuận danh mục (SLSQP): %{y:.2f}%<extra></extra>'
        ))
        fig_comp_slsqp.add_trace(go.Scatter(
            x=comparison_slsqp['time'],
            y=comparison_slsqp['Lợi nhuận VN-Index (%)'],
            mode='lines',
            name='Lợi nhuận VN-Index',
            line=dict(color='red', width=2),
            hovertemplate='Ngày: %{x}<br>Lợi nhuận VN-Index: %{y:.2f}%<extra></extra>'
        ))
        fig_comp_slsqp.update_layout(
            title="So sánh lợi nhuận danh mục (SLSQP) vs VN-Index",
            xaxis_title="Thời gian",
            yaxis_title="Lợi nhuận tích lũy (%)",
            template="plotly_white"
        )
        # Thêm ký hiệu % vào nhãn của trục Y
        fig_comp_slsqp.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_comp_slsqp, use_container_width=True)
        
        comparison_slsqp.to_csv("portfolio_vs_vnindex_comparison_slsqp.csv", index=False)
        st.write("Dữ liệu so sánh đã được lưu vào 'portfolio_vs_vnindex_comparison_slsqp.csv'.")

###########################################
# Tab 3: Tối ưu danh mục (SGD-Volatility)
###########################################
with tab3:
    st.header("Tối ưu danh mục (SGD-Volatility)")
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
        st.success("Đã tải dữ liệu xử lý từ file 'processed_stock_data.csv'.")
    except Exception:
        st.error("Không tìm thấy file 'processed_stock_data.csv'. Vui lòng tải dữ liệu ở tab 'Tải dữ liệu cổ phiếu'.")
        st.stop()
    
    expected_returns = processed_data.groupby('symbol')['daily_return'].mean()
    pivot_returns = processed_data.pivot(index='time', columns='symbol', values='daily_return')
    cov_matrix = pivot_returns.cov()
    
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def return_based_weights(expected_returns):
        return expected_returns / expected_returns.sum()
    
    # Thêm hàm chiếu trọng số lên simplex để đảm bảo tổng trọng số = 1 và các trọng số không âm
    def project_simplex(v, s=1):
        v = np.maximum(v, 0)
        total = np.sum(v)
        return v / total * s if total != 0 else v
    
    def sgd_optimization(expected_returns, cov_matrix, learning_rate=0.01, epochs=1000):
        weights = return_based_weights(expected_returns)
        for epoch in range(epochs):
            grad = np.dot(cov_matrix, weights) / portfolio_volatility(weights, cov_matrix)
            weights -= learning_rate * grad
            # Sử dụng project_simplex để đảm bảo ràng buộc
            weights = project_simplex(weights)
        return weights
    
    optimal_weights_sgd = sgd_optimization(expected_returns, cov_matrix, learning_rate=0.01, epochs=1000)
    
    # Lấy số tiền đầu tư từ session state
    investment_amount = st.session_state.get('investment_amount', 0)
    
    st.subheader("Trọng số và số tiền đầu tư tối ưu (SGD-Volatility)")
    # Tạo DataFrame chứa kết quả
    results_sgd = pd.DataFrame({
        'Cổ phiếu': expected_returns.index,
        'Trọng số tối ưu': optimal_weights_sgd,
        'Số tiền đầu tư (VND)': optimal_weights_sgd * investment_amount
    })
    # Định dạng cột 'Số tiền đầu tư (VND)' với dấu phẩy và đơn vị VND
    results_sgd['Số tiền đầu tư (VND)'] = results_sgd['Số tiền đầu tư (VND)'].apply(lambda x: f"{x:,.0f} VND")
    # Hiển thị bảng
    st.dataframe(results_sgd)
    
    # Biểu đồ trực quan: Pie & Bar
    portfolio_data_sgd = pd.DataFrame({
        'Cổ phiếu': expected_returns.index,
        'Trọng số tối ưu': optimal_weights_sgd,
        'Số tiền đầu tư': optimal_weights_sgd * investment_amount
    })
    portfolio_data_filtered = portfolio_data_sgd[portfolio_data_sgd['Trọng số tối ưu'] > 0]

    fig_sgd = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Trọng số tối ưu (Pie)', 'Số tiền đầu tư (Bar)'],
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )

    # Vẽ biểu đồ tròn với dữ liệu đã lọc
    fig_sgd.add_trace(
        go.Pie(
            labels=portfolio_data_filtered['Cổ phiếu'],
            values=portfolio_data_filtered['Trọng số tối ưu'],
            hole=0.3,
            textinfo='percent+label',
            textfont_size=14,
            marker=dict(
                colors=[random_color() for _ in range(len(portfolio_data_filtered))],
                line=dict(color='#000000', width=2)
            ),
            hoverinfo='label+percent'
        ),
        row=1, col=1
    )

    # Vẽ biểu đồ cột với dữ liệu số tiền đầu tư
    fig_sgd.add_trace(
        go.Bar(
            x=portfolio_data_filtered['Cổ phiếu'],
            y=portfolio_data_filtered['Số tiền đầu tư'],
            marker=dict(
                color=[random_color() for _ in range(len(portfolio_data_filtered))],
                line=dict(color='#000000', width=2)
            ),
            showlegend=False  # Ẩn legend của biểu đồ cột
        ),
        row=1, col=2
    )

    fig_sgd.update_layout(
        title="So sánh trọng số và số tiền đầu tư tối ưu (SGD-Volatility)",
        title_x=0.5,
        height=500,
        width=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    st.plotly_chart(fig_sgd, use_container_width=True)
    
    # Tính lợi nhuận tích lũy của danh mục (SGD-Volatility)
    processed_data['weighted_return_sgd'] = processed_data['daily_return'] * processed_data['symbol'].map(
        dict(zip(expected_returns.index, optimal_weights_sgd))
    )
    portfolio_daily_return_sgd = processed_data.groupby('time')['weighted_return_sgd'].sum().reset_index()
    portfolio_daily_return_sgd.rename(columns={'weighted_return_sgd': 'daily_return'}, inplace=True)
    # Tính lợi nhuận tích lũy và chuyển sang % (ví dụ: 125% thay vì 1.25)
    portfolio_daily_return_sgd['cumulative_portfolio_return'] = (1 + portfolio_daily_return_sgd['daily_return']).cumprod() * 100

    st.subheader("Lợi nhuận tích lũy của danh mục (SGD-Volatility)")
    # Sử dụng Plotly để hiển thị biểu đồ với ký hiệu %
    fig_portfolio_sgd = go.Figure()
    fig_portfolio_sgd.add_trace(go.Scatter(
        x=portfolio_daily_return_sgd['time'],
        y=portfolio_daily_return_sgd['cumulative_portfolio_return'],
        mode='lines',
        name='Lợi nhuận tích lũy',
        hovertemplate='Ngày: %{x}<br>Lợi nhuận tích lũy: %{y:.2f}%<extra></extra>'
    ))
    fig_portfolio_sgd.update_layout(
        title="Lợi nhuận tích lũy của danh mục (SGD-Volatility)",
        xaxis_title="Thời gian",
        yaxis_title="Lợi nhuận tích lũy (%)",
        template="plotly_white"
    )
    # Thêm ký hiệu % vào nhãn của trục Y
    fig_portfolio_sgd.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_portfolio_sgd, use_container_width=True)

    # So sánh với VN-Index
    with st.expander("So sánh với VN-Index"):
        try:
            vnindex_data = pd.read_csv("vnindex_data.csv")
            vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
            st.success("Đã tải dữ liệu VN-Index từ file 'vnindex_data.csv'.")
        except:
            st.warning("Không tìm thấy file 'vnindex_data.csv'. Đang tải dữ liệu VN-Index...")
            try:
                stock = Vnstock().stock(symbol='VNINDEX', source='VCI')
                vnindex_data = stock.quote.history(start='2020-01-01', end='2024-12-31')
                vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
                vnindex_data.to_csv("vnindex_data.csv", index=False)
                st.success("Đã lưu dữ liệu VN-Index vào file 'vnindex_data.csv'.")
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu VN-Index: {e}")
                st.stop()
        
        vnindex_data['market_return'] = vnindex_data['close'].pct_change()
        # Tính lợi nhuận tích lũy của VN-Index và chuyển sang %
        vnindex_data['cumulative_daily_return'] = (1 + vnindex_data['market_return']).cumprod() * 100
        
        comparison_sgd = pd.merge(
            portfolio_daily_return_sgd,
            vnindex_data[['time', 'cumulative_daily_return']],
            on='time',
            how='inner'
        )
        # Đổi tên cột để thể hiện đơn vị %
        comparison_sgd.rename(columns={
            'cumulative_portfolio_return': 'Lợi nhuận danh mục (SGD-Volatility) (%)',
            'cumulative_daily_return': 'Lợi nhuận VN-Index (%)'
        }, inplace=True)
        
        st.subheader("Bảng so sánh lợi nhuận (10 dòng cuối)")
        st.dataframe(comparison_sgd[['time', 'Lợi nhuận danh mục (SGD-Volatility) (%)', 'Lợi nhuận VN-Index (%)']].tail(10))
        
        fig_comp_sgd = go.Figure()
        fig_comp_sgd.add_trace(go.Scatter(
            x=comparison_sgd['time'],
            y=comparison_sgd['Lợi nhuận danh mục (SGD-Volatility) (%)'],
            mode='lines',
            name='Lợi nhuận danh mục (SGD)',
            line=dict(color='green', width=2),
            hovertemplate='Ngày: %{x}<br>Lợi nhuận danh mục (SGD-Volatility): %{y:.2f}%<extra></extra>'
        ))
        fig_comp_sgd.add_trace(go.Scatter(
            x=comparison_sgd['time'],
            y=comparison_sgd['Lợi nhuận VN-Index (%)'],
            mode='lines',
            name='Lợi nhuận VN-Index',
            line=dict(color='red', width=2),
            hovertemplate='Ngày: %{x}<br>Lợi nhuận VN-Index: %{y:.2f}%<extra></extra>'
        ))
        fig_comp_sgd.update_layout(
            title="So sánh lợi nhuận danh mục (SGD-Volatility) vs VN-Index",
            xaxis_title="Thời gian",
            yaxis_title="Lợi nhuận tích lũy (%)",
            template="plotly_white",
            hovermode="x unified"
        )
        # Thêm ký hiệu % vào nhãn của trục Y
        fig_comp_sgd.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_comp_sgd, use_container_width=True)
        comparison_sgd.to_csv("portfolio_vs_vnindex_comparison_sgd.csv", index=False)
        st.write("Dữ liệu so sánh đã được lưu vào 'portfolio_vs_vnindex_comparison_sgd.csv'.")

###########################################
# Tab 4: Tối ưu danh mục (SGD - Sharpe)
###########################################
with tab4:
    st.header("Tối ưu danh mục (SGD - Sharpe)")
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
        st.success("Đã tải dữ liệu xử lý từ file 'processed_stock_data.csv'.")
    except Exception:
        st.error("File 'processed_stock_data.csv' không tồn tại. Vui lòng tải dữ liệu ở tab 'Tải dữ liệu cổ phiếu'.")
        st.stop()

    # Tính lợi nhuận kỳ vọng và ma trận hiệp phương sai
    expected_returns = processed_data.groupby('symbol')['daily_return'].mean()
    cov_matrix = processed_data.pivot(index='time', columns='symbol', values='daily_return').cov()

    # Chuyển đổi thành mảng NumPy
    expected_returns_np = expected_returns.values
    cov_matrix_np = cov_matrix.values

    # Khởi tạo trọng số ban đầu
    weights = expected_returns_np / np.sum(expected_returns_np)

    # Tham số SGD
    learning_rate = 0.01
    epochs = 1000

    # Vòng lặp SGD để tối đa hóa Sharpe
    for epoch in range(epochs):
        # Tính lợi nhuận và độ biến động của danh mục
        portfolio_return = np.dot(weights, expected_returns_np)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_np, weights)))
        
        if portfolio_volatility > 0:
            # Tính gradient của tỷ số Sharpe
            numerator = expected_returns_np * portfolio_volatility**2 - portfolio_return * np.dot(cov_matrix_np, weights)
            grad = numerator / (portfolio_volatility**3)
            # Cập nhật trọng số theo hướng tối đa hóa Sharpe (gradient dương)
            weights += learning_rate * grad
        else:
            # Xử lý trường hợp độ biến động bằng 0
            weights += learning_rate * np.zeros_like(weights)
        
        # Chiếu trọng số lên simplex: đảm bảo tổng bằng 1 và không âm
        weights = np.maximum(weights, 0)
        total = np.sum(weights)
        weights = weights / total if total != 0 else weights

    # Chuyển trọng số tối ưu thành pandas Series để dễ thao tác
    optimal_weights_sgd_sharpe = pd.Series(weights, index=expected_returns.index)

    # Lấy số tiền đầu tư từ session state
    investment_amount = st.session_state.get('investment_amount', 0)
    
    # Hiển thị trọng số và số tiền đầu tư tối ưu
    st.subheader("Trọng số và số tiền đầu tư tối ưu (SGD - Sharpe)")
    # Tạo DataFrame chứa kết quả
    results_sharpe = pd.DataFrame({
        'Cổ phiếu': optimal_weights_sgd_sharpe.index,
        'Trọng số tối ưu': optimal_weights_sgd_sharpe.values,
        'Số tiền đầu tư (VND)': optimal_weights_sgd_sharpe * investment_amount
    })
    # Định dạng cột 'Số tiền đầu tư (VND)' với dấu phẩy và đơn vị VND
    results_sharpe['Số tiền đầu tư (VND)'] = results_sharpe['Số tiền đầu tư (VND)'].apply(lambda x: f"{x:,.0f} VND")
    # Hiển thị bảng
    st.dataframe(results_sharpe)

    # Tính và hiển thị tỷ số Sharpe
    portfolio_return = np.dot(optimal_weights_sgd_sharpe, expected_returns_np)
    portfolio_volatility = np.sqrt(np.dot(optimal_weights_sgd_sharpe.T, np.dot(cov_matrix_np, optimal_weights_sgd_sharpe)))
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    st.write(f"Tỷ lệ Sharpe tốt nhất: {sharpe_ratio:.4f}")

    # Biểu đồ trực quan: Pie & Bar
    portfolio_data_sharpe = pd.DataFrame({
        'Cổ phiếu': expected_returns.index,
        'Trọng số tối ưu': optimal_weights_sgd_sharpe,
        'Số tiền đầu tư': optimal_weights_sgd_sharpe * investment_amount
    })
    portfolio_data_filtered = portfolio_data_sharpe[portfolio_data_sharpe['Trọng số tối ưu'] > 0]

    fig_sharpe = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Trọng số tối ưu (Pie)', 'Số tiền đầu tư (Bar)'],
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )

    # Vẽ biểu đồ tròn với dữ liệu đã lọc
    fig_sharpe.add_trace(
        go.Pie(
            labels=portfolio_data_filtered['Cổ phiếu'],
            values=portfolio_data_filtered['Trọng số tối ưu'],
            hole=0.3,
            textinfo='percent+label',
            textfont_size=14,
            marker=dict(
                colors=[random_color() for _ in range(len(portfolio_data_filtered))],
                line=dict(color='#000000', width=2)
            ),
            hoverinfo='label+percent'
        ),
        row=1, col=1
    )

    # Vẽ biểu đồ cột với dữ liệu số tiền đầu tư
    fig_sharpe.add_trace(
        go.Bar(
            x=portfolio_data_filtered['Cổ phiếu'],
            y=portfolio_data_filtered['Số tiền đầu tư'],
            marker=dict(
                color=[random_color() for _ in range(len(portfolio_data_filtered))],
                line=dict(color='#000000', width=2)
            ),
            showlegend=False  # Ẩn legend của biểu đồ cột
        ),
        row=1, col=2
    )

    fig_sharpe.update_layout(
        title="So sánh trọng số và số tiền đầu tư tối ưu (SGD - Sharpe)",
        title_x=0.5,
        height=500,
        width=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )

    st.plotly_chart(fig_sharpe, use_container_width=True)

    # Tính lợi nhuận tích lũy của danh mục (SGD - Sharpe)
    processed_data['weighted_return_sharpe'] = processed_data['daily_return'] * processed_data['symbol'].map(optimal_weights_sgd_sharpe)
    portfolio_daily_return_sharpe = processed_data.groupby('time')['weighted_return_sharpe'].sum().reset_index()
    portfolio_daily_return_sharpe.rename(columns={'weighted_return_sharpe': 'daily_return'}, inplace=True)
    # Tính lợi nhuận tích lũy và chuyển sang % (ví dụ: 125% thay vì 1.25)
    portfolio_daily_return_sharpe['cumulative_portfolio_return'] = (1 + portfolio_daily_return_sharpe['daily_return']).cumprod() * 100

    st.subheader("Lợi nhuận tích lũy của danh mục (SGD - Sharpe)")
    # Sử dụng Plotly để hiển thị biểu đồ với ký hiệu %
    fig_portfolio_sharpe = go.Figure()
    fig_portfolio_sharpe.add_trace(go.Scatter(
        x=portfolio_daily_return_sharpe['time'],
        y=portfolio_daily_return_sharpe['cumulative_portfolio_return'],
        mode='lines',
        name='Lợi nhuận tích lũy',
        hovertemplate='Ngày: %{x}<br>Lợi nhuận tích lũy: %{y:.2f}%<extra></extra>'
    ))
    fig_portfolio_sharpe.update_layout(
        title="Lợi nhuận tích lũy của danh mục (SGD - Sharpe)",
        xaxis_title="Thời gian",
        yaxis_title="Lợi nhuận tích lũy (%)",
        template="plotly_white"
    )
    # Thêm ký hiệu % vào nhãn của trục Y
    fig_portfolio_sharpe.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_portfolio_sharpe, use_container_width=True)

    # So sánh với VN-Index
    with st.expander("So sánh với VN-Index"):
        try:
            vnindex_data = pd.read_csv("vnindex_data.csv")
            vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
            st.success("Đã tải dữ liệu VN-Index từ file 'vnindex_data.csv'.")
        except:
            st.warning("Không tìm thấy file 'vnindex_data.csv'. Đang tải dữ liệu VN-Index...")
            try:
                stock = Vnstock().stock(symbol='VNINDEX', source='VCI')
                vnindex_data = stock.quote.history(start='2020-01-01', end='2024-12-31')
                vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
                vnindex_data.to_csv("vnindex_data.csv", index=False)
                st.success("Đã lưu dữ liệu VN-Index vào file 'vnindex_data.csv'.")
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu VN-Index: {e}")
                st.stop()

        vnindex_data['market_return'] = vnindex_data['close'].pct_change()
        # Tính lợi nhuận tích lũy của VN-Index và chuyển sang %
        vnindex_data['cumulative_daily_return'] = (1 + vnindex_data['market_return']).cumprod() * 100

        comparison_sharpe = pd.merge(
            portfolio_daily_return_sharpe,
            vnindex_data[['time', 'cumulative_daily_return']],
            on='time',
            how='inner'
        )
        # Đổi tên cột để thể hiện đơn vị %
        comparison_sharpe.rename(columns={
            'cumulative_portfolio_return': 'Lợi nhuận danh mục (Sharpe) (%)',
            'cumulative_daily_return': 'Lợi nhuận VN-Index (%)'
        }, inplace=True)

        st.subheader("Bảng so sánh lợi nhuận (10 dòng cuối)")
        st.dataframe(comparison_sharpe[['time', 'Lợi nhuận danh mục (Sharpe) (%)', 'Lợi nhuận VN-Index (%)']].tail(10))

        fig_comp_sharpe = go.Figure()
        fig_comp_sharpe.add_trace(go.Scatter(
            x=comparison_sharpe['time'],
            y=comparison_sharpe['Lợi nhuận danh mục (Sharpe) (%)'],
            mode='lines',
            name='Lợi nhuận danh mục (Sharpe)',
            line=dict(color='orange', width=2),
            hovertemplate='Ngày: %{x}<br>Lợi nhuận danh mục (Sharpe): %{y:.2f}%<extra></extra>'
        ))
        fig_comp_sharpe.add_trace(go.Scatter(
            x=comparison_sharpe['time'],
            y=comparison_sharpe['Lợi nhuận VN-Index (%)'],
            mode='lines',
            name='Lợi nhuận VN-Index',
            line=dict(color='red', width=2),
            hovertemplate='Ngày: %{x}<br>Lợi nhuận VN-Index: %{y:.2f}%<extra></extra>'
        ))
        fig_comp_sharpe.update_layout(
            title="So sánh lợi nhuận danh mục (Sharpe) vs VN-Index",
            xaxis_title="Thời gian",
            yaxis_title="Lợi nhuận tích lũy (%)",
            template="plotly_white",
            hovermode="x unified"
        )
        # Thêm ký hiệu % vào nhãn của trục Y
        fig_comp_sharpe.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_comp_sharpe, use_container_width=True)

        comparison_sharpe.to_csv("portfolio_vs_vnindex_comparison_sharpe.csv", index=False)
        st.write("Dữ liệu so sánh đã được lưu vào 'portfolio_vs_vnindex_comparison_sharpe.csv'.")

###########################################
# Tab 5: Trực quan hóa dữ liệu
###########################################
with tab5:
    st.header("Trực quan hóa dữ liệu")
    try:
        processed_data = pd.read_csv("processed_stock_data.csv")
        processed_data['time'] = pd.to_datetime(processed_data['time'])
    except Exception as e:
        st.error("Không thể tải file 'processed_stock_data.csv'. Vui lòng tải dữ liệu ở tab 'Tải dữ liệu cổ phiếu'.")
        st.stop()
    
    st.subheader("Xu hướng giá đóng cửa cổ phiếu theo thời gian")
    fig1 = px.line(
        processed_data,
        x='time',
        y='close',
        color='symbol',
        title='Xu hướng giá đóng cửa cổ phiếu theo thời gian',
        labels={'time': 'Thời gian', 'close': 'Giá đóng cửa', 'symbol': 'Mã cổ phiếu'},
    )
    fig1.update_layout(
        xaxis_title='Thời gian',
        yaxis_title='Giá đóng cửa',
        legend_title='Mã cổ phiếu',
        template='plotly_white',
        hovermode='x unified',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Biểu đồ nhiệt tương quan giá đóng cửa")
    close_data = processed_data.pivot_table(values='close', index='time', columns='symbol')
    correlation_matrix = close_data.corr()
    rounded_correlation = correlation_matrix.round(2)
    fig2 = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        colorbar=dict(title='Hệ số tương quan'),
    ))
    for i in range(len(rounded_correlation)):
        for j in range(len(rounded_correlation.columns)):
            fig2.add_annotation(
                text=str(rounded_correlation.iloc[i, j]),
                x=rounded_correlation.columns[j],
                y=rounded_correlation.index[i],
                showarrow=False,
                font=dict(color='black' if rounded_correlation.iloc[i, j] < 0 else 'white')
            )
    fig2.update_traces(
        hovertemplate='<b>Mã cổ phiếu: %{x}</b><br>' +
                      '<b>Mã cổ phiếu: %{y}</b><br>' +
                      'Hệ số tương quan: %{z:.4f}<extra></extra>'
    )
    fig2.update_layout(
        title='Biểu đồ nhiệt tương quan giá đóng cửa',
        xaxis_title='Mã cổ phiếu',
        yaxis_title='Mã cổ phiếu'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Biểu đồ nhiệt tương quan lợi nhuận hàng ngày")
    returns_data = processed_data.pivot_table(index='time', columns='symbol', values='daily_return')
    correlation_matrix_returns = returns_data.corr()
    fig3 = ff.create_annotated_heatmap(
        z=correlation_matrix_returns.values,
        x=correlation_matrix_returns.columns.tolist(),
        y=correlation_matrix_returns.columns.tolist(),
        colorscale='RdBu',
        zmin=-1, zmax=1
    )
    fig3.update_layout(title="Ma trận tương quan giữa các cổ phiếu")
    st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("Biến động cổ phiếu theo thời gian")
    fig4 = px.line(processed_data, x='time', y='volatility', color='symbol', title="Biến động cổ phiếu theo thời gian")
    fig4.update_xaxes(title_text='Ngày')
    fig4.update_yaxes(title_text='Biến động')
    st.plotly_chart(fig4, use_container_width=True)

###########################################
# Tab 6: Thông tin công ty
###########################################
with tab6:
    st.header("Thông tin tổng hợp về các công ty")
    
    if 'symbols' not in st.session_state:
        st.error("Vui lòng nhập mã cổ phiếu ở tab 'Tải dữ liệu cổ phiếu' trước.")
    else:
        symbols = st.session_state['symbols']
        
        for symbol in symbols:
            st.subheader(f"Thông tin cho mã {symbol}")
            try:
                company = Vnstock().stock(symbol=symbol, source='TCBS').company
                
                with st.expander("**Hồ sơ công ty:**"):
                    profile = company.profile()
                    if isinstance(profile, pd.DataFrame):
                        st.dataframe(profile)
                    else:
                        st.write(profile)
                
                with st.expander("**Cổ đông:**"):
                    shareholders = company.shareholders()
                    if isinstance(shareholders, pd.DataFrame):
                        st.dataframe(shareholders)
                    else:
                        st.write(shareholders)
                
                with st.expander("**Giao dịch nội bộ:**"):
                    insider_deals = company.insider_deals()
                    if isinstance(insider_deals, pd.DataFrame):
                        st.dataframe(insider_deals)
                    else:
                        st.write(insider_deals)
                
                with st.expander("**Công ty con:**"):
                    subsidiaries = company.subsidiaries()
                    if isinstance(subsidiaries, pd.DataFrame):
                        st.dataframe(subsidiaries)
                    else:
                        st.write(subsidiaries)
                
                with st.expander("**Ban điều hành:**"):
                    officers = company.officers()
                    if isinstance(officers, pd.DataFrame):
                        st.dataframe(officers)
                    else:
                        st.write(officers)
                
                with st.expander("**Sự kiện:**"):
                    events = company.events()
                    if isinstance(events, pd.DataFrame):
                        st.dataframe(events)
                    else:
                        st.write(events)
                
                with st.expander("**Tin tức:**"):
                    news = company.news()
                    if isinstance(news, list) and all(isinstance(item, dict) for item in news):
                        for item in news:
                            st.write(f"- {item.get('title', 'N/A')} ({item.get('date', 'N/A')})")
                            st.write(item.get('summary', 'Không có tóm tắt'))
                            url = item.get('url', None)
                            if url:
                                st.write(f"[Đọc thêm]({url})")
                            else:
                                st.write("Không có URL")
                    else:
                        st.write("Tin tức không khả dụng hoặc định dạng không đúng:")
                        st.write(news)
                
                with st.expander("**Cổ tức:**"):
                    dividends = company.dividends()
                    if isinstance(dividends, pd.DataFrame):
                        st.dataframe(dividends)
                    else:
                        st.write(dividends)
                
                # Mục "Tình hình tài chính" với xử lý thay thế nếu thiếu thuộc tính ratio_summary
                with st.expander("**Tình hình tài chính:**"):
                    try:
                        if hasattr(company, 'ratio_summary'):
                            ratio_summary = company.ratio_summary()
                        else:
                            # Nếu không có thuộc tính, khởi tạo lại đối tượng từ module vnstock.explorer.vci
                            from vnstock.explorer.vci import Company as VCICompany
                            company_alt = VCICompany(symbol)
                            ratio_summary = company_alt.ratio_summary()
                            
                        if isinstance(ratio_summary, pd.DataFrame):
                            st.dataframe(ratio_summary)
                        else:
                            st.write(ratio_summary)
                    except Exception as e:
                        st.write(f"Không có dữ liệu tình hình tài chính cho mã {symbol}: {e}")
                
                # Mục "Thống kê giao dịch" với xử lý tương tự nếu cần
                with st.expander("**Thống kê giao dịch:**"):
                    try:
                        if hasattr(company, 'trading_stats'):
                            trading_stats = company.trading_stats()
                        else:
                            from vnstock.explorer.vci import Company as VCICompany
                            company_alt = VCICompany(symbol)
                            trading_stats = company_alt.trading_stats()
                            
                        if isinstance(trading_stats, pd.DataFrame):
                            st.dataframe(trading_stats)
                        else:
                            st.write(trading_stats)
                    except Exception as e:
                        st.write(f"Không có dữ liệu thống kê giao dịch cho mã {symbol}: {e}")
            
            except Exception as e:
                st.error(f"Lỗi khi tải thông tin cho mã {symbol}: {e}")


with tab7:
    st.header("Tổng hợp báo cáo tài chính")
    
    # Cấu hình Plotly: modebar luôn hiển thị
    config = {
        "displayModeBar": True,
        "displaylogo": False
    }
    
    if 'symbols' not in st.session_state:
        st.error("Vui lòng nhập mã cổ phiếu ở trang 'Fetch Stock Data' trước.")
    else:
        symbols = st.session_state['symbols']

        def rename_duplicate_columns(df):
            if df.empty:
                return df
            if isinstance(df.columns, pd.MultiIndex):
                flat_columns = [
                    '_'.join(str(col).strip() for col in multi_col if str(col).strip())
                    for multi_col in df.columns
                ]
            else:
                flat_columns = df.columns.tolist()
            seen = {}
            final_columns = []
            for col in flat_columns:
                if col in seen:
                    seen[col] += 1
                    final_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    final_columns.append(col)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = pd.Index(final_columns)
            else:
                df.columns = final_columns
            return df

        # CSS cho nội dung của expander (background trắng, đổ bóng,...)
        st.markdown(
            """
            <style>
            .streamlit-expanderContent {
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Hàm random_color (nếu chưa có)
        import random
        def random_color():
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
            return random.choice(colors)

        # --- Sử dụng caching để tải dữ liệu tài chính ---
        @st.cache_data(show_spinner=False)
        def get_financial_data(symbol, report_type, period='year'):
            try:
                stock = Vnstock().stock(symbol=symbol, source='VCI')
                if report_type == "balance":
                    data = stock.finance.balance_sheet(period=period, lang='vi', dropna=True)
                elif report_type == "income":
                    data = stock.finance.income_statement(period=period, lang='vi', dropna=True)
                elif report_type == "cashflow":
                    data = stock.finance.cash_flow(period=period, lang="vi", dropna=True)
                elif report_type == "ratios":
                    data = stock.finance.ratio(period=period, lang='vi', dropna=True)
                else:
                    data = pd.DataFrame()
                data = rename_duplicate_columns(data)
                return data
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu cho {symbol} - {report_type}: {e}")
                return pd.DataFrame()
        
        # ----------------------- Lặp qua từng mã cổ phiếu -----------------------
        for symbol in symbols:
            st.header(f"Báo cáo tài chính cho mã {symbol}")

            # ----------------------- 1) BẢNG CÂN ĐỐI KẾ TOÁN (Hàng năm) -----------------------
            with st.expander("Bảng cân đối kế toán (Hàng năm)"):
                balance_data = get_financial_data(symbol, "balance", period="year")
                if not balance_data.empty and 'Năm' in balance_data.columns:
                    # Loại bỏ dấu phẩy trong Năm, chuyển về int, sắp xếp tăng dần
                    balance_data['Năm'] = (
                        balance_data['Năm']
                        .astype(str)
                        .str.replace(',', '', regex=False)
                        .astype(int)
                    )
                    balance_data = balance_data.sort_values('Năm')

                    # Chuyển vị DataFrame
                    df_balance_transposed = balance_data.set_index('Năm').T
                    st.write("**Bảng cân đối kế toán (Hàng năm):**")
                    st.dataframe(df_balance_transposed)

                    # ---------- Phần biểu đồ (tham chiếu dữ liệu gốc) ----------
                    numeric_cols = [
                        col for col in balance_data.select_dtypes(include=['float64', 'int64']).columns
                        if col != 'Năm'
                    ]
                    if numeric_cols:
                        selected_cols = st.multiselect(
                            f"Chọn các chỉ số để hiển thị biểu đồ (Bảng cân đối {symbol}):",
                            options=numeric_cols,
                            default=[]
                        )
                        available_years = sorted(balance_data['Năm'].unique())
                        selected_years = st.multiselect(
                            f"Chọn năm hiển thị cho biểu đồ (Bảng cân đối {symbol}):",
                            options=available_years,
                            default=[]
                        )
                        df_filtered = (
                            balance_data[balance_data['Năm'].isin(selected_years)]
                            if selected_years else balance_data
                        )

                        if selected_cols:
                            for i in range(0, len(selected_cols), 5):
                                cols = st.columns(5)
                                for j, col in enumerate(selected_cols[i:i+5]):
                                    with cols[j]:
                                        st.markdown(f"**{col}**")
                                        tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ CAGR"])
                                        
                                        # Biểu đồ cột
                                        with tab1:
                                            fig_bar = go.Figure()
                                            fig_bar.add_trace(go.Bar(
                                                x=df_filtered['Năm'],
                                                y=df_filtered[col],
                                                name=col,
                                                marker_color=random_color(),
                                                hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                            ))
                                            fig_bar.update_layout(
                                                title=f"{col} - {symbol}",
                                                xaxis_title="Năm",
                                                yaxis_title="Giá trị (Tỷ đồng)",
                                                template="plotly_white",
                                                height=300,
                                                margin=dict(l=20, r=20, t=150, b=20)
                                            )
                                            st.plotly_chart(fig_bar, use_container_width=True, config=config, key=f"balance_{symbol}_{col}_bar")
                                        
                                        # Biểu đồ CAGR
                                        with tab2:
                                            if df_filtered.shape[0] >= 2:
                                                df_sorted = df_filtered.sort_values('Năm')
                                                start_year = df_sorted['Năm'].iloc[0]
                                                start_val = df_sorted[col].iloc[0]
                                                if start_val != 0:
                                                    years = df_sorted['Năm']
                                                    cagr_values = []
                                                    for y, val in zip(years, df_sorted[col]):
                                                        period = y - start_year
                                                        if period == 0:
                                                            cagr_values.append(None)
                                                        else:
                                                            cagr_val = (val / start_val)**(1/period) - 1
                                                            cagr_values.append(cagr_val * 100)
                                                    fig_cagr = go.Figure()
                                                    fig_cagr.add_trace(go.Scatter(
                                                        x=years,
                                                        y=cagr_values,
                                                        mode='lines+markers',
                                                        name='CAGR',
                                                        marker_color='red',
                                                        hovertemplate="CAGR: %{y:.2f}%<br>Năm: %{x}"
                                                    ))
                                                    fig_cagr.update_layout(
                                                        title=f"CAGR của {col} - {symbol}",
                                                        xaxis_title="Năm",
                                                        yaxis_title="CAGR (%)",
                                                        template="plotly_white",
                                                        height=300,
                                                        margin=dict(l=20, r=20, t=150, b=20)
                                                    )
                                                    st.plotly_chart(fig_cagr, use_container_width=True, config=config, key=f"balance_{symbol}_{col}_cagr")
                                                else:
                                                    st.info("Giá trị ban đầu bằng 0, không thể tính CAGR.")
                                            else:
                                                st.info("Không đủ dữ liệu để tính CAGR.")
                else:
                    st.warning(f"Không có dữ liệu hoặc cột 'Năm' cho bảng cân đối kế toán của {symbol}")

            # ----------------------- 2) BÁO CÁO LÃI LỖ (Hàng năm) -----------------------
            with st.expander("Báo cáo lãi lỗ (Hàng năm)"):
                income_data = get_financial_data(symbol, "income", period="year")
                if not income_data.empty and 'Năm' in income_data.columns:
                    # Loại bỏ dấu phẩy trong Năm, chuyển về int, sắp xếp tăng dần
                    income_data['Năm'] = (
                        income_data['Năm']
                        .astype(str)
                        .str.replace(',', '', regex=False)
                        .astype(int)
                    )
                    income_data = income_data.sort_values('Năm')

                    # Chuyển vị DataFrame
                    df_income_transposed = income_data.set_index('Năm').T
                    st.write("**Báo cáo lãi lỗ (Hàng năm):**")
                    st.dataframe(df_income_transposed)

                    # ---------- Phần biểu đồ (tham chiếu dữ liệu gốc) ----------
                    numeric_cols = [
                        col for col in income_data.select_dtypes(include=['float64', 'int64']).columns
                        if col != 'Năm'
                    ]
                    if numeric_cols:
                        selected_cols = st.multiselect(
                            f"Chọn các chỉ số để hiển thị biểu đồ (Báo cáo lãi lỗ {symbol}):",
                            options=numeric_cols,
                            default=[]
                        )
                        available_years = sorted(income_data['Năm'].unique())
                        selected_years = st.multiselect(
                            f"Chọn năm hiển thị cho biểu đồ (Báo cáo lãi lỗ {symbol}):",
                            options=available_years,
                            default=[]
                        )
                        df_filtered = (
                            income_data[income_data['Năm'].isin(selected_years)]
                            if selected_years else income_data
                        )

                        if selected_cols:
                            for i in range(0, len(selected_cols), 5):
                                cols = st.columns(5)
                                for j, col in enumerate(selected_cols[i:i+5]):
                                    with cols[j]:
                                        st.markdown(f"**{col}**")
                                        tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ CAGR"])
                                        
                                        with tab1:
                                            fig_bar = go.Figure()
                                            fig_bar.add_trace(go.Bar(
                                                x=df_filtered['Năm'],
                                                y=df_filtered[col],
                                                name=col,
                                                marker_color=random_color(),
                                                hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                            ))
                                            fig_bar.update_layout(
                                                title=f"{col} - {symbol}",
                                                xaxis_title="Năm",
                                                yaxis_title="Giá trị (Tỷ đồng)",
                                                template="plotly_white",
                                                height=300,
                                                margin=dict(l=20, r=20, t=150, b=20)
                                            )
                                            st.plotly_chart(fig_bar, use_container_width=True, config=config, key=f"income_{symbol}_{col}_bar")
                                        
                                        with tab2:
                                            if df_filtered.shape[0] >= 2:
                                                df_sorted = df_filtered.sort_values('Năm')
                                                start_year = df_sorted['Năm'].iloc[0]
                                                start_val = df_sorted[col].iloc[0]
                                                if start_val != 0:
                                                    years = df_sorted['Năm']
                                                    cagr_values = []
                                                    for y, val in zip(years, df_sorted[col]):
                                                        period = y - start_year
                                                        if period == 0:
                                                            cagr_values.append(None)
                                                        else:
                                                            cagr_val = (val / start_val)**(1/period) - 1
                                                            cagr_values.append(cagr_val * 100)
                                                    fig_cagr = go.Figure()
                                                    fig_cagr.add_trace(go.Scatter(
                                                        x=years,
                                                        y=cagr_values,
                                                        mode='lines+markers',
                                                        name='CAGR',
                                                        marker_color='red',
                                                        hovertemplate="CAGR: %{y:.2f}%<br>Năm: %{x}"
                                                    ))
                                                    fig_cagr.update_layout(
                                                        title=f"CAGR của {col} - {symbol}",
                                                        xaxis_title="Năm",
                                                        yaxis_title="CAGR (%)",
                                                        template="plotly_white",
                                                        height=300,
                                                        margin=dict(l=20, r=20, t=150, b=20)
                                                    )
                                                    st.plotly_chart(fig_cagr, use_container_width=True, config=config, key=f"income_{symbol}_{col}_cagr")
                                                else:
                                                    st.info("Giá trị ban đầu bằng 0, không thể tính CAGR.")
                                            else:
                                                st.info("Không đủ dữ liệu để tính CAGR.")
                else:
                    st.warning(f"Không có dữ liệu hoặc cột 'Năm' cho báo cáo lãi lỗ của {symbol}")

            # ----------------------- 3) BÁO CÁO LƯU CHUYỂN TIỀN TỆ (Hàng năm) -----------------------
            with st.expander("Báo cáo lưu chuyển tiền tệ (Hàng năm)"):
                cash_flow_data = get_financial_data(symbol, "cashflow", period="year")
                if not cash_flow_data.empty and 'Năm' in cash_flow_data.columns:
                    # Loại bỏ dấu phẩy trong Năm, chuyển về int, sắp xếp tăng dần
                    cash_flow_data['Năm'] = (
                        cash_flow_data['Năm']
                        .astype(str)
                        .str.replace(',', '', regex=False)
                        .astype(int)
                    )
                    cash_flow_data = cash_flow_data.sort_values('Năm')

                    # Chuyển vị DataFrame
                    df_cashflow_transposed = cash_flow_data.set_index('Năm').T
                    st.write("**Báo cáo lưu chuyển tiền tệ (Hàng năm):**")
                    st.dataframe(df_cashflow_transposed)

                    # ---------- Phần biểu đồ (tham chiếu dữ liệu gốc) ----------
                    numeric_cols = [
                        col for col in cash_flow_data.select_dtypes(include=['float64', 'int64']).columns
                        if col != 'Năm'
                    ]
                    if numeric_cols:
                        selected_cols = st.multiselect(
                            f"Chọn các chỉ số để hiển thị biểu đồ (Báo cáo lưu chuyển {symbol}):",
                            options=numeric_cols,
                            default=[]
                        )
                        available_years = sorted(cash_flow_data['Năm'].unique())
                        selected_years = st.multiselect(
                            f"Chọn năm hiển thị cho biểu đồ (Báo cáo lưu chuyển {symbol}):",
                            options=available_years,
                            default=[]
                        )
                        df_filtered = (
                            cash_flow_data[cash_flow_data['Năm'].isin(selected_years)]
                            if selected_years else cash_flow_data
                        )

                        if selected_cols:
                            for i in range(0, len(selected_cols), 5):
                                cols = st.columns(5)
                                for j, col in enumerate(selected_cols[i:i+5]):
                                    with cols[j]:
                                        st.markdown(f"**{col}**")
                                        tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ CAGR"])
                                        
                                        with tab1:
                                            fig_bar = go.Figure()
                                            fig_bar.add_trace(go.Bar(
                                                x=df_filtered['Năm'],
                                                y=df_filtered[col],
                                                name=col,
                                                marker_color=random_color(),
                                                hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                            ))
                                            fig_bar.update_layout(
                                                title=f"{col} - {symbol}",
                                                xaxis_title="Năm",
                                                yaxis_title="Giá trị (Tỷ đồng)",
                                                template="plotly_white",
                                                height=300,
                                                margin=dict(l=20, r=20, t=150, b=20)
                                            )
                                            st.plotly_chart(fig_bar, use_container_width=True, config=config, key=f"cashflow_{symbol}_{col}_bar")
                                        
                                        with tab2:
                                            if df_filtered.shape[0] >= 2:
                                                df_sorted = df_filtered.sort_values('Năm')
                                                start_year = df_sorted['Năm'].iloc[0]
                                                start_val = df_sorted[col].iloc[0]
                                                if start_val != 0:
                                                    years = df_sorted['Năm']
                                                    cagr_values = []
                                                    for y, val in zip(years, df_sorted[col]):
                                                        period = y - start_year
                                                        if period == 0:
                                                            cagr_values.append(None)
                                                        else:
                                                            cagr_val = (val / start_val)**(1/period) - 1
                                                            cagr_values.append(cagr_val * 100)
                                                    fig_cagr = go.Figure()
                                                    fig_cagr.add_trace(go.Scatter(
                                                        x=years,
                                                        y=cagr_values,
                                                        mode='lines+markers',
                                                        name='CAGR',
                                                        marker_color='red',
                                                        hovertemplate="CAGR: %{y:.2f}%<br>Năm: %{x}"
                                                    ))
                                                    fig_cagr.update_layout(
                                                        title=f"CAGR của {col} - {symbol}",
                                                        xaxis_title="Năm",
                                                        yaxis_title="CAGR (%)",
                                                        template="plotly_white",
                                                        height=300,
                                                        margin=dict(l=20, r=20, t=150, b=20)
                                                    )
                                                    st.plotly_chart(fig_cagr, use_container_width=True, config=config, key=f"cashflow_{symbol}_{col}_cagr")
                                                else:
                                                    st.info("Giá trị ban đầu bằng 0, không thể tính CAGR.")
                                            else:
                                                st.info("Không đủ dữ liệu để tính CAGR.")
                else:
                    st.warning(f"Không có dữ liệu hoặc cột 'Năm' cho báo cáo lưu chuyển tiền tệ của {symbol}")

            # ----------------------- 4) CHỈ SỐ TÀI CHÍNH (Hàng năm) -----------------------
            with st.expander("Chỉ số tài chính (Hàng năm)"):
                ratios_data = get_financial_data(symbol, "ratios", period="year")
                # Ở phần này cột năm là 'Meta_Năm'
                if not ratios_data.empty and 'Meta_Năm' in ratios_data.columns:
                    # Loại bỏ dấu phẩy trong Meta_Năm, chuyển về int, sắp xếp tăng dần
                    ratios_data['Meta_Năm'] = (
                        ratios_data['Meta_Năm']
                        .astype(str)
                        .str.replace(',', '', regex=False)
                        .astype(int)
                    )
                    ratios_data = ratios_data.sort_values('Meta_Năm')

                    # Chuyển vị DataFrame
                    df_ratios_transposed = ratios_data.set_index('Meta_Năm').T
                    st.write("**Chỉ số tài chính (Hàng năm):**")
                    st.dataframe(df_ratios_transposed)

                    # ---------- Phần biểu đồ (tham chiếu dữ liệu gốc) ----------
                    numeric_cols = [
                        col for col in ratios_data.select_dtypes(include=['float64', 'int64']).columns
                        if col != 'Meta_Năm'
                    ]
                    if numeric_cols:
                        selected_cols = st.multiselect(
                            f"Chọn các chỉ số để hiển thị biểu đồ (Chỉ số tài chính {symbol}):",
                            options=numeric_cols,
                            default=[]
                        )
                        available_years = sorted(ratios_data['Meta_Năm'].unique())
                        selected_years = st.multiselect(
                            f"Chọn năm hiển thị cho biểu đồ (Chỉ số tài chính {symbol}):",
                            options=available_years,
                            default=[]
                        )
                        df_filtered = (
                            ratios_data[ratios_data['Meta_Năm'].isin(selected_years)]
                            if selected_years else ratios_data
                        )

                        if selected_cols:
                            for i in range(0, len(selected_cols), 5):
                                cols = st.columns(5)
                                for j, col in enumerate(selected_cols[i:i+5]):
                                    with cols[j]:
                                        st.markdown(f"**{col}**")
                                        tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ CAGR"])
                                        
                                        with tab1:
                                            fig_bar = go.Figure()
                                            fig_bar.add_trace(go.Bar(
                                                x=df_filtered['Meta_Năm'],
                                                y=df_filtered[col],
                                                name=col,
                                                marker_color=random_color(),
                                                hovertemplate=f"{col}: %{{y:.2f}}<br>Năm: %{{x}}"
                                            ))
                                            fig_bar.update_layout(
                                                title=f"{col} - {symbol}",
                                                xaxis_title="Năm",
                                                yaxis_title="Giá trị",
                                                template="plotly_white",
                                                height=300,
                                                margin=dict(l=20, r=20, t=150, b=20)
                                            )
                                            st.plotly_chart(fig_bar, use_container_width=True, config=config, key=f"ratios_{symbol}_{col}_bar")
                                        
                                        with tab2:
                                            if df_filtered.shape[0] >= 2:
                                                df_sorted = df_filtered.sort_values('Meta_Năm')
                                                start_year = df_sorted['Meta_Năm'].iloc[0]
                                                start_val = df_sorted[col].iloc[0]
                                                if start_val != 0:
                                                    years = df_sorted['Meta_Năm']
                                                    cagr_values = []
                                                    for y, val in zip(years, df_sorted[col]):
                                                        period = y - start_year
                                                        if period == 0:
                                                            cagr_values.append(None)
                                                        else:
                                                            cagr_val = (val / start_val)**(1/period) - 1
                                                            cagr_values.append(cagr_val * 100)
                                                    fig_cagr = go.Figure()
                                                    fig_cagr.add_trace(go.Scatter(
                                                        x=years,
                                                        y=cagr_values,
                                                        mode='lines+markers',
                                                        name='CAGR',
                                                        marker_color='red',
                                                        hovertemplate="CAGR: %{y:.2f}%<br>Năm: %{x}"
                                                    ))
                                                    fig_cagr.update_layout(
                                                        title=f"CAGR của {col} - {symbol}",
                                                        xaxis_title="Năm",
                                                        yaxis_title="CAGR (%)",
                                                        template="plotly_white",
                                                        height=300,
                                                        margin=dict(l=20, r=20, t=150, b=20)
                                                    )
                                                    st.plotly_chart(fig_cagr, use_container_width=True, config=config, key=f"ratios_{symbol}_{col}_cagr")
                                                else:
                                                    st.info("Giá trị ban đầu bằng 0, không thể tính CAGR.")
                                            else:
                                                st.info("Không đủ dữ liệu để tính CAGR.")
                else:
                    st.warning(f"Không có dữ liệu hoặc cột 'Meta_Năm' cho chỉ số tài chính của {symbol}")

            # ----------------------- 5) BẢNG CÂN ĐỐI KẾ TOÁN (Hàng quý) -----------------------
            with st.expander("Bảng cân đối kế toán (Hàng quý)"):
                balance_data_quarter = get_financial_data(symbol, "balance", period="quarter")
                if not balance_data_quarter.empty and 'Năm' in balance_data_quarter.columns and 'Kỳ' in balance_data_quarter.columns:
                    # Tạo cột mới kết hợp Năm và Kỳ
                    balance_data_quarter['Năm_Kỳ'] = balance_data_quarter['Năm'].astype(str) + " Q" + balance_data_quarter['Kỳ'].astype(str)
                    balance_data_quarter = balance_data_quarter.sort_values(['Năm', 'Kỳ'])

                    # Chuyển vị DataFrame
                    df_balance_transposed_quarter = balance_data_quarter.set_index('Năm_Kỳ').drop(columns=['Năm', 'Kỳ']).T
                    st.write("**Bảng cân đối kế toán (Hàng quý):**")
                    st.dataframe(df_balance_transposed_quarter)

                    # ---------- Phần biểu đồ (tham chiếu dữ liệu gốc) ----------
                    numeric_cols = [
                        col for col in balance_data_quarter.select_dtypes(include=['float64', 'int64']).columns
                        if col not in ['Năm', 'Kỳ', 'Năm_Kỳ']
                    ]
                    if numeric_cols:
                        selected_cols = st.multiselect(
                            f"Chọn các chỉ số để hiển thị biểu đồ (Bảng cân đối hàng quý {symbol}):",
                            options=numeric_cols,
                            default=[]
                        )
                        available_quarters = sorted(balance_data_quarter['Năm_Kỳ'].unique())
                        selected_quarters = st.multiselect(
                            f"Chọn quý hiển thị cho biểu đồ (Bảng cân đối hàng quý {symbol}):",
                            options=available_quarters,
                            default=[]
                        )
                        df_filtered = (
                            balance_data_quarter[balance_data_quarter['Năm_Kỳ'].isin(selected_quarters)]
                            if selected_quarters else balance_data_quarter
                        )

                        if selected_cols:
                            for i in range(0, len(selected_cols), 5):
                                cols = st.columns(5)
                                for j, col in enumerate(selected_cols[i:i+5]):
                                    with cols[j]:
                                        st.markdown(f"**{col}**")
                                        tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ CAGR"])
                                        
                                        # Biểu đồ cột
                                        with tab1:
                                            fig_bar = go.Figure()
                                            fig_bar.add_trace(go.Bar(
                                                x=df_filtered['Năm_Kỳ'],
                                                y=df_filtered[col],
                                                name=col,
                                                marker_color=random_color(),
                                                hovertemplate=f"{col}: %{{y:.2f}}<br>Quý: %{{x}}"
                                            ))
                                            fig_bar.update_layout(
                                                title=f"{col} - {symbol}",
                                                xaxis_title="Quý",
                                                yaxis_title="Giá trị (Tỷ đồng)",
                                                template="plotly_white",
                                                height=300,
                                                margin=dict(l=20, r=20, t=150, b=20)
                                            )
                                            st.plotly_chart(fig_bar, use_container_width=True, config=config, key=f"balance_quarter_{symbol}_{col}_bar")
                                        
                                        # Biểu đồ CAGR
                                        with tab2:
                                            if df_filtered.shape[0] >= 2:
                                                df_sorted = df_filtered.sort_values(['Năm', 'Kỳ'])
                                                start_quarter = df_sorted['Năm_Kỳ'].iloc[0]
                                                start_val = df_sorted[col].iloc[0]
                                                if start_val != 0:
                                                    quarters = df_sorted['Năm_Kỳ']
                                                    cagr_values = []
                                                    for idx, (q, val) in enumerate(zip(quarters, df_sorted[col])):
                                                        period = idx / 4  # Mỗi quý là 1/4 năm
                                                        if period == 0:
                                                            cagr_values.append(None)
                                                        else:
                                                            cagr_val = (val / start_val)**(1/period) - 1
                                                            cagr_values.append(cagr_val * 100)
                                                    fig_cagr = go.Figure()
                                                    fig_cagr.add_trace(go.Scatter(
                                                        x=quarters,
                                                        y=cagr_values,
                                                        mode='lines+markers',
                                                        name='CAGR',
                                                        marker_color='red',
                                                        hovertemplate="CAGR: %{y:.2f}%<br>Quý: %{x}"
                                                    ))
                                                    fig_cagr.update_layout(
                                                        title=f"CAGR của {col} - {symbol}",
                                                        xaxis_title="Quý",
                                                        yaxis_title="CAGR (%)",
                                                        template="plotly_white",
                                                        height=300,
                                                        margin=dict(l=20, r=20, t=150, b=20)
                                                    )
                                                    st.plotly_chart(fig_cagr, use_container_width=True, config=config, key=f"balance_quarter_{symbol}_{col}_cagr")
                                                else:
                                                    st.info("Giá trị ban đầu bằng 0, không thể tính CAGR.")
                                            else:
                                                st.info("Không đủ dữ liệu để tính CAGR.")
                else:
                    st.warning(f"Không có dữ liệu hoặc cột 'Năm'/'Kỳ' cho bảng cân đối kế toán hàng quý của {symbol}")

            # ----------------------- 6) BÁO CÁO LÃI LỖ (Hàng quý) -----------------------
            with st.expander("Báo cáo lãi lỗ (Hàng quý)"):
                income_data_quarter = get_financial_data(symbol, "income", period="quarter")
                if not income_data_quarter.empty and 'Năm' in income_data_quarter.columns and 'Kỳ' in income_data_quarter.columns:
                    # Tạo cột mới kết hợp Năm và Kỳ
                    income_data_quarter['Năm_Kỳ'] = income_data_quarter['Năm'].astype(str) + " Q" + income_data_quarter['Kỳ'].astype(str)
                    income_data_quarter = income_data_quarter.sort_values(['Năm', 'Kỳ'])

                    # Chuyển vị DataFrame
                    df_income_transposed_quarter = income_data_quarter.set_index('Năm_Kỳ').drop(columns=['Năm', 'Kỳ']).T
                    st.write("**Báo cáo lãi lỗ (Hàng quý):**")
                    st.dataframe(df_income_transposed_quarter)

                    # ---------- Phần biểu đồ (tham chiếu dữ liệu gốc) ----------
                    numeric_cols = [
                        col for col in income_data_quarter.select_dtypes(include=['float64', 'int64']).columns
                        if col not in ['Năm', 'Kỳ', 'Năm_Kỳ']
                    ]
                    if numeric_cols:
                        selected_cols = st.multiselect(
                            f"Chọn các chỉ số để hiển thị biểu đồ (Báo cáo lãi lỗ hàng quý {symbol}):",
                            options=numeric_cols,
                            default=[]
                        )
                        available_quarters = sorted(income_data_quarter['Năm_Kỳ'].unique())
                        selected_quarters = st.multiselect(
                            f"Chọn quý hiển thị cho biểu đồ (Báo cáo lãi lỗ hàng quý {symbol}):",
                            options=available_quarters,
                            default=[]
                        )
                        df_filtered = (
                            income_data_quarter[income_data_quarter['Năm_Kỳ'].isin(selected_quarters)]
                            if selected_quarters else income_data_quarter
                        )

                        if selected_cols:
                            for i in range(0, len(selected_cols), 5):
                                cols = st.columns(5)
                                for j, col in enumerate(selected_cols[i:i+5]):
                                    with cols[j]:
                                        st.markdown(f"**{col}**")
                                        tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ CAGR"])
                                        
                                        # Biểu đồ cột
                                        with tab1:
                                            fig_bar = go.Figure()
                                            fig_bar.add_trace(go.Bar(
                                                x=df_filtered['Năm_Kỳ'],
                                                y=df_filtered[col],
                                                name=col,
                                                marker_color=random_color(),
                                                hovertemplate=f"{col}: %{{y:.2f}}<br>Quý: %{{x}}"
                                            ))
                                            fig_bar.update_layout(
                                                title=f"{col} - {symbol}",
                                                xaxis_title="Quý",
                                                yaxis_title="Giá trị (Tỷ đồng)",
                                                template="plotly_white",
                                                height=300,
                                                margin=dict(l=20, r=20, t=150, b=20)
                                            )
                                            st.plotly_chart(fig_bar, use_container_width=True, config=config, key=f"income_quarter_{symbol}_{col}_bar")
                                        
                                        # Biểu đồ CAGR
                                        with tab2:
                                            if df_filtered.shape[0] >= 2:
                                                df_sorted = df_filtered.sort_values(['Năm', 'Kỳ'])
                                                start_quarter = df_sorted['Năm_Kỳ'].iloc[0]
                                                start_val = df_sorted[col].iloc[0]
                                                if start_val != 0:
                                                    quarters = df_sorted['Năm_Kỳ']
                                                    cagr_values = []
                                                    for idx, (q, val) in enumerate(zip(quarters, df_sorted[col])):
                                                        period = idx / 4  # Mỗi quý là 1/4 năm
                                                        if period == 0:
                                                            cagr_values.append(None)
                                                        else:
                                                            cagr_val = (val / start_val)**(1/period) - 1
                                                            cagr_values.append(cagr_val * 100)
                                                    fig_cagr = go.Figure()
                                                    fig_cagr.add_trace(go.Scatter(
                                                        x=quarters,
                                                        y=cagr_values,
                                                        mode='lines+markers',
                                                        name='CAGR',
                                                        marker_color='red',
                                                        hovertemplate="CAGR: %{y:.2f}%<br>Quý: %{x}"
                                                    ))
                                                    fig_cagr.update_layout(
                                                        title=f"CAGR của {col} - {symbol}",
                                                        xaxis_title="Quý",
                                                        yaxis_title="CAGR (%)",
                                                        template="plotly_white",
                                                        height=300,
                                                        margin=dict(l=20, r=20, t=150, b=20)
                                                    )
                                                    st.plotly_chart(fig_cagr, use_container_width=True, config=config, key=f"income_quarter_{symbol}_{col}_cagr")
                                                else:
                                                    st.info("Giá trị ban đầu bằng 0, không thể tính CAGR.")
                                            else:
                                                st.info("Không đủ dữ liệu để tính CAGR.")
                else:
                    st.warning(f"Không có dữ liệu hoặc cột 'Năm'/'Kỳ' cho báo cáo lãi lỗ hàng quý của {symbol}")

            # ----------------------- 7) BÁO CÁO LƯU CHUYỂN TIỀN TỆ (Hàng quý) -----------------------
            with st.expander("Báo cáo lưu chuyển tiền tệ (Hàng quý)"):
                cash_flow_data_quarter = get_financial_data(symbol, "cashflow", period="quarter")
                if not cash_flow_data_quarter.empty and 'Năm' in cash_flow_data_quarter.columns and 'Kỳ' in cash_flow_data_quarter.columns:
                    # Tạo cột mới kết hợp Năm và Kỳ
                    cash_flow_data_quarter['Năm_Kỳ'] = cash_flow_data_quarter['Năm'].astype(str) + " Q" + cash_flow_data_quarter['Kỳ'].astype(str)
                    cash_flow_data_quarter = cash_flow_data_quarter.sort_values(['Năm', 'Kỳ'])

                    # Chuyển vị DataFrame
                    df_cashflow_transposed_quarter = cash_flow_data_quarter.set_index('Năm_Kỳ').drop(columns=['Năm', 'Kỳ']).T
                    st.write("**Báo cáo lưu chuyển tiền tệ (Hàng quý):**")
                    st.dataframe(df_cashflow_transposed_quarter)

                    # ---------- Phần biểu đồ (tham chiếu dữ liệu gốc) ----------
                    numeric_cols = [
                        col for col in cash_flow_data_quarter.select_dtypes(include=['float64', 'int64']).columns
                        if col not in ['Năm', 'Kỳ', 'Năm_Kỳ']
                    ]
                    if numeric_cols:
                        selected_cols = st.multiselect(
                            f"Chọn các chỉ số để hiển thị biểu đồ (Báo cáo lưu chuyển hàng quý {symbol}):",
                            options=numeric_cols,
                            default=[]
                        )
                        available_quarters = sorted(cash_flow_data_quarter['Năm_Kỳ'].unique())
                        selected_quarters = st.multiselect(
                            f"Chọn quý hiển thị cho biểu đồ (Báo cáo lưu chuyển hàng quý {symbol}):",
                            options=available_quarters,
                            default=[]
                        )
                        df_filtered = (
                            cash_flow_data_quarter[cash_flow_data_quarter['Năm_Kỳ'].isin(selected_quarters)]
                            if selected_quarters else cash_flow_data_quarter
                        )

                        if selected_cols:
                            for i in range(0, len(selected_cols), 5):
                                cols = st.columns(5)
                                for j, col in enumerate(selected_cols[i:i+5]):
                                    with cols[j]:
                                        st.markdown(f"**{col}**")
                                        tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ CAGR"])
                                        
                                        # Biểu đồ cột
                                        with tab1:
                                            fig_bar = go.Figure()
                                            fig_bar.add_trace(go.Bar(
                                                x=df_filtered['Năm_Kỳ'],
                                                y=df_filtered[col],
                                                name=col,
                                                marker_color=random_color(),
                                                hovertemplate=f"{col}: %{{y:.2f}}<br>Quý: %{{x}}"
                                            ))
                                            fig_bar.update_layout(
                                                title=f"{col} - {symbol}",
                                                xaxis_title="Quý",
                                                yaxis_title="Giá trị (Tỷ đồng)",
                                                template="plotly_white",
                                                height=300,
                                                margin=dict(l=20, r=20, t=150, b=20)
                                            )
                                            st.plotly_chart(fig_bar, use_container_width=True, config=config, key=f"cashflow_quarter_{symbol}_{col}_bar")
                                        
                                        # Biểu đồ CAGR
                                        with tab2:
                                            if df_filtered.shape[0] >= 2:
                                                df_sorted = df_filtered.sort_values(['Năm', 'Kỳ'])
                                                start_quarter = df_sorted['Năm_Kỳ'].iloc[0]
                                                start_val = df_sorted[col].iloc[0]
                                                if start_val != 0:
                                                    quarters = df_sorted['Năm_Kỳ']
                                                    cagr_values = []
                                                    for idx, (q, val) in enumerate(zip(quarters, df_sorted[col])):
                                                        period = idx / 4  # Mỗi quý là 1/4 năm
                                                        if period == 0:
                                                            cagr_values.append(None)
                                                        else:
                                                            cagr_val = (val / start_val)**(1/period) - 1
                                                            cagr_values.append(cagr_val * 100)
                                                    fig_cagr = go.Figure()
                                                    fig_cagr.add_trace(go.Scatter(
                                                        x=quarters,
                                                        y=cagr_values,
                                                        mode='lines+markers',
                                                        name='CAGR',
                                                        marker_color='red',
                                                        hovertemplate="CAGR: %{y:.2f}%<br>Quý: %{x}"
                                                    ))
                                                    fig_cagr.update_layout(
                                                        title=f"CAGR của {col} - {symbol}",
                                                        xaxis_title="Quý",
                                                        yaxis_title="CAGR (%)",
                                                        template="plotly_white",
                                                        height=300,
                                                        margin=dict(l=20, r=20, t=150, b=20)
                                                    )
                                                    st.plotly_chart(fig_cagr, use_container_width=True, config=config, key=f"cashflow_quarter_{symbol}_{col}_cagr")
                                                else:
                                                    st.info("Giá trị ban đầu bằng 0, không thể tính CAGR.")
                                            else:
                                                st.info("Không đủ dữ liệu để tính CAGR.")
                else:
                    st.warning(f"Không có dữ liệu hoặc cột 'Năm'/'Kỳ' cho báo cáo lưu chuyển tiền tệ hàng quý của {symbol}")

            # ----------------------- 8) CHỈ SỐ TÀI CHÍNH (Hàng quý) -----------------------
            with st.expander("Chỉ số tài chính (Hàng quý)"):
                ratios_data_quarter = get_financial_data(symbol, "ratios", period="quarter")
                if not ratios_data_quarter.empty and 'Meta_Năm' in ratios_data_quarter.columns and 'Meta_Kỳ' in ratios_data_quarter.columns:
                    # Tạo cột mới kết hợp Meta_Năm và Meta_Kỳ
                    ratios_data_quarter['Năm_Kỳ'] = ratios_data_quarter['Meta_Năm'].astype(str) + " Q" + ratios_data_quarter['Meta_Kỳ'].astype(str)
                    ratios_data_quarter = ratios_data_quarter.sort_values(['Meta_Năm', 'Meta_Kỳ'])
            
                    # Chuyển vị DataFrame (bỏ cột Meta_Năm và Meta_Kỳ)
                    df_ratios_transposed_quarter = ratios_data_quarter.set_index('Năm_Kỳ').drop(columns=['Meta_Năm', 'Meta_Kỳ']).T
                    st.write("**Chỉ số tài chính (Hàng quý):**")
                    st.dataframe(df_ratios_transposed_quarter)
            
                    # ---------- Phần biểu đồ (tham chiếu dữ liệu gốc) ----------
                    numeric_cols = [
                        col for col in ratios_data_quarter.select_dtypes(include=['float64', 'int64']).columns
                        if col not in ['Meta_Năm', 'Meta_Kỳ', 'Năm_Kỳ']
                    ]
                    if numeric_cols:
                        selected_cols = st.multiselect(
                            f"Chọn các chỉ số để hiển thị biểu đồ (Chỉ số tài chính hàng quý {symbol}):",
                            options=numeric_cols,
                            default=[]
                        )
                        available_quarters = sorted(ratios_data_quarter['Năm_Kỳ'].unique())
                        selected_quarters = st.multiselect(
                            f"Chọn quý hiển thị cho biểu đồ (Chỉ số tài chính hàng quý {symbol}):",
                            options=available_quarters,
                            default=[]
                        )
                        df_filtered = (
                            ratios_data_quarter[ratios_data_quarter['Năm_Kỳ'].isin(selected_quarters)]
                            if selected_quarters else ratios_data_quarter
                        )
            
                        if selected_cols:
                            for i in range(0, len(selected_cols), 5):
                                cols = st.columns(5)
                                for j, col in enumerate(selected_cols[i:i+5]):
                                    with cols[j]:
                                        st.markdown(f"**{col}**")
                                        tab1, tab2 = st.tabs(["Biểu đồ cột", "Biểu đồ CAGR"])
            
                                        # Biểu đồ cột
                                        with tab1:
                                            fig_bar = go.Figure()
                                            fig_bar.add_trace(go.Bar(
                                                x=df_filtered['Năm_Kỳ'],
                                                y=df_filtered[col],
                                                name=col,
                                                marker_color=random_color(),
                                                hovertemplate=f"{col}: %{{y:.2f}}<br>Quý: %{{x}}"
                                            ))
                                            fig_bar.update_layout(
                                                title=f"{col} - {symbol}",
                                                xaxis_title="Quý",
                                                yaxis_title="Giá trị",
                                                template="plotly_white",
                                                height=300,
                                                margin=dict(l=20, r=20, t=150, b=20)
                                            )
                                            st.plotly_chart(fig_bar, use_container_width=True, config=config, key=f"ratios_quarter_{symbol}_{col}_bar")
            
                                        # Biểu đồ CAGR
                                        with tab2:
                                            if df_filtered.shape[0] >= 2:
                                                df_sorted = df_filtered.sort_values(['Meta_Năm', 'Meta_Kỳ'])
                                                start_quarter = df_sorted['Năm_Kỳ'].iloc[0]
                                                start_val = df_sorted[col].iloc[0]
                                                if start_val != 0:
                                                    quarters = df_sorted['Năm_Kỳ']
                                                    cagr_values = []
                                                    for idx, (q, val) in enumerate(zip(quarters, df_sorted[col])):
                                                        period = idx / 4.0  # Mỗi quý là 1/4 năm
                                                        if period == 0:
                                                            cagr_values.append(None)
                                                        else:
                                                            cagr_val = (val / start_val)**(1/period) - 1
                                                            cagr_values.append(cagr_val * 100)
                                                    fig_cagr = go.Figure()
                                                    fig_cagr.add_trace(go.Scatter(
                                                        x=quarters,
                                                        y=cagr_values,
                                                        mode='lines+markers',
                                                        name='CAGR',
                                                        marker_color='red',
                                                        hovertemplate="CAGR: %{y:.2f}%<br>Quý: %{x}"
                                                    ))
                                                    fig_cagr.update_layout(
                                                        title=f"CAGR của {col} - {symbol}",
                                                        xaxis_title="Quý",
                                                        yaxis_title="CAGR (%)",
                                                        template="plotly_white",
                                                        height=300,
                                                        margin=dict(l=20, r=20, t=150, b=20)
                                                    )
                                                    st.plotly_chart(fig_cagr, use_container_width=True, config=config, key=f"ratios_quarter_{symbol}_{col}_cagr")
                                                else:
                                                    st.info("Giá trị ban đầu bằng 0, không thể tính CAGR.")
                                            else:
                                                st.info("Không đủ dữ liệu để tính CAGR.")
                else:
                    st.warning(f"Không có dữ liệu hoặc cột 'Meta_Năm'/'Meta_Kỳ' cho chỉ số tài chính hàng quý của {symbol}")


####Tab 8: Phân tích kỹ thuật
with tab8:
    st.header("Phân tích kỹ thuật")

    # **Chọn mã cổ phiếu**
    stock_symbol = st.text_input("Nhập mã cổ phiếu (ví dụ: VCI)", value="VCI").upper()

    # **Chọn khoảng thời gian**
    start_date = st.date_input("Chọn ngày bắt đầu", value=datetime.datetime(2020, 1, 1))
    end_date = st.date_input("Chọn ngày kết thúc", value=datetime.datetime.now())

    # **Lấy dữ liệu từ vnstock**
    try:
        stock = Vnstock().stock(symbol=stock_symbol, source='VCI')
        stock_data = stock.quote.history(start=start_date.strftime('%Y-%m-%d'),
                                         end=end_date.strftime('%Y-%m-%d'))
        if stock_data.empty:
            st.error(f"Không có dữ liệu cho mã {stock_symbol} trong khoảng thời gian đã chọn.")
            st.stop()
        stock_data['time'] = pd.to_datetime(stock_data['time'])
        stock_data = stock_data.sort_values('time')
        st.success(f"Đã tải dữ liệu cho mã {stock_symbol} từ {start_date} đến {end_date}.")
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        st.stop()

    # **Chọn chỉ báo kỹ thuật**
    indicators = st.multiselect(
        "Chọn chỉ báo kỹ thuật",
        [
            "SMA (Đường trung bình động đơn giản)", 
            "EMA (Đường trung bình động hàm mũ)", 
            "RSI (Chỉ số sức mạnh tương đối)", 
            "MACD", 
            "Bollinger Bands",
            "Stochastic Oscillator",
            "CCI (Commodity Channel Index)",
            "ADX (Average Directional Index)",
            "DMI"
        ]
    )

    # Nhập khoảng thời gian cho các chỉ báo nếu được chọn
    if "SMA (Đường trung bình động đơn giản)" in indicators:
        sma_period = st.number_input("Chọn khoảng thời gian cho SMA", min_value=1, max_value=200, value=50)
    if "EMA (Đường trung bình động hàm mũ)" in indicators:
        ema_period = st.number_input("Chọn khoảng thời gian cho EMA", min_value=1, max_value=200, value=50)
    if "RSI (Chỉ số sức mạnh tương đối)" in indicators:
        rsi_period = st.number_input("Chọn khoảng thời gian cho RSI", min_value=1, max_value=100, value=14)
    if "Bollinger Bands" in indicators:
        bb_period = st.number_input("Chọn khoảng thời gian cho Bollinger Bands", min_value=1, max_value=200, value=20)
    if "Stochastic Oscillator" in indicators:
        stoch_period = st.number_input("Chọn khoảng thời gian cho Stochastic Oscillator", min_value=1, max_value=100, value=14)
    if "CCI (Commodity Channel Index)" in indicators:
        cci_period = st.number_input("Chọn khoảng thời gian cho CCI", min_value=1, max_value=200, value=20)
    if "ADX (Average Directional Index)" in indicators:
        adx_period = st.number_input("Chọn khoảng thời gian cho ADX", min_value=1, max_value=100, value=14)
    if "DMI" in indicators:
        dmi_period = st.number_input("Chọn khoảng thời gian cho DMI", min_value=1, max_value=100, value=14)

    # Hàm tính toán các chỉ báo kỹ thuật
    def compute_indicators():
        global stock_data
        if "SMA (Đường trung bình động đơn giản)" in indicators:
            stock_data['SMA'] = stock_data['close'].rolling(window=sma_period).mean()
        if "EMA (Đường trung bình động hàm mũ)" in indicators:
            stock_data['EMA'] = stock_data['close'].ewm(span=ema_period, adjust=False).mean()
        if "RSI (Chỉ số sức mạnh tương đối)" in indicators:
            delta = stock_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            stock_data['RSI'] = 100 - (100 / (1 + rs))
        if "MACD" in indicators:
            stock_data['EMA12'] = stock_data['close'].ewm(span=12, adjust=False).mean()
            stock_data['EMA26'] = stock_data['close'].ewm(span=26, adjust=False).mean()
            stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
            stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
        if "Bollinger Bands" in indicators:
            stock_data['Middle_Band'] = stock_data['close'].rolling(window=bb_period).mean()
            stock_data['Upper_Band'] = stock_data['Middle_Band'] + 2 * stock_data['close'].rolling(window=bb_period).std()
            stock_data['Lower_Band'] = stock_data['Middle_Band'] - 2 * stock_data['close'].rolling(window=bb_period).std()
        if "Stochastic Oscillator" in indicators:
            low_min = stock_data['low'].rolling(window=stoch_period).min()
            high_max = stock_data['high'].rolling(window=stoch_period).max()
            stock_data['%K'] = (stock_data['close'] - low_min) / (high_max - low_min) * 100
            stock_data['%D'] = stock_data['%K'].rolling(window=3).mean()
        if "CCI (Commodity Channel Index)" in indicators:
            tp = (stock_data['high'] + stock_data['low'] + stock_data['close']) / 3
            sma_tp = tp.rolling(window=cci_period).mean()
            mad = tp.rolling(window=cci_period).apply(lambda x: np.fabs(x - x.mean()).mean())
            stock_data['CCI'] = (tp - sma_tp) / (0.015 * mad)
        if "ADX (Average Directional Index)" in indicators:
            high = stock_data['high']
            low = stock_data['low']
            close = stock_data['close']
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=adx_period).mean()
            up_move = high - high.shift()
            down_move = low.shift() - low
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=adx_period).sum() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=adx_period).sum() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            stock_data['ADX'] = dx.rolling(window=adx_period).mean()
        if "DMI" in indicators:
            high = stock_data['high']
            low = stock_data['low']
            close = stock_data['close']
            up_move = high.diff()
            down_move = -low.diff()
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=dmi_period).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=dmi_period).sum() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=dmi_period).sum() / atr)
            stock_data['+DI'] = plus_di
            stock_data['-DI'] = minus_di

    # Chạy tính toán chỉ báo trong một tiến trình riêng
    indicator_thread = threading.Thread(target=compute_indicators)
    indicator_thread.start()
    indicator_thread.join()  # Chờ tiến trình tính toán hoàn thành

    # **Tạo biểu đồ với khối lượng có trục Y phụ**
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                        row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{}]])

    # **Thêm biểu đồ nến vào hàng trên (trục Y chính)**
    fig.add_trace(go.Candlestick(
        x=stock_data['time'],
        open=stock_data['open'],
        high=stock_data['high'],
        low=stock_data['low'],
        close=stock_data['close'],
        name="Nến"
    ), row=1, col=1, secondary_y=False)

    # **Thêm khối lượng giao dịch vào trục Y phụ**
    fig.add_trace(go.Bar(
        x=stock_data['time'],
        y=stock_data['volume'],
        name="Khối lượng",
        marker_color='blue',
        opacity=0.4
    ), row=1, col=1, secondary_y=True)

    # **Thêm các chỉ báo kỹ thuật vào biểu đồ hàng trên**
    if "SMA (Đường trung bình động đơn giản)" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['SMA'], 
                                 name=f"SMA {sma_period}", line=dict(color='orange')),
                      row=1, col=1, secondary_y=False)
    if "EMA (Đường trung bình động hàm mũ)" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['EMA'], 
                                 name=f"EMA {ema_period}", line=dict(color='green')),
                      row=1, col=1, secondary_y=False)
    if "Bollinger Bands" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['Upper_Band'], 
                                 name="Upper Band", line=dict(color='red')),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['Middle_Band'], 
                                 name="Middle Band", line=dict(color='purple')),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['Lower_Band'], 
                                 name="Lower Band", line=dict(color='red')),
                      row=1, col=1, secondary_y=False)

    # **Thêm các chỉ báo kỹ thuật vào hàng dưới**
    if "RSI (Chỉ số sức mạnh tương đối)" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['RSI'], 
                                 name="RSI", line=dict(color='purple')),
                      row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    if "MACD" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['MACD'], 
                                 name="MACD", line=dict(color='blue')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['Signal_Line'], 
                                 name="Signal Line", line=dict(color='red')),
                      row=2, col=1)
    if "Stochastic Oscillator" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['%K'], 
                                 name="Stochastic %K", line=dict(color='blue')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['%D'], 
                                 name="Stochastic %D", line=dict(color='orange')),
                      row=2, col=1)
    if "CCI (Commodity Channel Index)" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['CCI'], 
                                 name="CCI", line=dict(color='brown')),
                      row=2, col=1)
        fig.add_hline(y=100, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=-100, line_dash="dash", line_color="green", row=2, col=1)
    if "ADX (Average Directional Index)" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['ADX'], 
                                 name="ADX", line=dict(color='magenta')),
                      row=2, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="gray", row=2, col=1)
    if "DMI" in indicators:
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['+DI'], 
                                 name="+DI", line=dict(color='blue')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=stock_data['time'], y=stock_data['-DI'], 
                                 name="-DI", line=dict(color='red')),
                      row=2, col=1)

    # **Cập nhật giao diện**
    fig.update_layout(
        title=f"Phân tích kỹ thuật cho {stock_symbol} từ {start_date} đến {end_date}",
        height=800,
        showlegend=True,
        xaxis_title="Thời gian",
        yaxis_title="Giá",
        yaxis2=dict(title="Khối lượng", overlaying="y", side="right"),
        template="plotly_white",
        xaxis_rangeslider_visible=False
    )

    # **Hiển thị biểu đồ**
    st.plotly_chart(fig, use_container_width=True)
###########################################
# Tab 9: Bảng giá giao dịch
###########################################
with tab9:
    st.header("Bảng giá giao dịch")
    if 'symbols' not in st.session_state:
        st.error("Vui lòng nhập mã cổ phiếu ở tab 'Tải dữ liệu cổ phiếu' trước.")
    else:
        symbols = st.session_state['symbols']
        try:
            # Khởi tạo đối tượng stock dựa trên mã đầu tiên trong danh sách và source 'VCI'
            stock_obj = Vnstock().stock(symbol=symbols[0], source='VCI')
            # Gọi phương thức price_board với danh sách các mã từ Tab 1
            price_board = stock_obj.trading.price_board(symbols)
            if isinstance(price_board, pd.DataFrame):
                st.dataframe(price_board)
            else:
                st.write(price_board)
        except Exception as e:
            st.error(f"Lỗi khi tải bảng giá giao dịch: {e}")
