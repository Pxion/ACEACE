from datetime import datetime, timedelta
from utils import up_down_api
import streamlit as st
import pandas as pd
import pyodbc
import bcrypt

import sys

import torch

# sys.path.append(r'D:\互联网+，大创\大创--多智能体市场分析\FinRL-master\examples')
# sys.path.append(r"D:\互联网+，大创\大创--多智能体市场分析\FinRL-master")
from Stock_NeurIPS2018_3_Backtest import backtest_stock_trading

torch.classes.__path__ = []



# # ---------------------------
# # 数据库和数据文件初始化
# # ---------------------------
# # SQL Server 连接信息（请根据实际情况修改）
# server = 'localhost'
# database = 'test'
# sql_username = 'a'
# sql_password = '1'
# driver = '{ODBC Driver 17 for SQL Server}'
#
# # 建立数据库连接
# conn = pyodbc.connect(
#     f'DRIVER={driver};SERVER={server};DATABASE={database};UID={sql_username};PWD={sql_password}'
# )
@st.cache_data
def init_data():
    # 读取 CSV 数据，并将 Date 列转换为日期格式
    df_dji = pd.read_csv("data/^DJI_new.csv")
    df_dji["Date"] = pd.to_datetime(df_dji["Date"])

    df_sh = pd.read_csv("data/000001.SS_new.csv")
    df_sh["Date"] = pd.to_datetime(df_sh["Date"])

    # 获取dji及上证最新价格和涨跌幅

    dji_price, dji_change = up_down_api.get_stock_info("道琼斯指数")
    sh_price, sh_change = up_down_api.get_stock_info("上证指数")
    return df_dji, df_dji,dji_price, dji_change, sh_price, sh_change


df_dji, df_sh,dji_price, dji_change, sh_price, sh_change = init_data()

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)




# ---------------------------
# 数据库操作相关函数
# ---------------------------
# def verify_login(username, password) :
#     """从数据库中验证用户名和密码"""
#     cursor = conn.cursor()
#     cursor.execute("SELECT password, role, active FROM Users WHERE username = ?", username)
#     row = cursor.fetchone()
#     if row :
#         stored_password, role, active = row
#         # active 为 1 表示激活状态
#         if active and bcrypt.checkpw(password.encode(), stored_password.encode()) :
#             return True, role
#     return False, None
#
# def delete_user(username):
#     cursor = conn.cursor()
#     cursor.execute("DELETE FROM Users WHERE username = ?", username)
#     conn.commit()
#     return f"用户 {username} 已删除!"
#
# def change_password(username, new_password):
#     cursor = conn.cursor()
#     cursor.execute("SELECT password FROM Users WHERE username = ?", username)
#     row = cursor.fetchone()
#     if row :
#         hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
#         cursor.execute("UPDATE Users SET password = ? WHERE username = ?", hashed, username)
#         conn.commit()
#         return f"密码已更新!"
#     return f"用户 {username} 不存在!"
#
# def get_user_list() :
#     """获取用户列表，返回 DataFrame"""
#     cursor = conn.cursor()
#     cursor.execute("SELECT username, role, active FROM Users")
#     rows = cursor.fetchall()
#     data = []
#     for username, role, active in rows :
#         status = "激活" if active else "禁用"
#         data.append({"用户名" : username, "角色" : role, "状态" : status})
#     return pd.DataFrame(data)
#
#
# def add_user(name, pwd, role) :
#     """向数据库添加新用户"""
#     cursor = conn.cursor()
#     cursor.execute("SELECT username FROM Users WHERE username = ?", name)
#     if cursor.fetchone() :
#         return f"用户 {name} 已存在!"
#     hashed = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()
#     cursor.execute(
#         "INSERT INTO Users (username, password, role, active) VALUES (?, ?, ?, ?)",
#         name, hashed, role, 1
#     )
#     conn.commit()
#     return f"用户 {name} 添加成功!"

# ---------------------------
# 初始化 Session 状态
# ---------------------------
if 'logged_in' not in st.session_state :
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.session_state['role'] = None

# ---------------------------
# 登录界面
# ---------------------------
if not st.session_state['logged_in'] :
    st.title("用户登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("登录") :
        valid, role = True, "user"
        if valid :
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['role'] = role
            st.success(f"登录成功，欢迎 {username}（{role}）！")
            st.rerun()  # 登录成功后重运行，切换界面
        else :
            st.error("登录失败，请检查用户名或密码")
else :
    # ---------------------------
    # 主界面（登录后）
    # ---------------------------
    st.sidebar.write(f"欢迎 {st.session_state['username']} ({st.session_state['role']})")
    if st.sidebar.button("登出") :
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['role'] = None
        st.rerun()  # 登出后重运行，返回登录界面

    # 如果涨跌幅为正，则用红色显示，否则用绿色显示（包括价格）
    # 如果涨跌幅为正，则显示一个开心的emoji，否则显示一个难过的emoji
    def display_index(container, title, price, change) :
        color = "red" if change > 0 else "green"
        emoji = "😊" if change > 0 else "😞"
        container.header(title)
        container.markdown(f"<h1 style='color:{color};'>{price} {emoji}</h1>", unsafe_allow_html=True)
        container.header('当前涨跌幅:')
        container.markdown(f"<h1 style='color:{color};'>{change}% {emoji}</h1>", unsafe_allow_html=True)


    container_Dji = st.sidebar.container(border=True)
    display_index(container_Dji, '当前道琼斯指数:', dji_price, dji_change)

    container_sh = st.sidebar.container(border=True)
    display_index(container_sh, '当前上证指数:', sh_price, sh_change)

    st.title("AI 算法可视化应用")

    # 根据角色显示不同的选项卡（管理员额外有用户管理功能）
    tabs = st.tabs(["历史行情", "功能界面"] + (["用户管理"] if st.session_state['role'] == "admin" else []))

    # ---------------------------
    # 历史行情选项卡
    # ---------------------------
    with tabs[0] :
        st.header("历史行情")
        # 使用子选项卡区分道琼斯指数和上证指数
        sub_tabs = st.tabs(["道琼斯指数", "上证指数"])

        # 道琼斯指数页面
        with sub_tabs[0] :
            st.subheader("道琼斯指数")
            dji_max = df_dji["Date"].max().date()
            dji_min = dji_max-timedelta(days=365)
            # 日期范围选择
            today = datetime.today().date()
            ten_years_ago = datetime(2012, 1, 1).date()
            start_date,end_date = st.date_input("选择日期范围", value=(dji_min,dji_max),min_value=ten_years_ago, max_value=today, key="dji_date")

            # 过滤数据
            filtered_df_dji = df_dji[(df_dji["Date"].dt.date >= start_date) & (df_dji["Date"].dt.date <= end_date)]
            st.markdown("**收盘价**")
            st.line_chart(filtered_df_dji.set_index("Date")["Close"])
            st.markdown("**最高价**")
            st.line_chart(filtered_df_dji.set_index("Date")["High"])
            st.markdown("**最低价**")
            st.line_chart(filtered_df_dji.set_index("Date")["Low"])
            st.markdown("**开盘价**")
            st.line_chart(filtered_df_dji.set_index("Date")["Open"])
            st.markdown("**成交量**")
            st.line_chart(filtered_df_dji.set_index("Date")["Volume"])

        # 上证指数页面
        with sub_tabs[1] :
            st.subheader("上证指数")
            sh_max = df_sh["Date"].max().date()
            sh_min = sh_max-timedelta(days=365)
            start_date, end_date = st.date_input("选择日期范围", value=(sh_min, sh_max),min_value=ten_years_ago, max_value=today, key="sh_date")
            filtered_df_sh = df_sh[(df_sh["Date"].dt.date >= start_date) & (df_sh["Date"].dt.date <= end_date)]
            st.markdown("**收盘价**")
            st.line_chart(filtered_df_sh.set_index("Date")["Close"])
            st.markdown("**最高价**")
            st.line_chart(filtered_df_sh.set_index("Date")["High"])
            st.markdown("**最低价**")
            st.line_chart(filtered_df_sh.set_index("Date")["Low"])
            st.markdown("**开盘价**")
            st.line_chart(filtered_df_sh.set_index("Date")["Open"])
            st.markdown("**成交量**")
            st.line_chart(filtered_df_sh.set_index("Date")["Volume"])

    # ---------------------------
    # 功能界面选项卡
    # ---------------------------

    # 读取CSV文件并获取 'tic' 列的所有不重复值,按照日期先后进行排列
    train_data = pd.read_csv('model_data/train_data.csv')
    trade_data = pd.read_csv('model_data/trade_data.csv')
    ticker_data = pd.concat([train_data, trade_data], ignore_index=True)
    ticker_data['date'] = pd.to_datetime(ticker_data['date'])
    ticker_data = ticker_data.sort_values(by='date', ascending=False)
    unique_tic_values = ticker_data['tic'].unique().tolist()

    with tabs[1] :
        st.header("功能界面")
        st.write("欢迎使用 AI 算法可视化界面。")

        # 创建选择框可以手动选择 tic 值
        selected_tic = st.selectbox("选择股票", options=['全部'] + unique_tic_values,placeholder='全部')

        if selected_tic == '全部':
            filtered_data = ticker_data
        else:
            filtered_data = ticker_data[ticker_data['tic'] == selected_tic]

        st.write(f"{selected_tic}数据如下：")
        st.dataframe(filtered_data,width=800, height=600)

        # 选择模型
        models_to_use = st.multiselect("选择模型", options=['全部'] + ['a2c', 'ddpg', 'ppo', 'td3', 'sac'], default='全部')
        if '全部' in models_to_use:
            models_to_use = ['a2c', 'ddpg', 'ppo', 'td3', 'sac']
        # 选择初始资金
        initial_cash = st.number_input("请输入初始资金：", min_value=0.0, value=1000000.0, step=100.0)
        # 获取用户输入的 buy_cost_pct 和 sell_cost_pct 值
        buy_cost_pct = st.number_input("购买股票的百分比成本", min_value=0.0, max_value=1.0, value=0.001,step=0.0001, format="%.4f")
        sell_cost_pct = st.number_input("卖出股票的百分比成本", min_value=0.0, max_value=1.0, value=0.001, step=0.0001, format="%.4f")
        # 选择模型调用时间范围
        model_start_date,model_end_date = st.date_input("选择日期范围", value=(datetime(2021, 7, 1).date(),datetime(2021, 10, 29).date()),min_value=datetime(2009, 1, 1).date(), max_value=today)

        # 点击“开始训练”按钮，调用回测函数
        start_button = st.button("开始训练")
        if start_button:
            fig, result = backtest_stock_trading(
                trade_data_path="model_data/trade_data.csv",
                train_data_path="model_data/train_data.csv",
                models_to_use=models_to_use,
                initial_cash=initial_cash,
                buy_cost_pct=buy_cost_pct,
                sell_cost_pct=sell_cost_pct,
                start_date=model_start_date,
                end_date=model_end_date,
            )
            st.success("训练已开始！")
            # 显示模型训练结果
            st.dataframe(result, width=800, height=600)
            st.pyplot(fig)

    # ---------------------------
    # 用户管理选项卡（仅管理员可见）
    # ---------------------------
    # if st.session_state['role'] == "admin" :
    #     with tabs[2] :
    #         st.header("用户管理中心")
    #         df_users = get_user_list()
    #         st.table(df_users)
    #
    #         col1, col2 = st.columns(2)
    #
    #         with col1 :
    #             st.subheader("添加新用户")
    #             new_username = st.text_input("新用户名", key="new_username")
    #             new_password = st.text_input("密码", type="password", key="new_password")
    #             new_role = st.selectbox("角色", ["admin", "user"], key="new_role")
    #             if st.button("添加用户") :
    #                 msg = add_user(new_username, new_password, new_role)
    #                 st.success(msg)
    #                 st.rerun()
    #
    #         with col2 :
    #             st.subheader("删除用户")
    #             del_username = st.text_input("要删除的用户名", key="del_username")
    #             if st.button("删除") :
    #                 msg = delete_user(del_username)
    #                 st.warning(msg)
    #                 st.rerun()
    #
    #         st.subheader("修改密码")
    #         mod_username = st.text_input("用户名", key="mod_username")
    #         mod_password = st.text_input("新密码", type="password", key="mod_password")
    #         if st.button("修改密码") :
    #             msg = change_password(mod_username, mod_password)
    #             st.info(msg)
    #             st.rerun()
