from datetime import timedelta

import streamlit as st
import pandas as pd
import pyodbc
import bcrypt


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

# 读取 CSV 数据，并将 Date 列转换为日期格式
df_dji = pd.read_csv("data/^DJI_new.csv")
df_dji["Date"] = pd.to_datetime(df_dji["Date"])

df_sh = pd.read_csv("data/000001.SS_new.csv")
df_sh["Date"] = pd.to_datetime(df_sh["Date"])



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
            start_date, end_date = st.date_input("选择日期范围", value=(dji_min, dji_max), key="dji_date")
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
            start_date, end_date = st.date_input("选择日期范围", value=(sh_min, sh_max), key="sh_date")
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
    with tabs[1] :
        st.header("功能界面")
        st.write("欢迎使用 AI 算法可视化界面。")

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