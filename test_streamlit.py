# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # 设置应用标题
# st.title("我的第一个 Streamlit 应用")

# # 显示文本
# st.write("这是一个简单的 Streamlit 应用示例")

# # 生成随机数据
# df = pd.DataFrame({
#     '列A': np.random.randn(100),
#     '列B': np.random.randn(100)
# })

# # 显示数据表
# st.write("数据表：", df.head())

# # 显示折线图
# st.line_chart(df)

# # 显示 Matplotlib 图表
# fig, ax = plt.subplots()
# ax.hist(df['列A'], bins=20)
# st.pyplot(fig)

# import streamlit as st

# st.title('Streamlit Info Example')

# # Display an info message
# st.info('This is an informational message.')

from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain