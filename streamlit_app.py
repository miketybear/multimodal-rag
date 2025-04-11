import streamlit as st

chat_page = st.Page(
    "pages/02_Chat.py",
    title="Chat with DocuBot",
    icon="💬",
    default=True,
)


upload_page = st.Page(
    "pages/01_Document_Upload.py",
    title="Document Upload",
    icon="📄",
)

pg = st.navigation(
    pages=[chat_page, upload_page],
)


st.logo(
    image="assets/logo_BDPOC.png",
    size="large",
)

pg.run()