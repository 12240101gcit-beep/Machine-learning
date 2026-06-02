import sys

if __name__ == "__main__":
    # Use Streamlit's CLI main to run the existing streamlit_app.py
    try:
        from streamlit.web import cli as stcli
    except Exception:
        # Fallback for older streamlit versions
        from streamlit import cli as stcli
    sys.argv = ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.headless=true"]
    sys.exit(stcli.main())
